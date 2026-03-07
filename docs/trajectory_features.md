# Tennis 3D Ball Tracking - 轨迹分析功能文档

## 1. 功能总览

本系统在双目三角测量得到 3D 球坐标的基础上，提供以下轨迹分析能力:

```
原始检测 (2 cameras × N detections)
  │
  ├─[Stage 0] 检测清洗 ──→ 去除噪声检测、场外检测、物理不合理检测
  │
  ├─[Stage 1] 自动时间偏移 + 插值三角测量 ──→ 3D 点云
  │
  ├─[Stage 2a] 回合分割 ──→ 按时间间隙和空间跳跃切割多个回合
  │
  ├─[Stage 2b] RANSAC 抛物线拟合 ──→ 鲁棒轨迹拟合 (逐回合)
  │
  ├─[分析1] 落地点检测 ──→ 球的第二次着地点 (Z=0)
  │
  ├─[分析2] 过网球速计算 ──→ 球经过网时的速度
  │
  └─[可视化] 3D 场景 + 2D 俯视图 ──→ Three.js 渲染
```

---

## 2. 检测清洗 (Stage 0)

**文件**: `app/trajectory.py` — `clean_detections()`

对每个相机的原始检测序列独立清洗，在三角测量之前去除噪声。

### 2.1 清洗流水线

```
原始检测 (frame_idx, pixel_x, pixel_y, confidence)
    │
    ├─[1] 置信度过滤  →  conf < 3.0 的检测被丢弃
    │    (blob 热力图总和，好的检测通常 > 5)
    │
    ├─[2] 球场边界检查  →  投影到世界坐标，超出球场 + 5m 余量则丢弃
    │    x ∈ [-5, 13.23],  y ∈ [-5, 28.77]
    │
    ├─[3] 速度一致性  →  世界坐标系中相邻检测速度 > 100 m/s 则丢弃跳跃点
    │    max_disp_per_frame = 100 / fps = 4.0 m/frame
    │
    └─[4] 孤立点移除  →  检测前后 ±10 帧内无邻居则丢弃
         (排除偶尔的误检)
```

### 2.2 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_confidence` | 3.0 | 最低 blob 置信度 |
| `court_margin` | 5.0 m | 球场外围余量 |
| `max_speed_world` | 100.0 m/s | 世界坐标最大合理速度 |
| `max_gap_frames` | 10 帧 | 孤立点判定阈值 (约 0.4s @ 25fps) |

### 2.3 返回值

返回清洗后的 3-tuple 列表 `(frame, px, py)` 和统计字典:

```python
{
    "input": 500,             # 原始检测数
    "after_confidence": 480,  # 置信度过滤后
    "after_bounds": 470,      # 边界过滤后
    "after_velocity": 465,    # 速度过滤后
    "after_isolation": 460,   # 孤立过滤后
    "output": 460,            # 最终输出
    "removed_confidence": 20,
    "removed_bounds": 10,
    "removed_velocity": 5,
    "removed_isolation": 5,
}
```

---

## 3. 自动时间偏移搜索 (Stage 1)

**文件**: `app/trajectory.py` — `find_offset_and_triangulate()`

两个相机不需要帧级同步。系统自动搜索最优时间偏移量 `dt`。

### 3.1 算法步骤

```
1. 粗搜索: 在 [-3s, +3s] 范围内等间隔采样 601 个偏移
2. 对每个偏移 dt:
   a. 对 cam_A 的每一帧 f_A，计算 cam_B 的对应帧 f_B = (t_A - dt) × fps_B
   b. 在 cam_B 中线性插值得到 f_B 处的像素坐标
   c. 两者投影到世界坐标后三角测量
   d. 用修剪均值 (trimmed mean) 评估射线距离
3. 精搜索: scipy.minimize_scalar 在 [best±0.2s] 范围内精细优化
4. 用最优 dt 三角测量所有匹配点
```

### 3.2 修剪均值 (Trimmed Mean)

传统方法使用 `mean(ray_distances)`，对异常值敏感。

本系统使用修剪均值:
- 对所有射线距离排序
- 丢弃最大的 20% (异常值)
- 对剩余 80% 取平均

```python
# trim_fraction = 0.2
ray_dists.sort()
n_keep = int(len(ray_dists) * 0.8)
cost = mean(ray_dists[:n_keep])
```

### 3.3 三角测量后过滤

三角测量得到的 3D 点还需经过:

| 过滤 | 条件 | 默认阈值 |
|------|------|----------|
| 射线距离 | `ray_dist > max_ray_dist` | 1.5 m |
| Z 范围 | `z < -0.5` 或 `z > 8.0` | 物理极限 |

---

## 4. 回合分割 (Rally Segmentation)

**文件**: `app/trajectory.py` — `segment_rallies()`

### 4.1 问题

球消失后（出界、落网、换球）到重新发球之间，球的检测点会产生跳跃。
如果不分割，整个轨迹会被当作一条连续曲线拟合，导致漂移。

### 4.2 分割逻辑

两个分割触发条件:

```
条件 A: 时间间隙
    相邻两个 3D 点的时间差 > 1.0 秒

条件 B: 空间跳跃
    相邻两点的时间差 > 0.2 秒，且
    3D 空间位移 / 时间差 > 80 m/s (物理上不可能)
```

满足任一条件即切割。段落长度 < 5 个点的段落被丢弃。

### 4.3 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_gap_seconds` | 1.0 s | 最大允许的时间间隙 |
| `min_rally_points` | 5 | 保留回合的最少点数 |
| 速度阈值 | 80 m/s | 空间跳跃的隐式速度阈值 |

### 4.4 前端展示

- 分割后的每个回合显示为标签按钮: `R1`, `R2`, `R3` ...
- 点击切换当前显示的回合
- 默认显示点数最多的回合 (主回合)

---

## 5. RANSAC 空间抛物线拟合 (Stage 2)

**文件**: `app/trajectory.py` — `fit_spatial_parabola_ransac()`

### 5.1 空间模型

抛物线拟合在空间中进行，**不依赖帧率/时间**:

```
X(Y) = ax · Y + bx          (水平偏移线性变化)
Z(Y) = az · Y² + bz · Y + cz  (重力导致的抛物线)
```

其中 Y 是球沿球场长轴的位置 (0→23.77m)。

物理约束:
- `az < 0` (重力使抛物线开口朝下)
- `az = -g / (2 · vy²)` → 可从 az 反推 vy

### 5.2 RANSAC 流程

```
for i in range(300):
    1. 随机抽取 4 个点 (X线性拟合需2个 + Z二次拟合需3个 = 最少4个)
    2. 拟合 fit_spatial_parabola(sample)
    3. 物理验证:
       - az 必须 < 0 (重力约束)
       - 推算球速 < 350 km/h
    4. 统计内点 (spatial_error < 0.5m)
    5. 更新最优模型

用最优模型的所有内点重新拟合
用重新拟合的模型再次识别内点 (可能捕获更多)
```

### 5.3 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_iterations` | 300 | RANSAC 迭代次数 |
| `inlier_threshold` | 0.5 m | 内点判定的空间误差 |
| `min_inlier_ratio` | 0.4 | 内点比例下限 |
| `max_speed_kmh` | 350 km/h | 物理速度上限 |

### 5.4 弹跳检测

使用 `_detect_bounce_robust()` 在内点集上检测弹跳:

```
1. 找 Z 的局部最小值 (z < 0.5m)
2. 验证: 前面 Z 更高，后面 Z 回升
3. 如果多个候选，选使两段拟合误差降低最多的分割点
4. 如果两段拟合比单段改善 > 0.05m，接受弹跳
```

弹跳后整体轨迹拆分为:
- pre-bounce: 弹跳前段 (从击球到落地)
- post-bounce: 弹跳后段 (反弹起来)

每段独立进行 RANSAC 拟合。

---

## 6. 过网球速计算

**文件**: `app/trajectory.py` — `_compute_speed_at_y()`, `_check_net_crossing()`

### 6.1 速度推导

从空间拟合系数推导球速:

```
已知:
    az = -g / (2 · vy²)

推导 vy (Y方向速度):
    vy = sqrt(g / (-2 · az))

X方向速度:
    dX/dY = ax  →  vx = ax · vy

Z方向速度:
    dZ/dY = 2·az·Y + bz  →  vz = vy · (2·az·Y + bz)

总速度:
    speed = vy · sqrt(1 + (dX/dY)² + (dZ/dY)²)
```

### 6.2 过网信息

在 `Y = 11.885m` (球网位置) 处计算:

```json
{
    "y": 11.885,
    "x": 4.12,           // 过网水平位置
    "z": 1.05,           // 过网高度 (m)
    "clears_net": true,   // 是否越过球网 (中心 0.914m)
    "speed_ms": 32.5,     // 速度 (m/s)
    "speed_kmh": 117.0    // 速度 (km/h)
}
```

### 6.3 前端展示

- 3D 场景中在网位置显示彩色球体:
  - 绿色 = 球过网
  - 红色 = 球未过网
- 球体旁显示速度标签: "117 km/h"
- 状态栏显示: "过网高度 1.05m | 球速 117 km/h"

---

## 7. 落地点检测

**文件**: `app/trajectory.py` — `_find_landing_point()`

### 7.1 算法

球的落地点是 Z(Y) = 0 的解:

```
az · Y² + bz · Y + cz = 0

判别式: disc = bz² - 4 · az · cz

两个根:
    Y₁ = (-bz + sqrt(disc)) / (2 · az)
    Y₂ = (-bz - sqrt(disc)) / (2 · az)
```

选择规则:
- 对分段轨迹: 使用 post-bounce 拟合, 选离弹跳点更远的根
- 对单段轨迹: 选离起点更远的根
- 根必须在 `[0, 23.77]` 范围内 (球场内)

### 7.2 返回值

```json
{
    "x": 6.5,    // 落地 X 坐标 (m)
    "y": 18.2,   // 落地 Y 坐标 (m)
    "z": 0.0     // Z = 0 (地面)
}
```

### 7.3 界内/界外判定

前端根据落地点坐标判定:

```javascript
// 单打场地边界
x ∈ [0, 8.23] 且 y ∈ [0, 23.77]  →  IN (界内)
否则                                  →  OUT (界外)
```

### 7.4 前端展示

- 3D 场景中显示橙色球体 + 扩散圆环
- 球体旁标注 "IN" (绿色) 或 "OUT" (红色)
- 状态栏显示落地坐标: "落点 (6.5, 18.2)"

---

## 8. 集成流程 (`app/orchestrator.py`)

### 8.1 `compute_3d_trajectory()` 完整流程

```python
# 1. 提取原始检测 (4-tuples)
raw_dets1 = [(frame, px, py, conf), ...]
raw_dets2 = [(frame, px, py, conf), ...]

# 2. 检测清洗 (每相机独立)
dets1, stats1 = clean_detections(raw_dets1, fps=25, H_i2w=H1)
dets2, stats2 = clean_detections(raw_dets2, fps=25, H_i2w=H2)

# 3. 自动偏移搜索 + 三角测量
best_dt, points_3d = find_offset_and_triangulate(
    dets1, dets2, fps, fps, H1, H2, cam1_pos, cam2_pos
)

# 4. 回合分割
rallies = segment_rallies(points_3d, fps=25, max_gap_seconds=1.0)

# 5. 逐回合 RANSAC 拟合
for rally_pts in rallies:
    traj = fit_trajectory(rally_pts)
    # traj 包含: type, smooth_curve, bounce_pos, net_crossing,
    #            landing_point, n_inliers, n_outliers
```

### 8.2 API 返回结构

```json
{
    "points": [...],           // 主回合的 3D 点
    "trajectory": {            // 主回合的轨迹拟合
        "type": "piecewise",   // "single" 或 "piecewise"
        "bounce_pos": {"x": 5.1, "y": 8.3, "z": 0.1},
        "pre_bounce": {...},   // 拟合系数
        "post_bounce": {...},  // 拟合系数
        "smooth_curve": [...], // 200个平滑点用于可视化
        "net_crossing": {      // 过网信息
            "z": 1.05,
            "clears_net": true,
            "speed_kmh": 117.0
        },
        "landing_point": {     // 落地点
            "x": 6.5, "y": 18.2, "z": 0.0
        },
        "n_inliers": 150,
        "n_outliers": 20
    },
    "rallies": [               // 所有回合
        {
            "rally_index": 0,
            "points": [...],
            "trajectory": {...}
        },
        ...
    ],
    "stats": {
        "cam66": {
            "raw_detections": 500,
            "cleaned_detections": 460,
            "cleaning": {...}
        },
        "cam68": {...},
        "matched_points": 400,
        "n_rallies": 3,
        "rally_sizes": [150, 120, 80],
        "time_offset_s": 0.52,
        "mean_ray_dist": 0.25,
        "n_inliers": 150,
        "n_outliers": 20
    }
}
```

---

## 9. 前端可视化

**文件**: `app/api/templates/dashboard.html`

### 9.1 3D 场景 (Three.js)

| 元素 | 颜色/形状 | 说明 |
|------|-----------|------|
| 球位置 | 黄色球体 | 三角测量得到的原始 3D 点 |
| 拟合曲线 | 绿色线条 | RANSAC 拟合的平滑曲线 |
| 弹跳点 | 红色球体 | Z 最低的转折点 |
| 过网球体 | 绿/红球体 | 绿=过网, 红=没过 |
| 过网速度 | 白色文字 | "XXX km/h" 标签 |
| 落地点 | 橙色球体+圆环 | 球第二次着地位置 |
| IN/OUT 标签 | 绿/红文字 | 界内/界外判定 |

### 9.2 回合选择器

```
[R1] [R2] [R3]    ← 点击切换当前显示的回合
 ↑
当前选中 (高亮)
```

### 9.3 X 轴镜像

由于物理观测与坐标系的映射差异，前端对所有 X 坐标应用镜像:

```javascript
// COURT_W = 8.23 (球场宽度)
display_x = COURT_W - world_x
```

影响范围:
- 3D 球体位置
- 轨迹线和平滑曲线
- 弹跳点、过网点、落地点
- 2D 俯视图球位置
- 场地侧边标签 (玻璃/窗帘)

### 9.4 状态面板

显示轨迹分析汇总:

```
轨迹类型: piecewise (弹跳) | 内点/外点: 150/20
过网高度: 1.05m | 球速: 117 km/h | 过网: YES
落点: (6.50, 18.20) | 回合: 3 个 (150, 120, 80 点)
```

---

## 10. 典型数值参考

### 10.1 网球球速范围

| 击球类型 | 速度范围 |
|---------|---------|
| 一发 (职业) | 180-250 km/h |
| 二发 (职业) | 130-180 km/h |
| 正手抽球 | 100-160 km/h |
| 反手切削 | 60-100 km/h |
| 业余击球 | 50-120 km/h |

### 10.2 合理性校验

| 指标 | 合理范围 | 异常范围 |
|------|---------|---------|
| 过网高度 | 0.914 - 4.0 m | < 0.5 或 > 6.0 m |
| 过网球速 | 50 - 250 km/h | > 350 km/h |
| 射线距离 (均值) | < 0.5 m | > 1.5 m |
| RANSAC 内点率 | > 60% | < 40% |
| 弹跳高度 | < 0.5 m | 弹跳 Z > 1.0 m |

---

## 11. 文件变更清单

| 文件 | 变更 |
|------|------|
| `app/trajectory.py` | 新增 `clean_detections()`, `_compute_speed_at_y()`, `_find_landing_point()`, `segment_rallies()`; 增强 `_check_net_crossing()` 加入球速; 增强 `_eval_offset()` 使用 trimmed mean; 新增 `fit_spatial_parabola_ransac()` 和 `_detect_bounce_robust()` |
| `app/orchestrator.py` | `compute_3d_trajectory()` 增加检测清洗、回合分割、逐回合拟合; 返回 `rallies` 数组 |
| `app/api/templates/dashboard.html` | 新增回合选择器、落地点可视化、球速标签、状态汇总面板; X 轴镜像修正 |
