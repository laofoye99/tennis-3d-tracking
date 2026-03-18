# 网球 3D 追踪系统 — 优化路线图

## 当前系统状态

### 已完成功能
| 模块 | 状态 | 关键指标 |
|------|------|----------|
| TrackNet 检测 | ✅ 成熟 | Raw recall 99.4%, <10px 精度 92.7% |
| 双相机 3D 重建 | ✅ 可用 | Homography + ray-crossing, 93.2% coverage |
| MultiBlobMatcher | ✅ 可用 | Top-2 blob + 时间连续性 + rank penalty |
| Bounce 检测 | ✅ 基础 | P=84.6% R=73.3% F1=78.6% (hybrid V+parabola) |
| 过网球速 | ✅ 基础 | 20-250 km/h 范围过滤 |
| 实时性 | ✅ 具备 | ~10ms/帧，25fps 余量充足 |
| GT 评估工具 | ✅ 可用 | eval_vs_gt.py, 支持 recall/pixel/bounce/coverage |

### 当前瓶颈
1. **无回合分割** — 不知道哪些帧属于一个回合，死球区间产生 FP
2. **Z 轴精度不足** — Homography 假设球在地面，远端 bounce z 偏移 0.3-0.6m
3. **无比赛统计** — 只有裸 bounce 和球速，没有比分、回合等结构化信息
4. **仅 1 段测试数据** — 参数可能过拟合

---

## Phase 1: 回合分割（Rally Segmentation）

### 目标
自动识别每个回合（point）的起止帧，消除非回合区间的 FP，为比赛统计提供基础。

### 预期收益
- Bounce precision 84.6% → 90%+（消除死球 FP）
- 解锁回合级统计（拍数、时长、球速分布）
- 为 serve 检测和比分追踪提供框架

### 方案

#### 1.1 基于 tracking density 的回合分割

**核心思路：** 回合期间球高速运动且持续被追踪（density 高），回合间球停止或不在场内（tracking 断裂或低速漂移）。

**实现位置：** 新建 `app/pipeline/rally_segmentation.py`

**算法：**

```
输入: smoothed_3d (平滑后的 3D 轨迹)
输出: list[Rally], 每个 Rally 包含 {start_frame, end_frame, type}

步骤:
1. 将 3D 轨迹按 gap > 30 帧（~1.2秒）分割为大段
2. 每个大段内，计算滑动窗口（20帧）的平均速度
3. 速度 > 5 m/s 的连续区间 = 回合活跃期
4. 活跃期前后扩展到最近的 tracking 断点 = 回合边界
5. 相邻回合如果间隔 < 60 帧（~2.4秒），合并（可能是换边接球）
```

**数据结构：**
```python
@dataclass
class Rally:
    start_frame: int
    end_frame: int
    serve_frame: Optional[int]  # 发球帧（如果检测到）
    bounces: list[dict]         # 该回合内的 bounce
    net_crossings: list[dict]   # 该回合内的过网
    winner_side: Optional[str]  # 'near' / 'far' / None
```

**参数：**
| 参数 | 值 | 说明 |
|------|-----|------|
| `min_rally_gap` | 30 帧 | 大于此间隔视为回合间断 |
| `min_speed` | 5 m/s | 低于此速度视为死球 |
| `speed_window` | 20 帧 | 速度计算窗口 |
| `merge_gap` | 60 帧 | 相邻回合合并阈值 |
| `min_rally_length` | 40 帧 | 过短的"回合"丢弃 |

#### 1.2 Serve 检测

**核心思路：** 发球的特征是：
1. 球从回合开始帧附近的一侧底线出发
2. 首次过网前球速通常 > 80 km/h
3. 发球后球落在对面发球区

**实现方法：**
- 在每个 Rally 的前 30 帧内，找第一个 net_crossing
- 如果该 crossing 的速度 > 60 km/h 且起点在底线附近（y < 2m 或 y > 21.77m）→ 标记为 serve
- serve 的落点（下一个 bounce）如果在发球区内 → 一发进
- 否则看第二个 bounce → 二发

**发球区定义（世界坐标）：**
```python
# 近端发球（y 从 0 增长方向）
SERVICE_BOX_NEAR_LEFT  = (0, NET_Y, COURT_W/2, SERVICE_FAR)    # 对角发球区
SERVICE_BOX_NEAR_RIGHT = (COURT_W/2, NET_Y, COURT_W, SERVICE_FAR)

# 远端发球（y 从 23.77 减小方向）
SERVICE_BOX_FAR_LEFT   = (0, SERVICE_NEAR, COURT_W/2, NET_Y)
SERVICE_BOX_FAR_RIGHT  = (COURT_W/2, SERVICE_NEAR, COURT_W, NET_Y)
```

#### 1.3 Bounce 过滤升级

在 `detect_bounces()` 中加入回合感知：
```python
# 只在 rally 活跃期内检测 bounce
bounces = [b for b in bounces if any(
    r.start_frame <= b["frame"] <= r.end_frame for r in rallies
)]
```

这一步直接消除死球 FP（frame 522 等）。

#### 1.4 验证方法
- 用 GT 的 serve/shot 标注验证 rally 边界是否正确
- GT 中有 5 个 serve、11 个 shot，可以检查是否都在检测到的 rally 内
- 计算 rally boundary 和 GT serve 的距离

---

## Phase 2: 比赛统计输出

### 目标
从原始 3D 轨迹 + bounce + rally 生成结构化比赛统计，可视化到视频和 JSON。

### 预期收益
- 系统从"技术 demo"变成"有用的分析工具"
- 为后续 UI/产品化提供数据基础

### 方案

#### 2.1 回合统计

**新建 `app/pipeline/match_stats.py`**

每个 Rally 计算：
```python
@dataclass
class RallyStats:
    rally_id: int
    start_frame: int
    end_frame: int
    duration_sec: float           # 回合时长
    num_shots: int                # 击球次数 = net_crossings 数
    num_bounces: int              # 弹跳次数
    serve_speed_kmh: float        # 发球速度
    avg_rally_speed_kmh: float    # 平均球速
    max_speed_kmh: float          # 最快球速
    last_bounce_in: bool          # 最后一个 bounce 是否 IN
    winner: Optional[str]         # 'near' / 'far'
```

**判断得分方：**
```
1. 最后一个 bounce 是 OUT → 击球方失分
2. 两次连续 bounce 在同一侧（没过网）→ 该侧失分
3. 其他情况暂不判断（需要更多信号）
```

#### 2.2 落点热力图

**方法：** 收集所有 bounce 的 (x, y) 世界坐标，按 serve/rally 分类：
```python
def generate_heatmap(bounces, court_img, resolution=0.1):
    """在球场图上绘制落点密度热力图。

    resolution: 每格 0.1m
    用 scipy.ndimage.gaussian_filter 平滑
    用 cv2.applyColorMap 上色
    """
```

#### 2.3 视频 HUD 增强

在 `render_video()` 中添加：

**顶部信息栏：**
```
Rally 3/8  |  Shot 5  |  2-1 (Near leads)
```

**minimap 增强：**
- 当前回合的轨迹（3D 投影到 2D）
- 发球落点用不同颜色标记
- 回合间显示统计卡片（时长、球速、得分方）

#### 2.4 JSON 输出

```python
def export_match_json(rallies, output_path):
    """导出完整比赛数据为 JSON。"""
    data = {
        "match_info": {
            "video_cam66": "...",
            "video_cam68": "...",
            "fps": 25,
            "total_frames": 1800,
        },
        "rallies": [
            {
                "id": 1,
                "start_frame": 118,
                "end_frame": 295,
                "serve": {"frame": 120, "speed_kmh": 145, "side": "near"},
                "bounces": [...],
                "net_crossings": [...],
                "stats": {...},
            },
            ...
        ],
        "summary": {
            "total_rallies": 8,
            "avg_rally_duration": 3.2,
            "max_serve_speed": 195,
            ...
        }
    }
```

---

## Phase 3: 相机标定升级

### 目标
从 Homography（平面映射）升级到完整的立体视觉标定，大幅提升 Z 轴精度。

### 当前问题
- Homography 将像素映射到**地平面**（z=0），隐含假设球在地面
- 球离地越高，从地平面反推的世界坐标偏差越大
- 两个相机的 ray 相交求 Z，但起点（地平面投影点）本身就偏了
- 实测：近端 bounce z ≈ 0.2-0.3m（合理），远端 bounce z ≈ 0.5-0.8m（偏高）

### 方案

#### 3.1 相机内参标定

**工具：** OpenCV `cv2.calibrateCamera()`

**步骤：**
1. 打印 9×6 棋盘格（每格 30mm）
2. 每个相机拍 15-20 张不同角度的棋盘格照片
3. 用 `cv2.findChessboardCorners()` 提取角点
4. `cv2.calibrateCamera()` 求解内参矩阵 K 和畸变系数

**输出：**
```python
# 每个相机
K = [[fx, 0, cx],      # 内参矩阵
     [0, fy, cy],
     [0,  0,  1]]
dist_coeffs = [k1, k2, p1, p2, k3]  # 畸变系数
```

**新建 `tools/calibrate_intrinsics.py`**

#### 3.2 相机外参标定（立体标定）

**方法 A：棋盘格立体标定（精度最高）**

两个相机同时拍棋盘格，用 `cv2.stereoCalibrate()` 求解相对位姿：
```python
retval, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
    objectPoints, imagePoints1, imagePoints2,
    K1, d1, K2, d2, imageSize,
    flags=cv2.CALIB_FIX_INTRINSIC
)
# R: 旋转矩阵 (3×3), T: 平移向量 (3×1)
```

**方法 B：球场关键点标定（不需要棋盘格）**

利用已有的球场关键点标注（12 个点，已在 src/ 中），用 `cv2.solvePnP()` 求每个相机的外参：
```python
# 球场 3D 关键点（已知世界坐标）
world_pts = np.array([
    [0, 0, 0],           # 近端左底线
    [8.23, 0, 0],        # 近端右底线
    [0, 23.77, 0],       # 远端左底线
    [8.23, 23.77, 0],    # 远端右底线
    # ... 共 12 个点
])

# 对应像素坐标（从 Labelme 标注读取）
img_pts = np.array([...])

# 求解外参
success, rvec, tvec = cv2.solvePnP(world_pts, img_pts, K, dist_coeffs)
R, _ = cv2.Rodrigues(rvec)  # 旋转矩阵
```

**推荐方法 B**——不需要额外拍棋盘格，直接用现有球场标注。内参用手机/相机规格估算（如果精度不够再用棋盘格）。

#### 3.3 三角化升级

**当前：** Homography 投影到地平面 → ray-crossing
**升级：** 像素坐标直接反投影为 3D 射线 → 标准三角化

**替换 `app/triangulation.py`：**
```python
def triangulate_stereo(px1, py1, px2, py2, P1, P2):
    """标准立体三角化。

    P1, P2: 3×4 投影矩阵 = K @ [R | t]
    """
    # 去畸变
    pt1 = cv2.undistortPoints(np.array([[[px1, py1]]]), K1, d1, P=K1)[0][0]
    pt2 = cv2.undistortPoints(np.array([[[px2, py2]]]), K2, d2, P=K2)[0][0]

    # DLT 三角化
    pts_4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2,1), pt2.reshape(2,1))
    pts_3d = pts_4d[:3] / pts_4d[3]  # 齐次→欧几里得

    return float(pts_3d[0]), float(pts_3d[1]), float(pts_3d[2])
```

**优势：**
- 不依赖地平面假设
- Z 精度直接由基线距离和角度决定（两相机距离 ~34m，基线很长）
- 预计 Z 精度从 ±0.5m 提升到 ±0.1-0.15m

#### 3.4 标定数据存储

```yaml
# config/stereo_calibration.yaml
cam66:
  K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  dist: [k1, k2, p1, p2, k3]
  R: [[...], [...], [...]]  # 世界坐标系 → 相机坐标系
  t: [tx, ty, tz]
  P: [[...], [...], [...]]  # 3×4 投影矩阵

cam68:
  K: [[...]]
  dist: [...]
  R: [[...]]
  t: [...]
  P: [[...]]
```

#### 3.5 验证方法
- 用 GT bounce 帧的 Z 值验证：理想 bounce 的 z ≈ 0-0.1m
- 对比当前 homography 方法和新方法在同一批 GT 上的 Z 误差分布
- 检查远端 bounce 的 Z 是否明显改善

---

## Phase 4: 泛化与鲁棒性

### 4.1 多视频验证

**步骤：**
1. 录制 3-5 段不同时间/光照的比赛视频（每段 2-5 分钟）
2. 标注每段的 bounce GT（至少 20 个 bounce/段）
3. 在不修改参数的情况下跑 eval，检查 F1 是否稳定
4. 如果 F1 波动大（>10%），分析哪些参数需要自适应

**重点关注：**
- 不同光照（上午/下午/阴天）下 TrackNet 的 recall
- 不同球速下的 Z 精度
- 远端 vs 近端的检测差异

#### 4.2 参数自适应

将当前的硬编码参数改为从数据自动推断：

```python
# 当前：硬编码
V_WINDOW = 8
MIN_BOUNCE_SPEED = 3.0

# 目标：从校准数据推断
V_WINDOW = int(fps * 0.32)             # 0.32 秒窗口
MIN_BOUNCE_SPEED = median_rally_speed * 0.15  # 回合平均速度的 15%
```

#### 4.3 异常处理

- 相机遮挡（人走过镜头）→ 检测到 tracking 突然全丢 → 暂停处理
- 多个球（训练时多球同时在场）→ MultiBlobMatcher 需要支持多目标
- 换场地 → 需要重新标定 homography/stereo，提供标定工具 UI

---

## 实施时间表

```
Phase 1: 回合分割          2-3 天
  ├─ 1.1 tracking density 分割    1 天
  ├─ 1.2 serve 检测               0.5 天
  ├─ 1.3 bounce 过滤升级          0.5 天
  └─ 1.4 验证与调参               0.5 天

Phase 2: 比赛统计          2-3 天
  ├─ 2.1 回合统计计算             0.5 天
  ├─ 2.2 落点热力图               0.5 天
  ├─ 2.3 视频 HUD 增强            1 天
  └─ 2.4 JSON 输出                0.5 天

Phase 3: 相机标定升级      3-5 天
  ├─ 3.1 内参标定工具             1 天
  ├─ 3.2 外参标定（solvePnP）     1 天
  ├─ 3.3 三角化模块替换           1 天
  └─ 3.4 验证与对比               1-2 天

Phase 4: 泛化验证          2-3 天
  ├─ 4.1 多视频录制与标注         1-2 天
  ├─ 4.2 参数自适应               0.5 天
  └─ 4.3 异常处理                 0.5 天
```

**总计：约 10-14 天**

---

## 文件变更预览

```
新建文件:
  app/pipeline/rally_segmentation.py   # Phase 1: 回合分割
  app/pipeline/match_stats.py          # Phase 2: 统计计算
  tools/calibrate_intrinsics.py        # Phase 3: 内参标定工具
  tools/calibrate_stereo.py            # Phase 3: 立体标定工具
  config/stereo_calibration.yaml       # Phase 3: 标定数据

修改文件:
  tools/render_tracking_video.py       # Phase 1-2: bounce 过滤 + HUD
  tools/eval_vs_gt.py                  # Phase 1: 加入 rally 评估
  app/triangulation.py                 # Phase 3: 三角化升级
  app/pipeline/homography.py           # Phase 3: 兼容新标定
  app/orchestrator.py                  # Phase 3: 使用新三角化
  config.yaml                          # Phase 3: 新标定参数
```

---

## 关键决策点

1. **Phase 3 标定方法选择：** 方法 B（球场关键点 solvePnP）不需要额外设备，建议先试。如果 Z 精度不够（>0.2m），再用棋盘格做方法 A。

2. **实时 vs 离线：** Phase 1-2 的回合分割/统计目前设计为离线（需要全段视频）。如果需要实时，需要改为滑动窗口 + 延迟确认模式（回合结束后 2-3 秒才能确认）。

3. **比分自动判断：** Phase 2.1 的得分判断只覆盖最简单的情况（OUT = 失分）。完整比分需要更多信号（击球未过网、球触网等），建议先做手动确认 + 自动建议。
