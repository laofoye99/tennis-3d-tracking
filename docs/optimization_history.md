# 优化历程记录

## 项目背景

双相机（cam66/cam68）室内单打网球场，基于 TrackNet V3 的 3D 球追踪系统。
目标：实时追踪比赛球、判断 bounce IN/OUT、输出球速和比赛统计。

---

## Phase 0: 基础 Pipeline 搭建

### 初始架构
```
TrackNet (3帧HRNet) → top-1 blob → Homography → 三角化 → 3D
```

### 问题
- TrackNet 用的是 HRNet ONNX（3帧输入），不是最优模型
- 只取 top-1 blob，错误检测无法修正
- 没有任何后处理（平滑、过滤、事件检测）

---

## Phase 1: 模型升级 + Top-K Blob 检测

### 改动
1. 模型从 HRNet (3帧) 升级为 **TrackNet V3 变体 (8帧+背景图)**
   - 输入: 8帧×3RGB + 1背景×3RGB = 27通道
   - 输出: 8帧 heatmap
   - Raw recall: 99.4%

2. `BallTracker.process_heatmap_multi()` 输出 top-K blobs（默认 K=2）

### 结果
- 检测率大幅提升，几乎每帧都能检测到球
- 但多 blob 引入了选择问题：哪个是比赛球？

---

## Phase 2: MultiBlobMatcher 跨相机匹配

### 思路
两个相机各自检测到 top-K blobs，用跨相机三角化找最佳配对：
```
score = ray_distance + blob_rank_penalty × (i+j) + temporal_weight × d3d
```

### 参数
- `blob_rank_penalty = 0.5`：非 top-1 blob 需要更好的 ray_dist 才会被选
- `temporal_weight = 0.3`：考虑时间连续性
- `max_ray_distance = 2.0`
- `lost_timeout = 30`

### 结果
| 指标 | top-1 only | MultiBlobMatcher |
|------|:---:|:---:|
| Recall | 96.4% | 79.5% |
| <10px 精度 | 95.8% | 95.5% |
| Jitter | 20.8px | 23.2px |

**结论：跨相机匹配反而降低了 recall，因为约束太强，很多帧被过滤掉了。**

---

## Phase 3: Hybrid 渲染策略

### 问题
MultiBlobMatcher 选的 blob 有时像素位置不对（选了 cam66 的 blob#2 而不是 blob#1），
导致渲染时球跳到错误位置。

### 解决方案
- 3D 重建用 MultiBlobMatcher（更好的 blob 配对）
- 视频渲染用 raw top-1 像素（更好的像素精度）

```python
if mode == "multi":
    points_3d, chosen_pixels, _ = triangulate_multi_blob(...)
    matched_frames = set(points_3d.keys())
    filt66 = {fi: det66[fi] for fi in matched_frames if fi in det66}  # raw top-1
```

### 结果
像素精度恢复到 92.7% <10px（从 MultiBlobMatcher 的 71.8% 恢复）

---

## Phase 4: Bounce 检测迭代

### V1: 基础 V-shape
在平滑后的 Z 轴轨迹上找局部最小值：
- 固定窗口 `window=8`
- 固定 margin 阈值 `v_margin=0.3`
- z_max = 1.5m

结果：很多 shot 被误判为 bounce（z=0.5-1.0m 的击球也有 V 形）

### V2: 两轮检测
- Strict: z_max=0.5m（真正的地面弹跳）
- Relaxed: z_max=0.8m（远端噪声大的弹跳）

结果：shot 误判减少，但仍有 FP

### V3: 非对称 V-shape
- 允许 V 形两侧不对称（快速回球时一侧 margin 小）
- 较强一侧 ≥ v_margin，较弱一侧 ≥ v_margin × 0.5

结果：捕获了更多真实 bounce（特别是快速回球后的弹跳）

### V4: 密集段 + 球场范围过滤
- 只在 ±20 帧内有 ≥20 个跟踪帧的密集段内检测 bounce
- bounce 的 3D 坐标必须在球场 ± 1m 范围内
- MIN_GAP = 8 帧

### 最终结果
P=84.6%, R=73.3%, F1=78.6% (11 TP, 2 FP, 4 FN)

### 遗留的 4 个 FN
- frame 32: 球滚动，无 V 形
- frame 788: 远端极度不对称
- frame 1146: 和 1141 只隔 5 帧，MIN_GAP 挡掉
- frame 1173: margin 在阈值边缘

---

## Phase 5: Viterbi 全局最优轨迹

### 动机
MultiBlobMatcher 逐帧贪心选择，一旦选错就错到底。
Viterbi 在所有帧的所有候选中找全局最优路径。

### 转移代价
```python
cost = speed_cost + z_cost + ray_cost + net_bonus + oscillation_penalty + static_penalty
```
- 过网奖励 -10.0（比赛球会过网，球鞋/废球不会）
- 振荡惩罚（来回在两个候选之间切换）
- 静止惩罚（几乎不动的点大概率是球鞋/反光）

### 结果
| 指标 | MultiBlobMatcher | Viterbi |
|------|:---:|:---:|
| Recall | 79.5% | 75.1% |
| Jitter | 23.2px | **31.5px** |
| Jump>30px | 10.4% | **15.9%** |

**结论：Viterbi 反而更差！全局优化在候选之间来回切换导致更大抖动。**

---

## Phase 6: 系统性方法对比（15种方法）

### 使用 eval_tracking.py 标准化评估框架

基于 473 个 match_ball GT 帧，系统对比了 15 种方法：

| 方法 | Recall | Prec | F1 | <10px | Jitter |
|------|:---:|:---:|:---:|:---:|:---:|
| **top1_conf20** | **91.5%** | **39.2%** | **54.9%** | **95.8%** | **18.0** |
| top1_confidence (p25) | 87.5% | 40.0% | 54.9% | 95.7% | 17.9 |
| top1_conf30 | 82.7% | 40.4% | 54.3% | 95.4% | 18.4 |
| conf_xcam | 77.0% | 40.7% | 53.3% | 95.3% | 17.1 |
| top1_xcam | 82.2% | 37.9% | 51.9% | 95.4% | 17.4 |
| multi_strict | 82.0% | 35.7% | 49.7% | 95.6% | 23.3 |
| top1 (baseline) | 96.4% | 33.1% | 49.3% | 95.8% | 20.8 |
| conf_xcam_gate | 55.0% | 43.9% | 48.8% | 95.8% | 14.5 |
| multi | 79.5% | 35.0% | 48.6% | 95.5% | 23.2 |
| viterbi | 75.1% | 33.6% | 46.5% | 95.8% | 31.5 |

### 核心发现
**简单的 heatmap 置信度过滤（去掉最弱 20%）全面优于任何复杂的跨相机匹配或全局优化算法。**

---

## Phase 7: 2D 像素平滑（30+种方法）

### 测试的方法
- Savitzky-Golay（窗口 5/7/9/11，阶数 2/3）→ ❌ 精度严重下降
- EMA（alpha 0.3/0.5/0.7/0.8）→ ❌ 精度下降
- Kalman 滤波（多种 Q/R 配置）→ ❌ 过度平滑
- 中值滤波（kernel 3/5/7）→ ✅ 保精度降抖动
- 双边滤波 → 🔶 精度好但抖动降不多
- 自适应 EMA → ❌ 精度下降

### 关键发现
**GT 本身的 jitter 是 17.8px（球真的在快速移动），jitter < 10px 物理上不可能。**

### 最优方案: conf20_median5
| 指标 | 目标 | 达成 |
|------|:---:|:---:|
| Recall | >90% | 90.7% ✅ |
| <10px | >90% | 93.7% ✅ |
| Jitter | 最小 | 16.3px ✅（低于 GT 17.8px）|

---

## Phase 8: Rally 分割模型

### 动机
Precision 天花板 ~40%，因为非比赛时段场上也有球被检测到。
只能通过回合分割过滤非比赛帧来突破。

### 方法
88 个特征（多尺度检测密度、像素速度、置信度、跨相机一致性、Fourier 周期性等），
训练 Random Forest + Gradient Boosting + BiLSTM 三模型 Ensemble。

### 结果
| 模型 | Accuracy | F1 (rally) |
|------|:---:|:---:|
| Random Forest | 94.9% | 0.903 |
| Gradient Boosting | 91.6% | 0.861 |
| BiLSTM | 90.4% | 0.813 |
| **Ensemble** | **95.5%** | **0.919** |

Rally recall 97%（几乎不漏），Precision 87%。

---

## Phase 9: 标定坐标系修正

### 发现的 Bug
球场 12 个标定关键点标注在**单打线**上（x=1.37, 6.86），
但代码将它们映射到了**双打线**坐标（x=0, 8.23）。

### 影响
- 所有 x 坐标被拉伸 1.5 倍（8.23/5.49）
- IN/OUT 判定偏移
- 三角化精度降低

### 修正
- `src/compute_homography.py`: SINGLES_LEFT=1.37, SINGLES_RIGHT=6.86
- `app/calibration.py`: 同步修正
- 重新生成 homography_matrices.json 和 camera_calibration.json

---

## Phase 10: 单相机 Homography IN/OUT 判定

### 动机
三角化 3D 坐标有 1.5m 误差（帧不同步导致），无法准确判断 IN/OUT。
但 bounce 瞬间球在地面（z≈0），homography 的地面假设完全成立。

### 方案
- bounce 在近端（y < 11.885）→ 用 cam66 homography（精度 ~5cm）
- bounce 在远端（y > 11.885）→ 用 cam68 homography（精度 ~3cm）
- 判断用哪个相机：根据最近一次过网方向

### 精度提升
三角化 IN/OUT: ±1.5m（不可用）
单相机 Homography: ±5cm（30 倍提升）

---

## Phase 11: Unity 合成数据

### 搭建
- 标准网球场 + 两个虚拟相机（匹配真实位置）
- 两个简化球员，完整对打模拟
- 球物理: 重力 + 空气阻力 + Magnus + 旋转弹跳
- 技能等级随机、击球类型随机（topspin/flat/slice/dropshot/winner）

### 数据
- 500,000 帧 / 19 分钟
- 91 rallies, 527 hits, 558 bounces

### 用途
训练 3D 轨迹事件分类器（不用于视觉模型，因为 domain gap）

---

## Phase 12: ML 事件分类器

### 尝试 1: CNN + Transformer (78K 参数)
- 在合成数据上训练
- 真实数据上 bounce recall=53.8%（比 V-shape 73.3% 差）

### 尝试 2: BiLSTM (4.1K 参数)
- 更适合小数据量
- 仍然受 domain gap 影响

### 结论
**ML 事件分类在当前数据量下投入产出比不高。V-shape 的 F1=78.6% 已经接近数据质量天花板（帧不同步和 Z 精度限制），而不是算法天花板。**

---

## 核心教训总结

| 教训 | 验证方式 |
|------|---------|
| 简单方法 > 复杂方法 | conf 过滤 > Viterbi > MultiBlobMatcher |
| 检测不是瓶颈 | 99.4% recall，换模型收益极小 |
| 精度瓶颈在硬件（帧不同步） | 1-2 帧偏移 = 1.5m 3D 误差 |
| bounce IN/OUT 不需要 3D | 单相机 homography 精度 30 倍于三角化 |
| GT 驱动的优化 > 直觉 | 每次改动用 eval_tracking.py 验证 |
| 中值滤波 > SG/Kalman/EMA | 保精度降抖动的最佳平衡 |
| dead_ball 不能和 match_ball 混训 | 物理动作一样，会污染分类器 |

---

## 当前最优 Pipeline (截至 2026-03-22)

```
TrackNet V3 (8帧+背景, 27ch)
  → top1_conf20 置信度过滤
  → Rally 分割 (Ensemble, 95.5%)
  → Homography 世界坐标
  → solvePnP 三角化 → 3D
  → SG 3D 平滑 + V-shape Bounce
  → 单相机 Homography IN/OUT
  → Median5 2D 像素平滑
  → 渲染 / 3D大屏 / JSON
```

## 下一步
1. 音频帧对齐（消除 1.5m 误差根因）
2. Layer 1+2 层级事件检测（需要更多标注数据）
3. YOLOv8-Pose 球员检测（区分 shot/bounce、rally/dead_ball）
4. 集成到实时 pipeline + 可视化 app
