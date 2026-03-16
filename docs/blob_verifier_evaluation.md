# TrackNet + YOLO Blob Verifier 评估报告

## 概述

本文档记录 TrackNet + YOLO 二次验证 pipeline 的评估过程、结果和结论。

## Pipeline 数据流

```
原始视频帧 (1920×1080 BGR)
    ↓
OSD 遮罩 (frame[0:41, 0:603] = 0) → TrackNet 输入
    ↓
TrackNet.infer(masked_frames × 8) → heatmaps × 8
    ↓
BallTracker.process_heatmap_multi(heatmap, threshold=0.1, max_blobs=5)
    → blob 列表 [{pixel_x, pixel_y, blob_sum, blob_max, blob_area}, ...]
    ↓ (当 blob 数 > 1 时触发 YOLO)
extract_crops(原始未遮罩帧, blobs, crop_size=128)
    → 以每个 blob 的 pixel_x/pixel_y 为中心，从原始帧裁剪 128×128 crop
    → 边界处 zero-padding
    ↓
BlobVerifier.detect_crops(crops)   ← YOLO 输入: 128×128 crops
    → YOLO 在每个 crop 上检测 ball → bbox + confidence
    → 返回 [{yolo_conf, crop_cx, crop_cy}, None, ...]
    ↓
verify_blobs() 过滤 + 重排序
    → yolo_conf >= 0.15 的保留
    → 按 yolo_conf × blob_sum 排序
    → 全部被过滤时 fallback 到 TrackNet top-1
    ↓
Court-X 过滤 (homography 检查 world_x 范围)
    ↓
最终输出 top-1
```

**关键点**：YOLO 的输入是从原始未遮罩帧（raw_buffer）裁剪的 128×128 crop。blob 位置来自 TrackNet 热力图，但 crop 图像来自原始帧。

## YOLO 模型训练

### 训练数据

| 项目 | 值 |
|------|-----|
| 正样本来源 | 用户手工标注的 rectangle bbox（LabelMe 格式） |
| 负样本来源 | 从有 box 标注的帧中随机裁剪（距任何球 > 100px） |
| 正样本数 | 1393 |
| 负样本数 | 979 |
| 总样本数 | 2372 |
| 数据来源 | cam66 + cam68 两个摄像头的 2min 视频 |
| GT box 总数 | 1003（cam66: 507, cam68: 496） |
| 训练/验证比 | 80/20 |

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | yolo11n.pt | COCO 预训练 YOLO11 nano |
| 模式 | detect（非 cls） | 输出 bbox + confidence，可修正球心位置 |
| single_cls | True | 忽略 COCO 80 类，统一为 class 0 (ball) |
| imgsz | 128 | 匹配 crop 大小 |
| epochs | 50 | |
| batch | 32 | |
| freeze | 10 | 冻结前 10 层 backbone |
| device | CUDA (RTX 5070 Ti) | |

### 为什么用 detect 而非 cls

| | cls 分类 | detect 检测 |
|---|---|---|
| 输出 | ball 概率 | bbox + 置信度 |
| 定位修正 | 不能 | 可以重新定位球心 |
| 信息量 | 仅有/无 | 有/无 + 精确位置 + 置信度 |

### 为什么 128×128 crop 而非全图

YOLO 的感受野设计用于检测中等到大目标。网球在 1920×1080 全图中仅 5-15px，远小于 YOLO 的最小检测尺寸。零样本测试（COCO pretrained YOLO11n 全图推理）在 3000 帧中检出率为 0%。

裁剪 128×128 局部区域后，网球占 crop 面积的 ~2-5%，进入 YOLO 的有效检测范围。

### 负样本策略

负样本只从用户标了 box 的帧中取（`box_w is not None`），因为：
- 未标注帧可能包含未标记的球（死球等）
- 旧的 point 标注帧可能没有标全所有球
- 只有 box 标注帧才能确保用户审查过该帧的所有球位置

### 标注注意事项

- 优先标注不同场景（发球、底线、网前），避免连续帧
- 小球（远端 5-8px）和模糊球都要标
- 死球也标（提高 YOLO 召回率，死球过滤交给后续策略）
- 单帧看不确定是不是球的不标（避免模型学错）
- 类名用 "ball"（single_cls=True 下类名不影响训练）

### 训练输出

- 模型权重：`model_weight/blob_verifier_yolo.pt`
- 训练脚本：`python -m tools.train_blob_verifier --device 0`
- 数据生成：`python -m tools.prepare_yolo_crops uploads/cam66_20260307_173403_2min uploads/cam68_20260307_173403_2min`

## 评估结果

### 评估数据

- cam66: 3000 帧视频，466 GT 帧
- cam68: 3000 帧视频，447 GT 帧
- GT 包含 rectangle 和 point 标注（score=None, label="ball"）
- 评估脚本：`python -m tools.eval_tracknet_yolo`

### Test 1: TrackNet 原始 Recall

TrackNet 在 threshold=0.1 时能否"看到"GT 球。

| 指标 | cam66 | cam68 |
|------|-------|-------|
| GT 帧数 | 466 | 447 |
| 完全无检测 | 50 (10.7%) | 47 (10.5%) |
| Recall @top-1 | **63.3%** | **66.4%** |
| Recall @top-3 | **73.8%** | **73.8%** |
| Recall @top-5 | 73.8% | 74.0% |
| Top-1 均值误差 | 103.4px | 91.9px |
| Top-1 中位误差 | 4.3px | 4.6px |
| 多 blob 比例 | 72.8% | 56.8% |

**分析**：
- Top-1 到 top-3 有 ~10% 的 recall 提升，说明正确的球经常在 blob 列表里但不是第一个
- 10.5% 帧完全漏检（TrackNet 盲区，与 YOLO 无关）
- 中位误差 4-5px 说明检测准确时非常精确

### Test 2: YOLO 验证效果

多 blob 帧中 YOLO 是否帮助选出正确的球。

| 指标 | cam66 | cam68 |
|------|-------|-------|
| 多blob GT帧 | 303 | 227 |
| YOLO 改善 | 24 (7.9%) | 20 (8.8%) |
| YOLO 恶化 | 9 (3.0%) | 1 (0.4%) |
| 无变化 | 270 | 206 |
| Rescued (wrong→correct) | **21** | **15** |
| Killed (correct→wrong) | 4 | **0** |
| Raw recall (多blob) | 72.3% | 74.4% |
| YOLO recall (多blob) | **77.9%** | **81.1%** |

**分析**：
- YOLO 净正贡献：cam66 rescue 21 帧 vs kill 4 帧，cam68 rescue 15 帧 vs kill 0 帧
- 多 blob 帧 recall 提升 5-7 个百分点
- cam68 的 YOLO 表现尤其好（0 killed）

### Test 3: 稳定性 / 漂移

检测位置在帧间是否"跳跃"到错误位置。

| 指标 | cam66 | cam68 |
|------|-------|-------|
| 连续帧对数 | 2136 | 2031 |
| 跳跃 (>100px) | 184 | 183 |
| 真实跳跃 | 4 | 9 |
| 漂移事件 | 28 | 20 |
| 不可验证 | 152 | 154 |
| 稳定率 | **98.7%** | **99.0%** |
| 平均位移 | 54.5px | 51.6px |
| 中位位移 | 5.5px | 4.5px |

**分析**：
- 稳定率 98.7-99.0%，漂移事件仅 20-28 帧
- cam66 漂移集中在 frame 59-76（可能是特定场景问题）
- 大量"不可验证"跳跃（152-154）是因为这些帧没有 GT 标注

### Test 4: 死球污染

检测是否"卡"在死球位置不动。

| 指标 | cam66 | cam68 |
|------|-------|-------|
| 静止窗口（5帧, <8px） | 244 | 398 |
| 静止帧数 | 510 | 782 |
| 死球帧 | 46 (2.1%) | 48 (2.3%) |
| 正确静止 | 54 | 124 |

**分析**：
- 死球污染率仅 2.1-2.3%，影响不大
- 大量"正确静止"帧（球确实短暂静止，如发球前、落地瞬间）
- cam68 静止帧更多，可能与摄像机角度有关

## 结论

### YOLO Blob Verifier 效果

1. **有效**：多 blob 帧 recall 提升 5-7%，rescued >> killed
2. **安全**：几乎不会把正确的选择变错（cam68 killed=0）
3. **必要**：70% 帧有多 blob，verifier 解决了 blob 选择问题

### 当前瓶颈

1. **10.5% 完全漏检**：TrackNet 本身没检测到，YOLO 无法补救
2. **Top-1 vs Top-3 gap**：recall 从 74% (top-3) 降到 64% (top-1)，说明排序还有优化空间
3. **Mean error 高**：中位 4px 但均值 100px，少数帧严重选错

### 下一步方向

- **BallTracker（时序关联）**：利用帧间运动连续性锁定比赛球，解决漂移和漏检
- **扩充 GT 标注**：当前 913 GT 帧覆盖 3000 帧的 30%，更多 GT 可更准确评估
- **YOLO 模型迭代**：随着更多标注数据，可重新训练提升 verifier 精度
