# 网球 3D 追踪系统 — 技术全景文档

## 1. 系统概述

双摄像头网球 3D 追踪系统，从球场两端俯拍的 RTSP 摄像头实时/离线检测网球像素位置，经 YOLO 验证、单应性变换、三角测量重建出球的 3D 空间轨迹。

```
┌────────────────────────────────────────────────────────────────┐
│                       系统架构总览                               │
│                                                                │
│   Camera 66 (近端)              Camera 68 (远端)                │
│        │                              │                        │
│        ▼                              ▼                        │
│   ┌──────────┐                  ┌──────────┐                   │
│   │ TrackNet │  ← 8帧时序输入    │ TrackNet │                   │
│   │ 热力图检测 │                  │ 热力图检测 │                   │
│   └────┬─────┘                  └────┬─────┘                   │
│        ▼                              ▼                        │
│   ┌──────────┐                  ┌──────────┐                   │
│   │ YOLO     │  ← 每帧验证       │ YOLO     │                   │
│   │ Blob验证  │                  │ Blob验证  │                   │
│   └────┬─────┘                  └────┬─────┘                   │
│        ▼                              ▼                        │
│   ┌──────────┐                  ┌──────────┐                   │
│   │Homography│  像素→世界坐标     │Homography│                   │
│   └────┬─────┘                  └────┬─────┘                   │
│        │                              │                        │
│        └──────────┬───────────────────┘                        │
│                   ▼                                            │
│          ┌────────────────┐                                    │
│          │  三角测量 (3D)   │ ← ray_distance 一致性检查          │
│          │  交叉验证        │                                    │
│          └───────┬────────┘                                    │
│                  ▼                                             │
│          ┌────────────────┐                                    │
│          │ 轨迹分析/API    │ ← 落点、球速、回合分割              │
│          └────────────────┘                                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 检测阶段：TrackNet

### 模型架构

- **网络**: U-Net 编解码器 + skip connections
- **编码器**: Conv64 → Conv128 → Conv256 → Conv512（逐层下采样）
- **解码器**: 对称上采样 + skip connection 拼接
- **输出**: sigmoid 激活，每帧一个热力图

### 推理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| input_size | 288 × 512 | 模型输入分辨率 |
| frames_in | 8 | 输入帧数（时序窗口） |
| frames_out | 8 | 输出热力图数 |
| threshold | 0.1 | 热力图二值化阈值 |
| device | cuda | GPU 推理 |

### 时序特性

8帧输入使 TrackNet 具有时序感知能力：
- **运动中的比赛球**：连续帧中位置有序变化，热力图响应强
- **静止的死球**：帧间位置不变，热力图响应较弱（TrackNet 天然抑制）
- **但不保证过滤**：低响应不等于零响应，死球仍可能被检出

### 热力图后处理（BallTracker）

```
TrackNet 输出 heatmap (288×512, float)
    ↓
resize → 原始分辨率 (1920×1080)
    ↓
OSD 遮罩 (frame[0:41, 0:603] = 0) → 消除时间戳干扰
    ↓
threshold > 0.1 → 二值图
    ↓
connectedComponents (8-connectivity) → blob 列表
    ↓
每个 blob 计算:
  - pixel_x, pixel_y: 加权质心（权重=热力值）
  - blob_sum: 热力值总和（检测置信度）
  - blob_max: 热力值峰值
  - blob_area: 像素面积
    ↓
按 blob_sum 降序排列，返回 top-N (max_blobs=5)
```

### 代码位置

| 文件 | 功能 |
|------|------|
| `app/pipeline/tracknet.py` | TrackNet 模型定义 |
| `app/pipeline/inference.py` | 推理桥接（TrackNet/HRNet 工厂） |
| `app/pipeline/postprocess.py` | BallTracker 热力图后处理 |

---

## 3. 验证阶段：YOLO Blob Verifier

### 为什么需要二次验证

TrackNet 高召回率但误检多：~70% 帧产生多个 blob（死球、反光、场线交叉点）。需要高精度模块筛选真球。

### 工作流程

```
每帧的 blob 列表 (来自 TrackNet)
    ↓
extract_crops(原始未遮罩帧, blobs, crop_size=128)
    → 以每个 blob 的 pixel_x/pixel_y 为中心裁剪 128×128
    → 图像边界处 zero-padding
    ↓
BlobVerifier.detect_crops(crops)
    → YOLO 在每个 crop 上检测 ball
    → 返回 [{yolo_conf, crop_cx, crop_cy}, None, ...]
    ↓
过滤: yolo_conf >= 0.15 的保留
排序: 按 yolo_conf × blob_sum 综合评分
    ↓
全部被过滤 → fallback 到 TrackNet top-1
有通过的 → 返回排序后的列表
```

**关键设计**：
- **每帧都跑 YOLO**（不只是多 blob 帧），确保单 blob 误检也被过滤
- **YOLO 输入是原始帧的 crop**（非 TrackNet 的 masked 帧）
- **128×128 crop** 因为网球在全图仅 5-15px，YOLO 无法检测

### 为什么用 detect 模式而非 cls 分类

| | detect 检测 | cls 分类 |
|---|---|---|
| 输出 | bbox + 置信度 | 有/无概率 |
| 位置修正 | 可以重新定位球心 | 不能 |
| 参数 | single_cls=True（所有类合并为 ball） | — |

### 训练概要

| 项目 | 值 |
|------|-----|
| 基础模型 | yolo11n.pt (COCO 预训练) |
| 正样本 | 1393 crops（用户手标 rectangle bbox） |
| 负样本 | 979 crops（从标注帧随机裁剪背景区域） |
| 关键参数 | epochs=100, freeze=0, mosaic=0.0, single_cls=True |
| 最终指标 | mAP50=0.771, P=0.659, R=0.736 |
| 推理速度 | ~0.3ms/crop (RTX 5070 Ti) |

**负样本策略**：只从有 box 标注的帧取负样本，避免未标注帧的遗漏球污染负样本集。

### 代码位置

| 文件 | 功能 |
|------|------|
| `app/pipeline/blob_verifier.py` | 验证模块（extract_crops, detect_crops, verify_blobs） |
| `tools/prepare_yolo_crops.py` | 训练数据生成 |
| `tools/train_blob_verifier.py` | 模型训练脚本 |
| `model_weight/blob_verifier_yolo.pt` | 训练好的模型权重 |

---

## 4. 坐标变换：Homography

### 像素 ↔ 世界坐标

每个摄像头预计算一个 3×3 单应性矩阵，实现球场平面上的像素坐标与世界坐标（米）的双向转换。

```
pixel_to_world(px, py):
    pt = H_img2world @ [px, py, 1]
    return (pt[0]/pt[2], pt[1]/pt[2])  → (world_x, world_y) 米

world_to_pixel(wx, wy):
    pt = H_world2img @ [wx, wy, 1]
    return (pt[0]/pt[2], pt[1]/pt[2])  → (pixel_x, pixel_y)
```

### 标定来源

- 12 个球场关键点（LabelMe 标注）：四角、发球线交点、中线端点
- 标准 ITF 单打场地：23.77m × 8.23m
- 通过 `cv2.findHomography()` 求解
- 重投影误差 ~2cm

### Court-X 过滤

检测点经 homography 变换后，检查 `world_x` 是否在球场宽度范围内：
- 有效范围：[-1.0, 9.23]（含 1m 边距）
- 超出范围的 blob 直接丢弃（场外干扰）

### 世界坐标系

```
        x=0                x=4.115              x=8.23
        (左边线)            (中线)               (右边线)

y=23.77 ┼───────────────────┼────────────────────┤ 远端底线
        │                                        │
y=18.285┼───────────────────┼────────────────────┤ 远发球线
        │                                        │
y=11.885╪════════════════════════════════════════╡ 球网
        │                                        │
y=5.485 ┼───────────────────┼────────────────────┤ 近发球线
        │                                        │
y=0     ┼───────────────────┼────────────────────┤ 近端底线

Camera 66: y < 0 (近端后方, 高度 7m)
Camera 68: y > 23.77 (远端后方, 高度 7m)
z 轴: 垂直向上, z=0 = 地面
```

### 代码位置

| 文件 | 功能 |
|------|------|
| `app/pipeline/homography.py` | HomographyTransformer 类 |
| `app/calibration.py` | CameraCalibrator / StereoCalibrator |
| `src/homography_matrices.json` | 预计算矩阵存储 |
| `src/camera_calibration.json` | 相机标定参数 |

---

## 5. 3D 重建：三角测量

### 算法原理

两个摄像头各自通过 homography 得到球在地面平面的投影点 `(wx, wy, 0)`，从摄像头 3D 位置到该投影点构成一条射线。两条射线在空间中不一定相交（异面直线），但可以求出最近点对。

```
cam1_pos ─────→ ground_point_1 (x1, y1, 0)    射线 1
cam2_pos ─────→ ground_point_2 (x2, y2, 0)    射线 2
                      ↓
        求两条射线的最近点 p1, p2
        3D 球位置 = (p1 + p2) / 2
        ray_distance = ||p1 - p2||  ← 一致性度量
```

### ray_distance 的意义

- **ray_distance 小**（< 1m）：两个摄像头观测一致，3D 位置可信
- **ray_distance 大**（> 2m）：有至少一个摄像头误检
- 物理约束：有效 z 范围 [0, 6] 米

### MultiBlobMatcher

当两个摄像头各有多个 blob 候选时，需要匹配最佳配对：

1. 枚举所有 (blob_cam66, blob_cam68) 配对
2. 对每个配对执行三角测量，得到 ray_distance
3. 过滤：ray_distance < 2.0m 且 z ∈ [0, 6]m
4. 选择 ray_distance 最小的配对

### 代码位置

| 文件 | 功能 |
|------|------|
| `app/triangulation.py` | `triangulate()` 函数 |
| `app/multi_blob_matcher.py` | MultiBlobMatcher 类 |

---

## 6. 交叉验证策略

### 目的

当一个摄像头误检时，利用另一个摄像头的检测来纠正。

### 判断方法

```
两摄像头检测 → triangulate() → ray_distance
    ↓
ray_distance < 阈值 (1.5m) → AGREE → 两个检测一致，输出 3D 位置
ray_distance > 阈值        → DISAGREE → 至少一个误检
    ↓
DISAGREE 时判断信任谁:
    - YOLO 置信度更高的
    - 与前一帧位移更合理的（运动连续性）
    - blob_sum 更高的
```

### 交叉投影

利用 `world_to_pixel()` 可以将一个摄像头的检测投影到另一个摄像头视图中：

```
cam66 检测 pixel → pixel_to_world (cam66) → world_x, world_y
    → world_to_pixel (cam68) → cam68 上的预期位置
    → 与 cam68 实际检测位置对比
```

### 已知局限

- **两个摄像头同时检测到同一个死球**时 ray_distance 也会很小（AGREE 不等于比赛球）
- 需要结合帧间跳跃检测断开轨迹：位移过大时即使 AGREE 也不可信

### 代码位置

| 文件 | 功能 |
|------|------|
| `tools/eval_cross_camera.py` | 交叉验证评估脚本 |
| `tools/render_cross_camera_video.py` | 可视化渲染（双画面 + ray_distance 指示条） |

---

## 7. 轨迹分析

### 三阶段处理

**Stage 0: 单摄像头清洗**
- 过滤低置信度、出界检测
- 速度一致性检查（最大 100 m/s）
- 移除孤立点（与相邻帧间隔 > 10 帧）

**Stage 1: 自动时间偏移搜索**
- 两个摄像头帧号不对齐，需要搜索最优偏移量 dt
- 遍历 dt 范围，最小化修剪均值 ray_distance
- 偏移后执行三角测量 + 过滤

**Stage 2: RANSAC 空间拟合**
- X(Y) = 线性（水平方向均匀运动）
- Z(Y) = 二次（重力抛物线，g = -9.81 m/s²）
- 分段拟合（落地点分割不同弹跳段）

### 分析输出

- 过网球速 (km/h)
- 落地点坐标 (in/out 判断)
- 回合分割（过网次数统计）

### 代码位置

| 文件 | 功能 |
|------|------|
| `app/trajectory.py` | 三阶段轨迹分析 |
| `app/analytics.py` | BounceDetector + RallyTracker（实时分析） |
| `notebooks/trajectory_analysis.ipynb` | 交互式分析 notebook |

---

## 8. 实时系统

### 多进程架构

```
主进程
├── FastAPI (uvicorn :8000)
├── Orchestrator
│   ├── cam66 子进程: CameraStream → TrackNet → BallTracker → Homography → Queue
│   ├── cam68 子进程: CameraStream → TrackNet → BallTracker → Homography → Queue
│   └── 消费线程: 配对检测 → MultiBlobMatcher → triangulate → 3D 位置
└── BounceDetector + RallyTracker (实时分析)
```

### 进程间通信

| 机制 | 方向 | 用途 |
|------|------|------|
| mp.Queue(64) | 子进程 → 主进程 | 检测结果 |
| mp.Event | 主进程 → 子进程 | 停止信号 |
| Manager().dict() | 双向 | 共享状态 |

### API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | Web Dashboard |
| `/api/status` | GET | 系统状态 |
| `/api/ball3d` | GET | 最新 3D 坐标 |
| `/api/ball3d/stream` | GET (SSE) | 实时 3D 位置推送 |
| `/api/analytics/live` | GET | 实时分析（落点/回合） |
| `/api/pipeline/{name}/start` | POST | 启动管道 |
| `/api/pipeline/{name}/stop` | POST | 停止管道 |
| `/api/video-test/start` | POST | 离线视频处理 |
| `/api/video-test/compute-3d` | POST | 帧匹配三角测量 |
| `/api/video-test/compute-trajectory` | POST | 轨迹拟合 |

### 代码位置

| 文件 | 功能 |
|------|------|
| `main.py` | 入口（FastAPI + Orchestrator） |
| `app/orchestrator.py` | 进程管理 + 消费线程 |
| `app/pipeline/camera_pipeline.py` | 子进程入口 |
| `app/pipeline/camera_stream.py` | RTSP 帧读取（线程 + 自动重连） |
| `app/api/routes.py` | REST API 路由 |
| `app/api/templates/dashboard.html` | Web 管理界面 |

---

## 9. 训练工具链

### 数据准备

```bash
python -m tools.prepare_yolo_crops <标注目录1> <标注目录2> [--crop-size 128]
```

- 支持 LabelMe 格式的 rectangle 和 point 标注
- 正样本：以 GT 球心为中心裁剪 128×128
- 负样本：从标注帧的背景区域随机裁剪（距球 > 100px）
- 输出 YOLO 格式：`data/blob_crops/{images,labels}/{train,val}/`

### 模型训练

```bash
python -m tools.train_blob_verifier --device 0 --epochs 100
```

- 基础模型：yolo11n.pt → 微调
- 关键参数：freeze=0, mosaic=0.0, single_cls=True
- 输出：`model_weight/blob_verifier_yolo.pt`

### 训练经验总结

1. **bbox 高度=0 bug 是致命的**：LabelMe rectangle 解析必须用 pts[0] 和 pts[2]（对角点）
2. **freeze=0 优于 freeze=10**：COCO 特征不适合 128px 网球 crop，需全层微调
3. **mosaic=0.0 是必须的**：mosaic 拼接缩小子图至 64px，10px 球几乎不可见
4. **负样本只从标注帧取**：未标注帧可能含未标记的球

---

## 10. 评估工具链

### TrackNet + YOLO 评估

```bash
python -m tools.eval_tracknet_yolo
```

4 项测试：
1. **TrackNet 原始 Recall**：top-1/3/5 召回率 vs GT
2. **YOLO 影响**：YOLO 在多 blob 帧中的 rescue/kill 比例
3. **稳定性分析**：帧间跳跃（>100px）的来源分析
4. **死球污染**：静止窗口检测（5帧内位移 < 8px）

### 交叉验证评估

```bash
python -m tools.eval_cross_camera
```

- ray_distance 分布统计
- AGREE/DISAGREE 帧的 GT 误差对比
- 交叉验证纠正率

### 可视化渲染

```bash
python -m tools.render_cross_camera_video
```

- 双画面并排显示（cam66 | cam68）
- 检测标记：绿圈=正确, 红叉=错误, 黄圈=无 GT
- 底部状态栏：ray_distance 指示条 + AGREE/DISAGREE
- GT 菱形标记
- 轨迹拖尾线

---

## 11. 完整数据流图

```
                    RTSP 摄像头 (1920×1080, 25fps)
                    cam66 (近端)    cam68 (远端)
                        │               │
                        ▼               ▼
              ┌─────────────────────────────────────┐
              │      CameraStream (线程读帧)          │
              │      自动重连, 帧缓冲                  │
              └─────────┬───────────────┬───────────┘
                        │               │
                  8帧缓冲填满         8帧缓冲填满
                        │               │
                        ▼               ▼
              ┌─────────────────────────────────────┐
              │      TrackNet 推理 (PyTorch GPU)      │
              │      输入: (1, 24, 288, 512)          │
              │      输出: 8张热力图                    │
              └─────────┬───────────────┬───────────┘
                        │               │
                   逐帧处理          逐帧处理
                        │               │
                        ▼               ▼
              ┌─────────────────────────────────────┐
              │      BallTracker 后处理                │
              │      OSD遮罩 → 阈值 → connectedComp   │
              │      → blob列表 (最多5个, 按blob_sum排)│
              └─────────┬───────────────┬───────────┘
                        │               │
                   blob列表          blob列表
                        │               │
                        ▼               ▼
              ┌─────────────────────────────────────┐
              │      YOLO Blob Verifier (每帧)        │
              │      extract_crops(原始帧, 128×128)    │
              │      → YOLO detect → 过滤+重排序       │
              │      → fallback: TrackNet top-1        │
              └─────────┬───────────────┬───────────┘
                        │               │
                  验证后blob列表     验证后blob列表
                        │               │
                        ▼               ▼
              ┌─────────────────────────────────────┐
              │      HomographyTransformer            │
              │      pixel → world (米)               │
              │      Court-X 边界过滤                   │
              └─────────┬───────────────┬───────────┘
                        │               │
                   world坐标         world坐标
                        │               │
               ──── mp.Queue ──── mp.Queue ────
                        │               │
                        ▼               ▼
              ┌─────────────────────────────────────┐
              │      Orchestrator 消费线程              │
              │      配对条件: |ts1 - ts2| < 0.1s     │
              └────────────────┬────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────┐
              │      MultiBlobMatcher                 │
              │      枚举 blob 配对 → triangulate()    │
              │      选择 ray_distance 最小的配对       │
              │      过滤: ray_dist < 2m, z ∈ [0,6]  │
              └────────────────┬────────────────────┘
                               │
                          3D (x,y,z)
                               │
                               ▼
              ┌─────────────────────────────────────┐
              │      实时分析                          │
              │      BounceDetector: V形拟合检测落地    │
              │      RallyTracker: 过网计数/回合状态     │
              └────────────────┬────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────┐
              │      FastAPI 接口 / Dashboard          │
              │      REST + SSE 实时推送               │
              └─────────────────────────────────────┘
```

---

## 12. 关键参数汇总

| 组件 | 参数 | 当前值 | 说明 |
|------|------|--------|------|
| TrackNet | threshold | 0.1 | 热力图二值化阈值 |
| TrackNet | input_size | 288×512 | 模型输入分辨率 |
| TrackNet | frames_in/out | 8 / 8 | 时序窗口 |
| TrackNet | heatmap_mask | [0,0,620,40] | OSD 遮罩区域 |
| YOLO | conf | 0.15 | blob 验证置信度阈值 |
| YOLO | crop_size | 128 | 裁剪区域大小 |
| YOLO | model | blob_verifier_yolo.pt | 微调后的 yolo11n |
| Homography | court_x_margin | 1.0m | Court-X 过滤边距 |
| 三角测量 | max_ray_distance | 2.0m | MultiBlobMatcher 配对阈值 |
| 三角测量 | valid_z_range | [0, 6]m | 物理合理性约束 |
| 交叉验证 | agree_threshold | 1.5m | AGREE/DISAGREE 分界 |
| 相机 | cam66 position | (3.77, -15.73, 7.0) | 近端, 高度 7m |
| 相机 | cam68 position | (5.81, 45.04, 7.0) | 远端, 高度 7m |
| Ensemble | agree_distance | 3.0px | TrackNet/HRNet 一致性阈值 |
| 落地检测 | vertex_z | < 0.2m | 接近地面判断 |

---

## 13. 文件索引

```
tennis-3d-tracking/
├── main.py                              # 入口
├── config.yaml                          # 全局配置
├── requirements.txt                     # 依赖
│
├── app/
│   ├── config.py                        # Pydantic 配置模型
│   ├── schemas.py                       # 数据结构定义
│   ├── orchestrator.py                  # 多进程编排 + 三角测量消费
│   ├── triangulation.py                 # 射线交叉 3D 重建
│   ├── trajectory.py                    # 轨迹拟合 (RANSAC/球速/落点)
│   ├── multi_blob_matcher.py            # 跨摄像头 blob 最优配对
│   ├── analytics.py                     # 落地检测 + 回合追踪
│   ├── calibration.py                   # 相机标定 (PnP)
│   │
│   ├── pipeline/                        # 子进程管道
│   │   ├── camera_pipeline.py           # 子进程入口
│   │   ├── camera_stream.py             # RTSP 帧读取
│   │   ├── tracknet.py                  # TrackNet 模型
│   │   ├── inference.py                 # 推理工厂
│   │   ├── postprocess.py              # 热力图 → blob
│   │   ├── homography.py               # 像素 ↔ 世界坐标
│   │   ├── blob_verifier.py            # YOLO 验证
│   │   ├── ensemble.py                 # TrackNet+HRNet 集成
│   │   └── video_pipeline.py           # 离线视频处理管道
│   │
│   └── api/
│       ├── routes.py                    # REST API
│       └── templates/dashboard.html     # Web 界面
│
├── tools/                               # 工具脚本
│   ├── prepare_yolo_crops.py            # YOLO 训练数据生成
│   ├── train_blob_verifier.py           # YOLO 模型训练
│   ├── evaluate_blob_verifier.py        # Blob Verifier 评估
│   ├── eval_tracknet_yolo.py            # TrackNet+YOLO 4项评估
│   ├── eval_cross_camera.py             # 交叉验证评估
│   ├── render_cross_camera_video.py     # 双画面可视化渲染
│   └── test_yolo_baseline.py            # YOLO 零样本基线测试
│
├── model_weight/
│   ├── TrackNet_best.pt                 # TrackNet 模型
│   ├── blob_verifier_yolo.pt            # YOLO 验证模型
│   └── hrnet_tennis.onnx               # HRNet ONNX (可选)
│
├── src/                                 # 标定数据
│   ├── homography_matrices.json         # 预计算 homography
│   ├── camera_calibration.json          # 相机内外参
│   └── *.json                           # LabelMe 关键点标注
│
├── docs/                                # 文档
│   ├── technical_overview.md            # 本文件：技术全景
│   ├── architecture.md                  # 系统架构（旧版）
│   ├── blob_verifier_training.md        # YOLO 训练指南
│   ├── blob_verifier_evaluation.md      # 评估报告
│   ├── algorithms.md                    # 算法详解
│   ├── trajectory_analysis_guide.md     # 轨迹分析指南
│   └── trajectory_features.md           # 轨迹功能说明
│
└── notebooks/
    └── trajectory_analysis.ipynb        # 交互式分析
```
