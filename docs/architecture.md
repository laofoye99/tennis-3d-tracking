# Tennis 3D Ball Tracking System - 系统架构文档

## 1. 系统总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Main Process (主进程)                         │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│  │   FastAPI     │   │   Orchestrator   │   │  Triangulation    │   │
│  │  Web + API    │◄──│   进程管理器      │──►│  3D 三角测量       │   │
│  │  :8000        │   │                  │   │  (线程)            │   │
│  └──────────────┘   └────────┬─────────┘   └───────────────────┘   │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              │ mp.Queue      │ mp.Queue      │                      │
│              ▼               ▼               │                      │
│  ┌───────────────┐  ┌───────────────┐       │                      │
│  │ Camera 66     │  │ Camera 68     │       │                      │
│  │ Pipeline      │  │ Pipeline      │       │                      │
│  │ [子进程]       │  │ [子进程]       │       │                      │
│  └───────────────┘  └───────────────┘       │                      │
└─────────────────────────────────────────────────────────────────────┘
```

两个摄像头从球场两端俯拍同一块单打网球场，分别通过 ONNX 模型检测球的像素坐标，
经单应性矩阵转换到世界坐标后，由三角测量模块计算出球的 3D 空间位置。

---

## 2. 文件结构

```
D:/tennis/tennis_1.0/
│
├── main.py                          # 入口: 加载配置 → Orchestrator → FastAPI
├── config.yaml                      # 全局配置 (RTSP/相机位置/模型/服务器)
├── requirements.txt                 # Python 依赖
│
├── app/                             # ===== 应用主体 =====
│   ├── config.py                    # 配置层: Pydantic 模型 + YAML 加载
│   ├── schemas.py                   # 数据层: 所有数据模型定义
│   ├── orchestrator.py              # 控制层: 子进程生命周期 + 三角测量消费
│   ├── triangulation.py             # 算法层: 射线交叉法 3D 重建
│   ├── trajectory.py                # 算法层: 轨迹拟合 (RANSAC/球速/落点/回合)
│   │
│   ├── pipeline/                    # ===== 管道层 (子进程内) =====
│   │   ├── camera_stream.py         # 输入: RTSP 帧读取 (线程 + 自动重连)
│   │   ├── inference.py             # 推理: ONNX HRNet 热力图 (CUDA/CPU)
│   │   ├── postprocess.py           # 后处理: 热力图 → 像素坐标 + 追踪
│   │   ├── homography.py            # 变换: 像素 ↔ 世界坐标 (预计算矩阵)
│   │   └── camera_pipeline.py       # 组装: 子进程入口函数
│   │
│   └── api/                         # ===== 接口层 =====
│       ├── routes.py                # REST API + SSE 端点
│       └── templates/
│           └── dashboard.html       # Web 管理界面
│
├── model_weight/
│   └── hrnet_tennis.onnx            # ONNX 模型 (输入 9ch, 输出 3ch)
│
├── src/                             # ===== 标定工具 =====
│   ├── compute_homography.py        # 单应性矩阵计算脚本
│   ├── homography_matrices.json     # 预计算矩阵 (cam66 + cam68)
│   ├── *.json                       # Labelme 标注文件 (12个球场关键点)
│   └── *.jpeg                       # 标定参考图像
│
└── docs/                            # ===== 文档 =====
    ├── architecture.md              # 本文件
    ├── algorithms.md                # 算法与计算逻辑
    └── trajectory_features.md       # 轨迹分析功能 (球速/落点/回合分割)
```

---

## 3. 分层架构

### 3.1 配置层 — `config.yaml` + `app/config.py`

```yaml
cameras:
  cam66:
    rtsp_url: "rtsp://admin:admin@192.168.1.66:554/..."
    position_3d: [4.115, -1.0, 5.0]     # 世界坐标 [x, y, z] 米
    homography_key: "cam66"
  cam68:
    rtsp_url: "rtsp://admin:admin@192.168.1.68:554/..."
    position_3d: [4.115, 24.77, 5.0]
    homography_key: "cam68"
model:
  path: "model_weight/hrnet_tennis.onnx"
  input_size: [288, 512]                 # 模型输入 H×W
  frames_in: 3                           # 输入帧数
  frames_out: 3                          # 输出帧数
  threshold: 0.5                         # 热力图阈值
  device: "cuda"
```

Pydantic 模型链:
```
AppConfig
├── cameras: dict[str, CameraConfig]
├── model: ModelConfig
├── homography: HomographyConfig
└── server: ServerConfig
```

### 3.2 数据层 — `app/schemas.py`

```
BallDetection2D          像素级检测结果
  camera_name, frame_id, pixel_x, pixel_y, confidence, timestamp

WorldPoint2D             世界坐标 (米)
  camera_name, x, y, pixel_x, pixel_y, confidence, timestamp

BallPosition3D           三角测量 3D 结果
  x, y, z, timestamp, cam66_world?, cam68_world?

PipelineStatus           单管道状态
  name, state(running/stopped/error), fps, last_detection_time, error_msg?

SystemStatus             系统总状态
  pipelines: dict, triangulation_active, latest_ball_3d?
```

### 3.3 管道层 — `app/pipeline/` (子进程内运行)

每个摄像头独立运行一个子进程，内部由 4 个模块串联:

```
┌─────────────────── 子进程 (mp.Process) ───────────────────┐
│                                                            │
│  CameraStream (线程)                                       │
│  ├── RTSP 连接 (cv2.VideoCapture + CAP_FFMPEG)            │
│  ├── 后台线程持续读帧                                       │
│  ├── 自动重连 (间隔 3 秒)                                   │
│  └── read() → (frame, frame_id, timestamp)                │
│          │                                                 │
│          ▼ 缓冲 3 帧                                       │
│                                                            │
│  BallDetector (ONNX)                                       │
│  ├── 预处理: resize(288×512) + RGB + normalize + CHW       │
│  ├── 3帧 stack → [1, 9, 288, 512]                         │
│  ├── ONNX IOBinding (CUDA GPU 加速)                        │
│  ├── 自动回退 CPU                                          │
│  └── sigmoid → heatmaps [3, 288, 512]                     │
│          │                                                 │
│          ▼ 逐帧处理                                        │
│                                                            │
│  BallTracker                                               │
│  ├── resize 热力图 → 1920×1080                             │
│  ├── 阈值过滤 (>0.5)                                       │
│  ├── connectedComponents → blob 列表                       │
│  ├── 加权质心计算                                           │
│  ├── 线性预测选择最近 blob                                  │
│  └── → (pixel_x, pixel_y, confidence)                     │
│          │                                                 │
│          ▼                                                 │
│                                                            │
│  HomographyTransformer                                     │
│  ├── H_img2world @ [px, py, 1] → [wx, wy, w]             │
│  └── → (world_x, world_y) 米                              │
│          │                                                 │
│          ▼                                                 │
│  result_queue.put(detection_dict)  ──→  主进程              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.4 控制层 — `app/orchestrator.py`

```
Orchestrator
│
├── start_pipeline(name)
│   ├── 创建 mp.Queue(maxsize=64)     # 结果队列
│   ├── 创建 mp.Event()               # 停止信号
│   ├── 创建 Manager().dict()         # 共享状态
│   └── 启动 mp.Process(run_pipeline, daemon=True)
│
├── stop_pipeline(name)
│   ├── stop_event.set()
│   ├── process.join(timeout=10)
│   └── process.terminate() (超时强杀)
│
├── _consume_loop() [后台线程]
│   ├── 从所有 result_queue 非阻塞读取
│   ├── 存入 _latest_detections[camera_name]
│   ├── 配对检查: 两个相机的时间差 < 0.1 秒
│   └── 满足条件 → triangulate() → BallPosition3D
│
├── get_system_status()  → SystemStatus
├── get_latest_3d()      → BallPosition3D?
└── shutdown()           → 停止所有子进程
```

### 3.5 算法层 — `app/triangulation.py` + `app/trajectory.py`

基础三角测量详见 `docs/algorithms.md`

轨迹分析功能详见 `docs/trajectory_features.md`，包括:
- 检测清洗 (置信度/边界/速度/孤立点)
- 自动时间偏移搜索 (修剪均值)
- RANSAC 空间抛物线拟合
- 过网球速计算
- 落地点检测
- 回合分割

### 3.6 接口层 — `app/api/`

```
FastAPI Routes
│
├── GET  /                           Web Dashboard (HTML)
├── GET  /api/status                 系统总状态 (JSON)
├── GET  /api/ball3d                 最新 3D 坐标 (JSON)
├── GET  /api/ball3d/stream          SSE 实时推送
├── POST /api/pipeline/{name}/start  启动管道
├── POST /api/pipeline/{name}/stop   停止管道
├── GET  /api/pipeline/{name}/detection  单相机最新检测
│
├── POST /api/video-test/start       启动视频文件处理
├── POST /api/video-test/stop        停止视频处理
├── GET  /api/video-test/status      视频处理进度
├── POST /api/video-test/compute-3d  帧匹配三角测量
└── POST /api/video-test/compute-trajectory  轨迹拟合 (球速/落点/回合)
```

---

## 4. 数据流总览

```
                      192.168.1.66               192.168.1.68
                      (RTSP摄像头)                (RTSP摄像头)
                           │                          │
                           ▼                          ▼
                    ┌─────────────┐           ┌─────────────┐
                    │ CameraStream│           │ CameraStream│
                    │  (线程读帧)  │           │  (线程读帧)  │
                    └──────┬──────┘           └──────┬──────┘
                           │ frame                    │ frame
                           ▼                          ▼
                    ┌─────────────┐           ┌─────────────┐
                    │ BallDetector│           │ BallDetector│
                    │  (ONNX GPU) │           │  (ONNX GPU) │
                    └──────┬──────┘           └──────┬──────┘
                           │ heatmap                  │ heatmap
                           ▼                          ▼
                    ┌─────────────┐           ┌─────────────┐
                    │ BallTracker │           │ BallTracker │
                    │  (blob检测)  │           │  (blob检测)  │
                    └──────┬──────┘           └──────┬──────┘
                           │ pixel (x, y)             │ pixel (x, y)
                           ▼                          ▼
                    ┌─────────────┐           ┌─────────────┐
                    │ Homography  │           │ Homography  │
                    │ pixel→world │           │ pixel→world │
                    └──────┬──────┘           └──────┬──────┘
                           │ world (x, y)             │ world (x, y)
                           │                          │
                    ───── mp.Queue ─────────── mp.Queue ─────
                           │                          │
                           ▼                          ▼
                    ┌─────────────────────────────────────┐
                    │         Orchestrator 消费线程         │
                    │  配对条件: |ts1 - ts2| < 0.1 秒      │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │         Triangulation                │
                    │  射线交叉法 → 3D 坐标 (x, y, z)      │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │           FastAPI 接口                │
                    │  /api/ball3d   /api/ball3d/stream    │
                    │  /api/status   Web Dashboard         │
                    └─────────────────────────────────────┘
```

---

## 5. 进程间通信 (IPC)

| 机制 | 方向 | 用途 |
|------|------|------|
| `mp.Queue(64)` | 子进程 → 主进程 | 传递 WorldPoint2D 检测结果 |
| `mp.Event` | 主进程 → 子进程 | 停止信号 |
| `Manager().dict()` | 双向 | 共享状态 (state/fps/error_msg) |

### 崩溃隔离

- 子进程设置 `daemon=True`，主进程退出时自动回收
- 子进程异常被 `try/except` 捕获，状态设为 `error`，不影响主进程
- 主进程 `stop_pipeline()` 有 10 秒超时 + 强制 `terminate()` 兜底
- `result_queue.put_nowait()` 防止队列满时阻塞子进程

---

## 6. 世界坐标系

```
        x=0                x=4.115              x=8.23
        (左边线)            (中线)               (右边线)
         │                   │                    │
y=23.77 ─┼───────────────────┼────────────────────┤─ 远端底线
         │                   │                    │
y=18.285─┼───────────────────┼────────────────────┤─ 远发球线
         │                   │                    │
         │              发球区 (上)                 │
         │                   │                    │
y=11.885─┼═══════════════════╪════════════════════╡─ 球网
         │                   │                    │
         │              发球区 (下)                 │
         │                   │                    │
y=5.485 ─┼───────────────────┼────────────────────┤─ 近发球线
         │                   │                    │
y=0     ─┼───────────────────┼────────────────────┤─ 近端底线
         │                   │                    │

  Camera 66 位于 y < 0 (近端后方, 俯视远端)
  Camera 68 位于 y > 23.77 (远端后方, 俯视近端)
  z 轴: 垂直向上, z=0 为地面
```

- 单位: 米 (m)
- 原点: Camera 66 一侧底线左角
- 标准 ITF 单打场地: 23.77m × 8.23m

---

## 7. API 接口参考

### GET /api/status
```json
{
  "pipelines": {
    "cam66": {
      "name": "cam66",
      "state": "running",
      "fps": 25.3,
      "last_detection_time": 1709567890.123,
      "error_msg": null
    },
    "cam68": { ... }
  },
  "triangulation_active": true,
  "latest_ball_3d": {
    "x": 5.72, "y": 7.02, "z": 1.35,
    "timestamp": 1709567890.100,
    "cam66_world": { "camera_name": "cam66", "x": 5.72, "y": 7.02, ... },
    "cam68_world": { "camera_name": "cam68", "x": 5.50, "y": 7.10, ... }
  }
}
```

### GET /api/ball3d
```json
{ "detected": true, "x": 5.72, "y": 7.02, "z": 1.35, "timestamp": ... }
```

### GET /api/ball3d/stream (SSE)
```
data: {"x": 5.72, "y": 7.02, "z": 1.35, "timestamp": ...}

data: {"x": 5.80, "y": 7.15, "z": 1.20, "timestamp": ...}
```

### POST /api/pipeline/{name}/start
```json
{ "status": "ok", "message": "cam66 starting" }
```

---

## 8. 依赖关系

```
main.py
  ├── app/config.py         (load_config)
  ├── app/orchestrator.py   (Orchestrator)
  │     ├── app/schemas.py
  │     ├── app/triangulation.py
  │     ├── app/trajectory.py     (轨迹拟合/球速/落点/回合)
  │     └── app/pipeline/camera_pipeline.py  [子进程]
  │           ├── app/pipeline/camera_stream.py
  │           ├── app/pipeline/inference.py
  │           ├── app/pipeline/postprocess.py
  │           └── app/pipeline/homography.py
  └── app/api/routes.py     (FastAPI router)
        └── templates/dashboard.html
```

### 外部依赖
| 包 | 版本 | 用途 |
|---|---|---|
| opencv-python | >=4.8 | 视频流读取、图像处理 |
| numpy | >=1.24 | 数组计算 |
| onnxruntime-gpu | >=1.16 | ONNX 模型推理 |
| torch | >=2.0 | CUDA 张量 + sigmoid |
| scipy | >=1.11 | 三角测量优化 |
| fastapi | >=0.104 | Web API 框架 |
| uvicorn | >=0.24 | ASGI 服务器 |
| pydantic | >=2.5 | 数据校验 |
| pyyaml | >=6.0 | 配置文件解析 |
