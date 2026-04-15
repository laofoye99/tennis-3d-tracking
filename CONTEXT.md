# Tennis 3D Tracking — Project Context

## 项目概述
双海康相机实时网球追踪系统。检测球位置 → 三角化 → 3D 轨迹 → bounce/速度分析 → Dashboard 展示 + 3D WebSocket 推送。

## 当前状态 (commit 0e4be45)
- **检测**: TrackNet V3 (8帧输入, ONNX Runtime推理)
- **背景**: 预算静态中值帧 `src/bg_median_cam66.png` / `cam68.png`
- **三角化**: 逐帧配对 (capture_ts ±2s window) → ray intersection
- **Bounce**: `detect_bounces()` — find_peaks on -z, prominence=0.10, z_max=0.5, z>0.02
- **速度**: 5帧中位数, 30-150 km/h 范围, 过网时记录
- **Dashboard**: 实时 MJPEG + minimap + 3D iframe + GT overlay + Report 生成

## 坐标系 (V2)
- 原点: 场地中心, 网在 y=0
- x: [-4.115, +4.115] (单打=全场宽度)
- y: [-11.89, +11.89] (半场长度)
- cam66: [0.165, -17.042, 6.217] (近端 y<0)
- cam68: [0.211, 17.156, 5.286] (远端 y>0)

## 关键精度指标 (TrackNet 自动检测)
### 像素跟踪 (px,py vs GT)
- cam66: 87.1% ≤25px, 中位误差 2.8px, **9.1% 完全错误 (>100px)**
- cam68: 87.6% ≤25px, 中位误差 3.3px, **8.7% 完全错误 (>100px)**

### Bounce 检测 (TrackNet → 三角化 → find_peaks vs GT 30个bounce)
- Recall: 76.7% (23/30)
- Precision: 46.0% — **27个假阳性**
- 落地误差中位: 15.2cm (matched bounces)

### 根因分析
- 87% 帧跟踪准确, 但 9% 完全跟错 → 这些错误帧三角化后产生假3D点 → find_peaks误判为bounce
- 假阳性集中在非比赛帧 (颠球/准备动作/球员移动)
- GT数据上跑 detect_bounces = 30/30, 0 FP → bounce算法本身没问题, 问题在检测质量

## 评估框架
`eval/` 目录已搭建, 4种方法组合评估结果在 `eval/results/comparison.json`:
| 方法 | Recall | Precision | F1 | FP | 落地误差median |
|------|--------|-----------|-----|-----|---------------|
| TrackNet + per-frame | 76.7% | 54.8% | 63.9% | 19 | 15.2cm |
| TrackNet + track-first | 50.0% | 65.2% | 56.6% | 8 | 9.5cm |

## 文件结构
```
app/
  orchestrator.py      — 主控: 消费检测 → 配对 → 三角化 → bounce → 速度 → WS推送
  report.py            — 报告生成 (从JSONL数据)
  analytics.py         — BounceDetector, PeakBounceDetector, RallyStateMachine
  config.py            — Pydantic配置模型
  pipeline/
    inference.py       — TrackNetDetector (ONNX) + MedianBGDetector + static median加载
    camera_pipeline.py — 相机子进程: RTSP→检测→homography→结果队列
    camera_stream.py   — RTSP线程读取
    postprocess.py     — BallTracker: 热力图→blob坐标
    bounce_detect.py   — detect_bounces (find_peaks) + detect_events
    tracker.py         — track_single_camera + match_and_triangulate
    blob_detector.py   — BallBlobDetector (30帧中值背景减除)
    tracknet.py        — TrackNet U-Net模型定义
    homography.py      — HomographyTransformer
  api/
    routes.py          — FastAPI路由
    templates/
      dashboard.html   — 主Dashboard (minimap + 3D iframe + GT overlay + Reports)
      report_template.html — Chart.js报告模板

config.yaml            — 运行配置 (detector_type: auto/median_bg)
src/
  bg_median_cam66.png  — 预算静态中值背景
  bg_median_cam68.png
  homography_matrices.json
```

## 待优化方向
1. **降低9%的像素跟踪错误率** — 这是bounce假阳性的根源
2. **区分比赛帧和非比赛帧** — 颠球/准备动作不应触发bounce
3. **Transformer轨迹关联模块** — 训练数据已准备在 `D:\tennis\trajectory_transformer\`
4. **速度精度** — speed_buffer入口已加z/距离/场内三层过滤 (未commit)
5. **录制功能稳定性** — JPEG编码竞态/丢帧问题已定位 (未commit)

## GT数据位置
- `D:/tennis/blob_frame_different/bounce_results.json` — 30个bounce的3D坐标GT
- `D:/tennis/blob_frame_different/GT/` — 像素标注 (25K帧, 17.8K球标注)
- `uploads/cam66_20260307_173403_2min/` — GT视频帧+标注 (3000帧)
- `uploads/cam68_20260307_173403_2min/` — 同上

## WebSocket 3D推送格式
```json
{"msg":{"message":"bounce_data","data":{"bounce":{"timeStamp":ms,"x":V2_x*10,"y":V2_y*10,"speed":kmh}}}}
```
目标: wss://tennisserver.motionrivalry.com:8086/general
speed=0 时不推送

## 配置切换
TrackNet: `detector_type: auto, frames_in: 8`
MedianBG: `detector_type: median_bg, frames_in: 30`
