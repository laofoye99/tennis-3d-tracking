# 标注工作流：模型预标注 → 人工校验

## 概述

使用 TrackNet 模型对视频进行推理，将所有检测到的 blob 导出为 LabelMe 格式的 JSON 标注文件，
然后人工在 LabelMe 软件中校验、修正、补充标注。

## 工作流步骤

### 第1步：模型预标注

使用 `tools/export_labelme.py` 对目标视频跑 TrackNet 推理，导出每帧所有 blob。

```bash
# 从项目根目录运行
python -m tools.export_labelme <视频路径> <输出目录> <threshold>
```

**参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| 视频路径 | 输入视频文件 | `uploads/cam68_20260307_173403_2min.mp4` |
| 输出目录 | LabelMe JSON 输出路径 | `exports/cam68_2min_blobs_t035` |
| threshold | heatmap 阈值（越高越严格） | `0.35` |

**示例：**

```bash
# cam68 视频，threshold=0.35
python -m tools.export_labelme uploads/cam68_20260307_173403_2min.mp4 exports/cam68_2min_blobs_t035 0.35

# cam66 视频，threshold=0.35
python -m tools.export_labelme uploads/cam66_20260307_173403_2min.mp4 exports/cam66_2min_blobs_t035 0.35
```

**输出：** 每帧一个 JSON 文件（`00000.json` ~ `02999.json`），LabelMe 格式。

### 第2步：准备 LabelMe 工作目录

将视频抽帧为图片，和 JSON 放在同一目录，LabelMe 才能打开查看。

```bash
# 抽帧（如果还没有图片）
python -c "
import cv2
cap = cv2.VideoCapture('uploads/cam68_20260307_173403_2min.mp4')
i = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imwrite(f'exports/cam68_2min_blobs_t035/{i:05d}.jpg', frame)
    i += 1
cap.release()
print(f'Exported {i} frames')
"
```

目录结构：
```
exports/cam68_2min_blobs_t035/
├── 00000.jpg     ← 视频帧图片
├── 00000.json    ← 模型预标注
├── 00001.jpg
├── 00001.json
├── ...
```

### 第3步：用 LabelMe 打开校验

```bash
labelme exports/cam68_2min_blobs_t035/
```

在 LabelMe 中：
- 每帧会显示模型预标注的所有 blob 点
- 每个 blob 的 `description` 字段包含 `blob_sum`、`blob_max`、`area` 信息

### 第4步：人工标注规则

#### 标注对象

| 情况 | 是否标注 | label 值 |
|------|---------|----------|
| 运动中的比赛球 | ✅ 标注 | `ball` |
| 地上静止的死球 | ❌ 不标 | — |
| 被踢动的非比赛球 | ✅ 标注 | `ball` |
| 人/球拍等误检 | ❌ 删除该标注点 | — |

**原则：TrackNet 是时序模型，通过多帧差异检测运动物体。静止球与背景融合不会被检测到，也不需要标注。**

#### 操作方式

- **正确的 blob** → 保留不动，确认 label 为 `ball`
- **误检的 blob** → 删除该标注点
- **遗漏的球** → 手动添加 point 标注，label 设为 `ball`
- **位置有偏差** → 拖动标注点到正确位置

### 第5步：运行评估（可选）

标注完成后，用评估模块对比模型输出和人工标注：

```bash
# 评估单个摄像头
python -m app.pipeline.evaluate --camera cam68

# 评估所有摄像头，输出 JSON 报告
python -m app.pipeline.evaluate --output eval_report.json
```

---

## threshold 选择参考

| threshold | 效果 | 适用场景 |
|-----------|------|---------|
| 0.1 | blob 非常多，噪声多 | 不推荐用于标注 |
| 0.3 | blob 较多，仍有少量噪声 | 需要高召回率时 |
| **0.35** | **平衡：blob 数量适中，噪声少** | **推荐用于预标注** |
| 0.5 | blob 较少，可能漏掉弱信号帧 | 仅需高置信度检测时 |

**建议使用 0.35**，在减少噪声的同时保留大部分真实检测。

---

## 输出 JSON 格式说明

每个 JSON 文件符合 LabelMe 标准格式：

```json
{
  "version": "2.5.4",
  "flags": {},
  "shapes": [
    {
      "label": "ball",
      "score": 18.53,
      "points": [[992.09, 247.65]],
      "group_id": 0,
      "description": "blob_sum=18.53 blob_max=0.797 area=39",
      "shape_type": "point",
      "flags": {},
      "attributes": {}
    }
  ],
  "imagePath": "00051.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

**字段说明：**

| 字段 | 含义 |
|------|------|
| `score` | blob_sum，heatmap 响应总和（越大越可能是真球） |
| `points` | 像素坐标 `[x, y]` |
| `group_id` | 同帧多个 blob 的编号（0=最强 blob） |
| `description` | blob_sum / blob_max / area 详情 |
| `shapes` 为空 | 该帧无检测 |

---

## 快速参考

```bash
# 1. 模型预标注
python -m tools.export_labelme <视频> <输出目录> <threshold>

# 2. 抽帧（如需）
python -c "import cv2; cap=cv2.VideoCapture('<视频>'); i=0
while cap.read()[0]: cv2.imwrite(f'<输出目录>/{i:05d}.jpg', cap.read()[1]); i+=1"

# 3. 打开 LabelMe 校验
labelme <输出目录>/

# 4. 评估
python -m app.pipeline.evaluate --camera <cam> --output eval.json
```
