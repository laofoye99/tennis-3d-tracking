# 标定坐标系修正记录

## 发现的问题

球场 12 个关键点标注（cam66.json, cam68.json）标记的是**单打线**角点位置，
但 `compute_homography.py` 将这些点映射到了**双打线**的世界坐标。

### 具体错误

```
compute_homography.py 中:

WORLD_COORDS_CAM66 = {
    "left_top":           (0.0,      23.77),    ← 错: 双打左边线
    "left_bottom":        (0.0,      0.0),      ← 错
    "right_top":          (8.23,     23.77),    ← 错: 双打右边线
    "right_bottom":       (8.23,     0.0),      ← 错
    "center_top":         (4.115,    23.77),    ← 对: 中心线
    "center_bottom":      (4.115,    0.0),      ← 对
    ...
}

实际标注的是:
    "left_*":  单打左边线 x = 1.37m
    "right_*": 单打右边线 x = 6.86m
    "center_*": 中心线 x = 4.115m (恰好一致)
```

### 影响

| 受影响的文件 | 说明 |
|-------------|------|
| `src/compute_homography.py` | 世界坐标定义错误 |
| `src/homography_matrices.json` | H 矩阵基于错误坐标计算 |
| `src/camera_calibration.json` | K, R, t 基于错误坐标 |
| `tools/render_tracking_video.py` | 3D 重建、bounce IN/OUT 判断 |
| `app/pipeline/homography.py` | pixel_to_world 转换 |
| `app/triangulation.py` | 射线交叉三角化 |

### 影响程度分析

- **x 坐标**: 系统性偏移。标定认为球场宽 8.23m（双打），实际标注区域宽 5.49m（单打）。
  所有 x 坐标被拉伸了 8.23/5.49 = 1.5 倍。
- **y 坐标**: 不受影响。球场长度 23.77m 和 y 方向标注一致。
- **z 坐标**: 间接受影响。x 偏差导致三角化射线方向有误，Z 精度降低。
- **center 系列**: 恰好正确。(1.37+6.86)/2 = 4.115 = 8.23/2。

## 修正方案

```python
# 单打场尺寸
SINGLES_LEFT  = 1.37   # 单打左边线
SINGLES_RIGHT = 6.86   # 单打右边线
SINGLES_WIDTH = 5.49   # 单打场宽

# Camera 66
WORLD_COORDS_CAM66 = {
    "left_top":           (SINGLES_LEFT,   COURT_LENGTH),     # 1.37, 23.77
    "left_top_serve":     (SINGLES_LEFT,   SERVICE_FAR_Y),    # 1.37, 18.285
    "left_bottom_serve":  (SINGLES_LEFT,   SERVICE_NEAR_Y),   # 1.37, 5.485
    "left_bottom":        (SINGLES_LEFT,   0.0),              # 1.37, 0.0
    "center_top":         (CENTER_X,       COURT_LENGTH),     # 4.115, 23.77
    "center_top_serve":   (CENTER_X,       SERVICE_FAR_Y),
    "center_bottom_serve":(CENTER_X,       SERVICE_NEAR_Y),
    "center_bottom":      (CENTER_X,       0.0),
    "right_top":          (SINGLES_RIGHT,  COURT_LENGTH),     # 6.86, 23.77
    "right_top_serve":    (SINGLES_RIGHT,  SERVICE_FAR_Y),
    "right_bottom_serve": (SINGLES_RIGHT,  SERVICE_NEAR_Y),
    "right_bottom":       (SINGLES_RIGHT,  0.0),              # 6.86, 0.0
}
```

## 修正后需要重新生成

1. `src/homography_matrices.json` — 重新计算 H
2. `src/camera_calibration.json` — 重新计算 K, R, t
3. 所有 tracking 视频输出需要重新渲染
