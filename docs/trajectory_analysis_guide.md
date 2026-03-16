# 3D 轨迹分析指南

## 概述

`notebooks/trajectory_analysis.ipynb` 用于分析双摄像头网球追踪系统的 3D 轨迹质量，
测试轨迹拟合与噪声过滤方法，以及 GT 真值与模型输出的对比评估。

## 数据流

```
cam66 像素检测 ──→ 单应性投影 ──→ 世界坐标 (x,y) ──┐
                                                    ├→ 三角测量 → 3D (x,y,z)
cam68 像素检测 ──→ 单应性投影 ──→ 世界坐标 (x,y) ──┘
                                                          │
                                                          ▼
                                               速度计算 → 轨迹分段 → 段内滤波
                                                          │
                                                          ▼
                                               GT vs 模型 → 像素误差 + 3D 误差
```

## 数据目录结构

```
uploads/
  cam66_20260307_173403_2min/     # GT + 模型混合（手动标注覆盖了部分帧）
    00000.json ~ 02999.json       # LabelMe 格式
  cam68_20260307_173403_2min/     # 同上

exports/
  cam66_model/                    # 纯模型输出（threshold=0.3）
    00000.json ~ 02999.json       # LabelMe 格式，所有帧都有
  cam68_model/                    # 纯模型输出（threshold=0.35）
```

### 数据格式

检测数据为 LabelMe JSON 格式，区分方式：
- **GT（手标）**：`shapes[0].score = null`
- **模型输出**：`shapes[0].score = float`（blob_sum 值）

### 生成纯模型输出

```bash
python -m tools.export_labelme uploads/cam66_20260307_173403_2min.mp4 exports/cam66_model 0.3
python -m tools.export_labelme uploads/cam68_20260307_173403_2min.mp4 exports/cam68_model 0.35
```

## Notebook 结构

### 第1-2节：环境配置与数据加载

加载两个摄像头的检测数据到 DataFrame，包含字段：
- `pixel_x`, `pixel_y`：像素坐标
- `is_gt`：是否为手标真值
- `score`：模型输出的 blob_sum
- `has_detection`：该帧是否有检测

### 第3节：单应性投影

将像素坐标通过 3×3 单应性矩阵投影到地面世界坐标：

```
[wx·w, wy·w, w]ᵀ = H_image_to_world × [pixel_x, pixel_y, 1]ᵀ
world_x = wx·w / w
world_y = wy·w / w
```

注意：单应性假设 z=0（球在地面），球在空中时投影的是"影子"位置。

### 第4节：三角测量

两个摄像头各提供一条射线（从相机位置穿过地面投影点），求两射线最近点中点：

```
射线1: P₁(s) = cam66_pos + s × (ground66 - cam66_pos)
射线2: P₂(t) = cam68_pos + t × (ground68 - cam68_pos)

解析求解 s, t 使 ‖P₁(s) - P₂(t)‖ 最小
球位置 = (P₁(s) + P₂(t)) / 2
```

### 第5节：原始轨迹可视化

- **图1**：俯视/侧视/正视 2D 投影（matplotlib）
- **图2**：3D 交互式轨迹（plotly，可旋转/缩放）
- **图3**：x/y/z 时间序列

### 第6节：噪声分析

- **图4**：帧间 3D 速度时间序列
- 网球物理极限约 70 m/s（发球），超过此速度的帧很可能是误检

### 第7节：轨迹分段

- **图5**：分段后 Z 高度可视化

按以下条件将连续轨迹切分为独立飞行段：
- 帧间速度突变（超过阈值 → 击球/弹地/误检）
- 帧间隔过大（连续丢帧 → 轨迹中断）

每段内部近似恒定加速度（重力 + 空气阻力），可独立滤波。

### 第8节：段内滤波方法对比

三种方法（均为后处理，不反馈回检测，详见 `docs/trajectory_filtering.md`）：

| 方法 | 参数 | 特点 |
|------|------|------|
| 中值滤波 | kernel=5 | 抗单帧跳点，不改变整体趋势 |
| Savitzky-Golay | window=7, poly=2 | 保轨迹形态，匹配抛物线 |
| 多项式拟合 | z:2次, xy:1次 | 全局拟合，z轴用抛物线 |

- **图6**：最长段 x/y/z 三种滤波方法对比
- **图7**：3D 滤波对比（plotly 交互式）

### 第9节：异常点识别

- **图8**：异常点标注

通过 SG 滤波残差识别噪声点：原始值与滤波值偏差超过阈值（默认 0.5m）的帧标记为异常。

### 第10节：统计摘要

输出全局和各段的统计信息：坐标范围、速度统计、异常帧数。

### 第11节：GT vs 模型输出对比

加载 `exports/` 中的纯模型输出，与 `uploads/` 中的 GT 标注逐帧对比。

- **图9**：GT vs 模型 像素误差柱状图（逐摄像头，颜色编码误差等级）
- **图10**：3D 误差统计（GT 三角测量 vs 模型三角测量）
- **图11**：3D GT vs 模型轨迹叠加（plotly 交互式，红×=GT，蓝点=模型，红线=误差）
- **图12**：2D 像素级 GT vs 模型散点对比

误差分级：
- `<10px`：正确（绿色）
- `10-50px`：偏差（橙色）
- `>50px`：错误（红色）
- 无检测：漏检（灰色竖线）

## 可视化总览

| 图号 | 内容 | 类型 |
|------|------|------|
| 图1 | 俯视/侧视/正视 2D 投影 | matplotlib |
| 图2 | 3D 交互式轨迹 | plotly（可旋转） |
| 图3 | x/y/z 时间序列 | matplotlib |
| 图4 | 帧间速度 | matplotlib |
| 图5 | 轨迹分段 (Z 高度) | matplotlib |
| 图6 | 滤波方法对比 (x/y/z) | matplotlib |
| 图7 | 3D 滤波对比 | plotly（可旋转） |
| 图8 | 异常点标注 | matplotlib |
| 图9 | GT vs 模型 像素误差 | matplotlib |
| 图10 | GT vs 模型 3D 误差 | 数值统计 |
| 图11 | 3D GT vs 模型轨迹叠加 | plotly（可旋转） |
| 图12 | 2D 像素级 GT vs 模型 | matplotlib |

## 关键参数调整

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `FRAME_START/END` | 第1节 | 0/341 | 分析帧范围 |
| `speed_threshold` | 第7节 | 50.0 m/s | 分段速度阈值 |
| `gap_threshold` | 第7节 | 5 帧 | 分段间隔阈值 |
| `min_segment_len` | 第7节 | 4 帧 | 最短段长度 |
| `residual_threshold` | 第9节 | 0.5 m | 异常点判定阈值 |
| SG window/polyorder | 第8节 | 7/2 | 滤波窗口和多项式阶数 |

## 网球物理约束参考

| 参数 | 值 | 说明 |
|------|-----|------|
| 球场宽 | 8.23 m | 单打场地 |
| 球场长 | 23.77 m | |
| 球网高度 | 0.914 m | 中点处 |
| 最高球速 | ~70 m/s | 发球极限 |
| 帧率 | 25 fps | |
| 每帧最大位移 | ~2.8 m | 70/25 |

## 依赖

- numpy, pandas, matplotlib, scipy
- plotly（交互式 3D 可视化）
- 项目模块：`app.pipeline.homography`, `app.triangulation`
