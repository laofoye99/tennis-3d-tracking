# 3D 轨迹分析指南

## 概述

`notebooks/trajectory_analysis.ipynb` 用于分析双摄像头网球追踪系统的 3D 轨迹质量，
测试轨迹拟合与噪声过滤方法。

## 数据流

```
cam66 像素检测 ──→ 单应性投影 ──→ 世界坐标 (x,y) ──┐
                                                    ├→ 三角测量 → 3D (x,y,z)
cam68 像素检测 ──→ 单应性投影 ──→ 世界坐标 (x,y) ──┘
                                                          │
                                                          ▼
                                               速度计算 → 轨迹分段 → 段内滤波
```

## 数据来源

| 数据 | 路径 | 说明 |
|------|------|------|
| cam66 检测 | `uploads/cam66_20260307_173403_2min/` | 帧0-341含GT（score=None）和模型输出 |
| cam68 检测 | `uploads/cam68_20260307_173403_2min/` | 同上 |
| 单应性矩阵 | `src/homography_matrices.json` | cam66/cam68 的 H_image_to_world |
| 相机位置 | `config.yaml` → cameras | cam66: (3.77, -15.73, 7.0), cam68: (5.81, 45.04, 7.0) |

### 数据格式

检测数据为 LabelMe JSON 格式，区分方式：
- **GT（手标）**：`shapes[0].score = null`
- **模型输出**：`shapes[0].score = float`（blob_sum 值）

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

三种视图：
- **俯视图** (X-Y)：球场平面，叠加球场线
- **侧视图** (Y-Z)：高度随球场长度变化
- **3D 视图**：完整空间轨迹，含球场和相机位置

### 第6节：噪声分析

计算帧间 3D 速度（m/s），网球物理极限约 70 m/s（发球），
超过此速度的帧很可能是误检或噪声。

### 第7节：轨迹分段

按以下条件将连续轨迹切分为独立飞行段：
- 帧间速度突变（超过阈值 → 击球/弹地/误检）
- 帧间隔过大（连续丢帧 → 轨迹中断）

每段内部近似恒定加速度（重力 + 空气阻力），可独立滤波。

### 第8节：段内滤波方法对比

三种方法（均为后处理，不反馈回检测）：

| 方法 | 参数 | 特点 |
|------|------|------|
| 中值滤波 | kernel=5 | 抗单帧跳点，不改变整体趋势 |
| Savitzky-Golay | window=7, poly=2 | 保轨迹形态，匹配抛物线 |
| 多项式拟合 | z:2次, xy:1次 | 全局拟合，z轴用抛物线 |

### 第9节：异常点识别

通过 SG 滤波残差识别噪声点：原始值与滤波值偏差超过阈值（默认 0.5m）的帧标记为异常。

### 第10节：统计摘要

输出全局和各段的统计信息：坐标范围、速度统计、异常帧数。

## 使用方法

```bash
cd notebooks/
jupyter notebook trajectory_analysis.ipynb
```

### 关键参数调整

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `FRAME_START/END` | 第1节 | 0/341 | 分析帧范围 |
| `speed_threshold` | 第7节 | 50.0 m/s | 分段速度阈值 |
| `gap_threshold` | 第7节 | 5 帧 | 分段间隔阈值 |
| `min_segment_len` | 第7节 | 4 帧 | 最短段长度 |
| `residual_threshold` | 第9节 | 0.5 m | 异常点判定阈值 |
| SG window/polyorder | 第8节 | 7/2 | 滤波窗口和多项式阶数 |

## 依赖

- numpy, pandas, matplotlib, scipy
- 项目模块：`app.pipeline.homography`, `app.triangulation`
