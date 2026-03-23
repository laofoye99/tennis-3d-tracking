# TenniSync 模型架构设计方案

## 设计理念

结合 TrackNet V2/V3/V4 的检测能力和 arxiv 2506.05763 的 3D 轨迹推理能力，
设计一个**两阶段 pipeline**：

```
Stage 1: 视觉检测（改进的 TrackNet）
  输入: 连续 N 帧 RGB + 背景帧
  输出: 每帧 heatmap + 置信度 + 运动特征

Stage 2: 3D 轨迹推理（Canonical Ray LSTM）
  输入: 双相机 2D 检测 → canonical ray 参数化
  输出: 3D 坐标 + 事件分类 (fly/bounce/hit/serve) + 轨迹分段 (EoT)
```

---

## Stage 1: 改进的 TrackNet（TenniSync-Det）

### 基于 TrackNet V4 的改进

**V4 核心思想：** 在 V2 的基础上加 Motion Attention，用极小的速度代价换取稳定提升。

**我们的改进点：**

### 1.1 输入设计

```
当前 TrackNet:
  输入 = bg(3ch) + 8帧(24ch) = 27ch, 288×512
  问题: 背景帧是静态的，不随光照变化

改进:
  输入 = 8帧(24ch) + 帧差特征(7ch) = 31ch, 288×512

帧差特征 (7ch):
  - diff_t_t1:  帧t和帧t-1的差 (3ch)  — 短期运动
  - diff_t_t3:  帧t和帧t-3的差 (3ch)  — 中期运动
  - max_diff:   8帧中最大差值 (1ch)     — 运动显著性
```

**为什么去掉背景帧：**
- 背景帧是整段视频的中值，不反映局部光照变化
- 帧差直接提取运动信息，比背景减除更高效
- TrackNet 的核心能力就是从帧差中检测运动目标

### 1.2 Motion Attention（来自 V4）

在 encoder 最后加一个轻量 attention 模块：

```python
class MotionAttention(nn.Module):
    """Temporal attention on frame-difference channels."""
    def __init__(self, channels):
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.query(x).flatten(2)  # B, C//8, HW
        k = self.key(x).flatten(2)
        v = self.value(x).flatten(2)
        attn = torch.softmax(q.transpose(1,2) @ k, dim=-1)
        out = (v @ attn.transpose(1,2)).view_as(x)
        return x + self.gamma * out
```

### 1.3 输出设计

```
当前: 8 个 heatmap (每帧一个)
改进: 8 个 heatmap + 8 个 confidence score + 8 个 motion vector

输出 shape: (batch, 8+8+16, 288, 512) = (batch, 32, 288, 512)
  - heatmap[0:8]:     球的位置热图 (sigmoid)
  - confidence[8:16]:  检测置信度热图 (sigmoid)
  - motion[16:32]:     2D 运动向量场 (dx, dy per frame)
```

**confidence 通道** 的作用：
- 让模型学会对自己的检测打分
- 高置信度 = 确定是球，低置信度 = 不确定（可能是球鞋/反光）
- 后处理直接用这个分数替代 blob_sum

### 1.4 网络结构

保持 TrackNet V2 的 U-Net 结构，修改：
1. 输入通道: 27 → 31 (去掉背景帧，加帧差)
2. Encoder bottleneck 后加 MotionAttention
3. 输出通道: 8 → 32 (heatmap + confidence + motion)
4. 参数量: ~10.5M (比原来多 ~7%)

```
输入 (31, 288, 512)
  │
  ├── Encoder (和 TrackNet V2 相同)
  │   ├── DoubleConv(31→64) → MaxPool
  │   ├── DoubleConv(64→128) → MaxPool
  │   ├── TripleConv(128→256) → MaxPool
  │   └── TripleConv(256→512) [bottleneck]
  │
  ├── MotionAttention(512) ← 新增
  │
  ├── Decoder (和 TrackNet V2 相同)
  │   ├── Up + TripleConv(768→256)
  │   ├── Up + DoubleConv(384→128)
  │   └── Up + DoubleConv(192→64)
  │
  └── Output Heads
      ├── heatmap_head: Conv(64→8) + sigmoid
      ├── confidence_head: Conv(64→8) + sigmoid
      └── motion_head: Conv(64→16) (no activation, raw dx/dy)
```

---

## Stage 2: 3D 轨迹推理（TenniSync-3D）

### 基于 arxiv 2506.05763 的改进

论文的核心贡献是 **canonical ray 参数化**——把 2D 像素转换成 3D 射线的平面交点，
让 LSTM 直接在 3D 空间学习轨迹模式。

### 2.1 Canonical Ray 参数化

```
对每个检测到的 2D 点 (u, v):

1. 反投影为 3D 射线: r(s) = cam_pos + dir * s
   其中 dir = K_inv @ [u, v, 1]^T (相机内参的逆)

2. 求射线与两个平面的交点:
   - 地面 (z=0): p_ground = (x_g, y_g)
   - 垂直面 (y=0): p_vert = (x_v, z_v)

3. Canonical ray 参数: P = (x_g, y_g, x_v, z_v) ∈ R^4
```

**优势：**
- 不依赖 homography 的精度
- 两个相机的 canonical ray 在同一个 3D 空间中
- LSTM 可以直接学习 3D 物理（重力、弹跳）

### 2.2 网络结构

```
输入 (per frame):
  cam66_ray: (x_g, y_g, x_v, z_v)     4 维
  cam68_ray: (x_g, y_g, x_v, z_v)     4 维
  cam66_conf: confidence score          1 维
  cam68_conf: confidence score          1 维
  cam66_motion: (dx, dy)               2 维
  cam68_motion: (dx, dy)               2 维
  ─────────────────────────────────────
  总计: 14 维 per frame

序列长度: T = 当前轨迹段所有帧
```

### 2.3 三个子网络

**A. EoT 网络（轨迹分段）**

```
输入: canonical ray 序列的时间差 ΔP_t = P_t - P_{t-1}
结构: 2层 BiLSTM (hidden=64) + 3层 FC → sigmoid
输出: ε_t ∈ [0,1] 轨迹结束概率

作用: 自动分割回合/轨迹段
替代: 当前的 rally segmentation 模型
```

**B. 高度预测网络**

```
输入: (ΔP_t, ε_t, h_t^init) 其中 h_t^init = 三角化的初始 z
结构: 2层 BiLSTM (hidden=96) + FC → ReLU (z ≥ 0)
输出: 精确的 z 坐标

作用: 修正三角化的 z 误差（当前 ±0.5m → 目标 ±0.1m）
```

**C. 3D 精修网络**

```
输入: (x_t, y_t, z_refined_t, P_t)
结构: 2层 BiLSTM (hidden=64) + FC
输出: (δx, δy, δz) 坐标修正量

作用: 最终的 3D 坐标精修
```

### 2.4 事件分类头（我们的扩展）

在高度预测网络的输出上加一个分类头：

```
输入: BiLSTM 的 hidden state (每帧)
结构: FC(96→32) + ReLU + FC(32→5)
输出: [fly, bounce, hit, serve, dead] 的概率

作用: 替代 V-shape bounce 检测和 rally segmentation
```

### 2.5 损失函数

```
L_total = λ_ε * L_eot       (轨迹分段 BCE)
        + λ_3D * L_3d       (3D 坐标 L2)
        + λ_B * L_below     (地面以下惩罚: max(0, -z)²)
        + λ_evt * L_event   (事件分类 CE)
        + λ_bounce * L_bz   (bounce 帧的 z 应该 ≈ 0)

超参数: (λ_ε, λ_3D, λ_B, λ_evt, λ_bounce) = (10, 1, 10, 5, 5)
```

---

## 训练数据策略

### 真实数据
- cam66/cam68 已标注的 1800 帧 (match_ball GT)
- 新处理的 999/ 视频 3000 帧 (TrackNet 自动标注 + 人工校正)

### 合成数据（Unity）
- 500,000 帧 / 558 bounces（已生成）
- 需要噪声注入匹配真实误差分布

### 数据使用
```
Stage 1 (TrackNet-Det):
  训练: 真实数据 (有像素级标注)
  微调: 合成数据可用于预训练

Stage 2 (3D LSTM):
  预训练: 合成数据 (完美 3D GT)
  微调: 真实数据 (有 match_ball + bounce GT)
```

---

## 实施计划

### Phase 1: 先跑通基础版本
1. 保持当前 TrackNet 不变（Stage 1 不改）
2. 实现 Canonical Ray 转换
3. 实现简化版 Stage 2（只有高度预测 + 事件分类）
4. 用合成数据训练
5. 在真实数据上验证

### Phase 2: 改进 Stage 1
1. 加入帧差输入
2. 加入 MotionAttention
3. 加入 confidence 输出头
4. 在真实数据上微调

### Phase 3: 完整 Stage 2
1. 加入 EoT 网络
2. 加入 3D 精修网络
3. 端到端微调

---

## 预期指标

| 指标 | 当前 | Phase 1 目标 | Phase 3 目标 |
|------|:---:|:---:|:---:|
| Detection recall | 99.4% | 99.4% | 99.5%+ |
| Bounce F1 | 78.6% | 85% | 92%+ |
| Z precision | ±0.5m | ±0.2m | ±0.1m |
| IN/OUT accuracy | ~85% | 90% | 95%+ |
| Rally segmentation | 95.5% | - | 97%+ (EoT) |
| 实时性 | ~10ms/帧 | ~12ms/帧 | ~15ms/帧 |
