"""
网球击球点检测器 v2
==================
基于 world_y 极值检测 + 速度验证 + 近场/远场分离的新策略。

核心原理：
  击球时球到达离击球球员最近的位置（world_y 局部极值），
  同时球拍施加能量导致速度突增。

  与 v1 的关键区别：
  - v1: 纯图像空间的角度反转检测 → 误检多（视角反转被误判为击球）
  - v2: world 坐标空间极值检测 → 利用物理规律，更可靠

策略：
  1. homography 投影 → 图像坐标转世界坐标
  2. world_y 局部极小值 → 候选击球帧
  3. 速度增大确认 → 过滤反弹/噪声
  4. world_y 阈值分离近场/远场击球
"""

import math
import json
from typing import List, Tuple, Optional, Dict


# ============================================================
#  V2: world_y 极值 + 速度确认策略
# ============================================================

def detect_hits_v2(
    frame_numbers: List[int],
    x_img: List[float],
    y_img: List[float],
    H_image_to_world: List[List[float]],
    min_interval: int = 5,
    speed_increase_ratio: float = 1.05,
    detect_side: str = "near",
    near_far_threshold: Optional[float] = None,
    extrema_window: int = 2,
    velocity_change_threshold: float = 0.5,
) -> List[Tuple[int, float, float]]:
    """
    基于 world_y 极值检测网球击球点（v2）。

    参数
    ----
    frame_numbers, x_img, y_img : 图像坐标序列
    H_image_to_world : 3x3 homography 矩阵 (image → world)
    min_interval : 两次击球最小帧间隔，默认 5
    detect_side : 检测哪一侧的击球
        "near"  — 只检测近场（Cam66 用）
        "far"   — 只检测远场（Cam68 用）
        "both"  — 两侧都检测
    near_far_threshold : world_y 阈值，<=阈值视为近场，>阈值视为远场
        若为 None，自动取 world_y 的中位数作为阈值
    extrema_window : 极值检测的半窗口大小，默认 2
    velocity_change_threshold : 速度向量变化阈值，默认 0.5
        变化量 = |v_after - v_before| / max(|v_before|, 1.0)
        > 0.5 表示击球前后速度向量有显著改变（方向或大小）
        设为 0 关闭验证

    返回
    ----
    List[Tuple[int, float, float]]
        检测到的击球点 (帧号, 图像x, 图像y)
    """
    n = len(frame_numbers)
    if n < 2 * extrema_window + 1:
        return []

    # ── 1. 投影到世界坐标 ──
    world_y = [_project_y(x_img[i], y_img[i], H_image_to_world) for i in range(n)]

    # ── 2. 自动阈值（若未指定）──
    if near_far_threshold is None:
        sorted_wy = sorted(world_y)
        near_far_threshold = sorted_wy[len(sorted_wy) // 2]

    # ── 3. 计算帧间速度向量 ──
    vx = [0.0] * (n - 1)
    vy = [0.0] * (n - 1)
    for i in range(n - 1):
        vx[i] = x_img[i + 1] - x_img[i]
        vy[i] = y_img[i + 1] - y_img[i]

    # ── 4. 检测 world_y 局部极小值 ──
    candidates = []

    for i in range(extrema_window, n - extrema_window):
        wy = world_y[i]

        is_min = True
        for j in range(i - extrema_window, i + extrema_window + 1):
            if j == i:
                continue
            if world_y[j] < wy - 1e-6:
                is_min = False
                break
        if not is_min:
            continue

        # 计算极小值深度（与邻域均值的差），过滤浅谷
        neighbors = [world_y[j] for j in range(i - extrema_window, i + extrema_window + 1) if j != i]
        avg_neighbor = sum(neighbors) / len(neighbors)
        depth = avg_neighbor - wy
        if depth < 0.3:
            continue

        candidates.append((i, depth))

    # ── 5. 速度向量变化验证 + 过滤 ──
    hits = []
    last_hit_idx = -min_interval

    for idx, depth in candidates:
        fn = frame_numbers[idx]

        if idx - last_hit_idx < min_interval:
            continue

        # v_before / v_after = 击球前后的局部平均速度向量
        vx_before = _local_avg(vx, idx - 1, backward=True)
        vy_before = _local_avg(vy, idx - 1, backward=True)
        vx_after = _local_avg(vx, idx, backward=False)
        vy_after = _local_avg(vy, idx, backward=False)

        speed_before = math.sqrt(vx_before * vx_before + vy_before * vy_before)
        speed_after  = math.sqrt(vx_after  * vx_after  + vy_after  * vy_after)

        if speed_before < 0.5 and speed_after < 0.5:
            continue

        # 速度向量变化量（同时捕捉方向反转 + 速度突变）
        dvx = vx_after - vx_before
        dvy = vy_after - vy_before
        delta_v = math.sqrt(dvx * dvx + dvy * dvy)
        vel_change = delta_v / max(speed_before, 1.0)

        if vel_change < velocity_change_threshold:
            continue

        # 场地侧过滤
        wy = world_y[idx]
        if detect_side == "near" and wy > near_far_threshold:
            continue
        if detect_side == "far" and wy <= near_far_threshold:
            continue

        hits.append((fn, x_img[idx], y_img[idx]))
        last_hit_idx = idx

    return hits


def _project_y(x_img: float, y_img: float, H: List[List[float]]) -> float:
    """homography 投影：图像坐标 → world_y"""
    u = H[0][0] * x_img + H[0][1] * y_img + H[0][2]
    v = H[1][0] * x_img + H[1][1] * y_img + H[1][2]
    w = H[2][0] * x_img + H[2][1] * y_img + H[2][2]
    if abs(w) < 1e-10:
        return float("inf")
    return v / w


def _local_avg(speeds: List[float], start_idx: int, backward: bool, window: int = 2) -> float:
    """计算局部平均速度"""
    n = len(speeds)
    values = []
    for k in range(window):
        idx = start_idx - k if backward else start_idx + k
        if 0 <= idx < n:
            values.append(speeds[idx])
    return sum(values) / len(values) if values else 0.0


def load_homography(json_path: str, camera: str = "cam66") -> List[List[float]]:
    """从 JSON 文件加载 homography 矩阵"""
    with open(json_path) as f:
        data = json.load(f)
    return data[camera]["H_image_to_world"]


def get_court_net_y(json_path: str) -> float:
    """获取球网 world_y 位置"""
    with open(json_path) as f:
        data = json.load(f)
    return data["court_dimensions"]["net_y_m"]


# ============================================================
#  V1: 保留兼容（纯图像角度反转）
# ============================================================

def detect_hits(
    frame_numbers: List[int],
    x: List[float],
    y: List[float],
    angle_threshold: float = 90.0,
    min_interval: int = 5,
    speed_increase_required: bool = True,
) -> List[Tuple[int, float, float]]:
    """v1 角度反转检测（保留兼容），参数同旧版"""
    n = len(frame_numbers)
    if n < 3:
        return []

    vx = [0.0] * (n - 1)
    vy = [0.0] * (n - 1)
    speeds = [0.0] * (n - 1)
    for i in range(n - 1):
        vx[i] = x[i + 1] - x[i]
        vy[i] = y[i + 1] - y[i]
        speeds[i] = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])

    angle_changes = [0.0] * (n - 2)
    for i in range(1, n - 1):
        sp, sc = speeds[i - 1], speeds[i]
        if sp < 1e-8 or sc < 1e-8:
            continue
        dot = vx[i - 1] * vx[i] + vy[i - 1] * vy[i]
        cos_a = max(-1.0, min(1.0, dot / (sp * sc)))
        angle_changes[i - 1] = math.degrees(math.acos(cos_a))

    hits = []
    last_hit = -min_interval
    for i in range(1, n - 1):
        if angle_changes[i - 1] < angle_threshold:
            continue
        if speed_increase_required and speeds[i] <= speeds[i - 1]:
            continue
        if frame_numbers[i] - last_hit < min_interval:
            continue
        hits.append((frame_numbers[i], x[i], y[i]))
        last_hit = frame_numbers[i]

    return hits


def detect_hits_with_sign(
    frame_numbers: List[int],
    x: List[float],
    y: List[float],
    sign_axis: str = "x",
    angle_threshold: float = 70.0,
    min_interval: int = 5,
    speed_increase_required: bool = True,
) -> List[Tuple[int, float, float]]:
    """v1 角度反转 + 符号反转检测（保留兼容）"""
    n = len(frame_numbers)
    if n < 3:
        return []

    values = x if sign_axis == "x" else y
    vx = [0.0] * (n - 1)
    vy = [0.0] * (n - 1)
    speeds = [0.0] * (n - 1)
    dv = [0.0] * (n - 1)
    for i in range(n - 1):
        vx[i] = x[i + 1] - x[i]
        vy[i] = y[i + 1] - y[i]
        speeds[i] = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
        dv[i] = values[i + 1] - values[i]

    angle_changes = [0.0] * (n - 2)
    for i in range(1, n - 1):
        sp, sc = speeds[i - 1], speeds[i]
        if sp < 1e-8 or sc < 1e-8:
            continue
        dot = vx[i - 1] * vx[i] + vy[i - 1] * vy[i]
        cos_a = max(-1.0, min(1.0, dot / (sp * sc)))
        angle_changes[i - 1] = math.degrees(math.acos(cos_a))

    hits = []
    last_hit = -min_interval
    for i in range(1, n - 1):
        if angle_changes[i - 1] < angle_threshold:
            continue
        if dv[i - 1] * dv[i] >= 0:
            continue
        if speed_increase_required and speeds[i] <= speeds[i - 1]:
            continue
        if frame_numbers[i] - last_hit < min_interval:
            continue
        hits.append((frame_numbers[i], x[i], y[i]))
        last_hit = frame_numbers[i]

    return hits
