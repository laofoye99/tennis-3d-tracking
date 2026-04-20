"""Simulate real-time pipeline on video files.
Tests the exact same code path as live inference.
Compares 3D BounceDetector vs Pixel BounceDetector vs combined.
"""
import yaml, numpy as np, cv2
from app.pipeline.inference import create_detector
from app.pipeline.postprocess import BallTracker
from app.pipeline.homography import HomographyTransformer
from app.triangulation import triangulate
from app.analytics import BounceDetector, PixelBounceDetector

with open('config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg['model']
detector = create_detector(
    model_cfg['path'], tuple(model_cfg['input_size']),
    model_cfg['frames_in'], model_cfg.get('frames_out', model_cfg['frames_in']),
    model_cfg['device']
)
tracker = BallTracker(threshold=model_cfg.get('threshold', 0.3))

h66 = HomographyTransformer(cfg['homography']['path'], 'cam66')
h68 = HomographyTransformer(cfg['homography']['path'], 'cam68')
cam66_pos = cfg['cameras']['cam66']['position_3d']
cam68_pos = cfg['cameras']['cam68']['position_3d']

# Both detectors
bounce_3d = BounceDetector()
bounce_px66 = PixelBounceDetector(window_size=15, min_margin_px=15, cooldown_frames=8)
bounce_px68 = PixelBounceDetector(window_size=15, min_margin_px=15, cooldown_frames=8)

cap66 = cv2.VideoCapture('uploads/cam66_20260307_173403_2min.mp4')
cap68 = cv2.VideoCapture('uploads/cam68_20260307_173403_2min.mp4')

frames_in = model_cfg['frames_in']
buf66, buf68 = [], []
det66, det68 = {}, {}

print('Phase 1: TrackNet detection...')
for fi in range(1800):
    ret66, f66 = cap66.read()
    ret68, f68 = cap68.read()
    if not ret66 or not ret68: break
    buf66.append(f66); buf68.append(f68)
    if len(buf66) == frames_in:
        hm66 = detector.infer(buf66)
        hm68 = detector.infer(buf68)
        for i, hm in enumerate(hm66):
            d = tracker.process_heatmap(hm)
            if d: det66[fi - frames_in + 1 + i] = d
        for i, hm in enumerate(hm68):
            d = tracker.process_heatmap(hm)
            if d: det68[fi - frames_in + 1 + i] = d
        buf66.clear(); buf68.clear()

cap66.release(); cap68.release()
print(f'  det66={len(det66)}, det68={len(det68)}')

print('\nPhase 2: Stream processing...')

# Net crossing uses homography world_y
net_crossings = []
prev_wy66 = None; prev_fi = None

# Process ALL detected frames (not just common)
all_frames = sorted(set(det66.keys()) | set(det68.keys()))
print(f'  All detected frames: {len(all_frames)}')

bounces_3d_list = []
bounces_px_list = []

for fi in all_frames:
    # --- Pixel bounce detection (single camera, no triangulation needed) ---
    if fi in det66:
        px66, py66, conf66 = det66[fi]
        w66 = h66.pixel_to_world(px66, py66)
        px_pt66 = {
            'pixel_y': py66, 'world_x': w66[0], 'world_y': w66[1],
            'frame_index': fi, 'timestamp': fi / 25.0, 'camera': 'cam66'
        }
        b = bounce_px66.update(px_pt66)
        if b:
            bounces_px_list.append({'frame': b.frame_index, 'x': b.x, 'y': b.y, 'cam': 'cam66', 'in_court': b.in_court})
            print(f'  PX_BOUNCE cam66 f{b.frame_index}: ({b.x:.2f},{b.y:.2f}) {"IN" if b.in_court else "OUT"}')

    if fi in det68:
        px68, py68, conf68 = det68[fi]
        w68 = h68.pixel_to_world(px68, py68)
        px_pt68 = {
            'pixel_y': py68, 'world_x': w68[0], 'world_y': w68[1],
            'frame_index': fi, 'timestamp': fi / 25.0, 'camera': 'cam68'
        }
        b = bounce_px68.update(px_pt68)
        if b:
            bounces_px_list.append({'frame': b.frame_index, 'x': b.x, 'y': b.y, 'cam': 'cam68', 'in_court': b.in_court})
            print(f'  PX_BOUNCE cam68 f{b.frame_index}: ({b.x:.2f},{b.y:.2f}) {"IN" if b.in_court else "OUT"}')

    # --- 3D bounce detection (needs both cameras) ---
    if fi in det66 and fi in det68:
        px66, py66, _ = det66[fi]
        px68, py68, _ = det68[fi]
        w66 = h66.pixel_to_world(px66, py66)
        w68 = h68.pixel_to_world(px68, py68)
        x, y, z = triangulate(w66, w68, cam66_pos, cam68_pos)

        pt3d = {'x': x, 'y': y, 'z': z, 'timestamp': fi/25.0, 'frame_index': fi}
        b = bounce_3d.update(pt3d)
        if b:
            bounces_3d_list.append({'frame': b.frame_index, 'z': b.z})
            print(f'  3D_BOUNCE f{b.frame_index}: z={b.z:.2f} {"IN" if b.in_court else "OUT"}')

    # --- Net crossing (from cam66 homography world_y) ---
    if fi in det66:
        px66, py66, _ = det66[fi]
        w66 = h66.pixel_to_world(px66, py66)
        wy = w66[1]
        if prev_wy66 is not None and prev_fi is not None:
            if (prev_wy66 < 0 and wy >= 0) or (prev_wy66 > 0 and wy <= 0):
                dt = (fi - prev_fi) / 25.0
                if 0 < dt < 2:
                    dx = w66[0] - prev_w66[0]
                    dy = wy - prev_wy66
                    spd = min(np.sqrt(dx**2 + dy**2) / dt * 3.6, 150)
                    net_crossings.append({'frame': fi, 'speed': spd})
                    print(f'  NET_CROSSING f{fi}: {spd:.0f} km/h')
        prev_wy66 = wy; prev_fi = fi; prev_w66 = w66

# --- Evaluation ---
gt = [169, 234, 290, 788, 831, 851, 1498, 1554, 1607, 1651]

def eval_bounces(detected, name):
    tp = 0; mg = set()
    for b in detected:
        for g in gt:
            if abs(b['frame'] - g) <= 10 and g not in mg:
                tp += 1; mg.add(g); break
    fp = len(detected) - tp; fn = len(gt) - tp
    p = tp/(tp+fp) if tp+fp else 0; r = tp/len(gt); f1 = 2*p*r/(p+r) if p+r else 0
    missed = [g for g in gt if g not in mg]
    print(f'  {name:20s}: TP={tp} FP={fp} FN={fn} P={p:.1%} R={r:.1%} F1={f1:.1%}  missed={missed}')

print(f'\n{"="*60}')
print(f'=== RESULTS ===')
print(f'Net crossings: {len(net_crossings)}')
print(f'\nBounce detection comparison (GT has {len(gt)} bounces):')
eval_bounces(bounces_3d_list, '3D BounceDetector')
eval_bounces(bounces_px_list, 'Pixel BounceDetector')

# Combined: union of both (deduplicate within 10 frames)
combined = list(bounces_px_list)
for b3d in bounces_3d_list:
    if not any(abs(b3d['frame'] - bp['frame']) <= 10 for bp in combined):
        combined.append(b3d)
eval_bounces(combined, 'Combined (PX + 3D)')
