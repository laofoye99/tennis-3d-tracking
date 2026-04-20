"""Render GT match_ball demo with minimap bounces and net crossing speed."""
import cv2, json, glob, os, numpy as np
from scipy.optimize import minimize

def compute_homography(src, dst):
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 5.0)
    return H

# Use pre-computed H matrices directly (V2 coords, origin at court center)
import json as _json
with open('src/homography_matrices.json') as _f:
    _hdata = _json.load(_f)
H1 = np.array(_hdata['cam66']['H_image_to_world'])
H2 = np.array(_hdata['cam68']['H_image_to_world'])

def tp(pt, H):
    p = np.array([pt[0], pt[1], 1.0]); t = H @ p; return (t/t[2])[:2]

# V2 coordinates from config.yaml
import yaml
with open('config.yaml') as _f:
    _cfg = yaml.safe_load(_f)
camera_1 = np.array(_cfg['cameras']['cam66']['position_3d'])
camera_2 = np.array(_cfg['cameras']['cam68']['position_3d'])
print(f'Camera positions from config: cam66={camera_1.tolist()}, cam68={camera_2.tolist()}')
NET_Y = 0.0
HW = 4.115    # half width
HL = 11.89    # half length
SX_MIN, SX_MAX, CL = -HW, HW, 2*HL

def calc_3d(v1_w, v2_w):
    d1 = np.append(v1_w, 0) - camera_1
    d2 = np.append(v2_w, 0) - camera_2
    def dist(p): s,t=p; return np.linalg.norm((camera_1+s*d1)-(camera_2+t*d2))
    def cons(p): s,t=p; return min(camera_1[2]+s*d1[2], camera_2[2]+t*d2[2])
    res = minimize(dist,[0.5,0.5],constraints=({'type':'ineq','fun':cons}),bounds=[(0,1),(0,1)])
    s,t=res.x; return ((camera_1+s*d1)+(camera_2+t*d2))/2

folder66 = 'uploads/cam66_20260307_173403_2min'
folder68 = 'uploads/cam68_20260307_173403_2min'

def get_ball(folder, fi, require_mb=True):
    path = os.path.join(folder, f'{fi:05d}.json')
    if not os.path.exists(path): return None, None, None
    with open(path) as f: data = json.load(f)
    for s in data.get('shapes', []):
        desc = s.get('description', '').lower().replace('\uff0c', ',')
        label = s.get('label', '').lower()
        if require_mb and 'match_ball' not in desc: continue
        if not require_mb and label != 'ball' and 'ball' not in desc: continue
        pts = s['points']
        if s['shape_type'] == 'rectangle':
            return (pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2, desc
        elif s['shape_type'] == 'point':
            return pts[0][0], pts[0][1], desc
    return None, None, None

# Collect data
mb = {}
for fi in range(3000):
    px66, py66, desc = get_ball(folder66, fi, True)
    if px66 is None: continue
    px68, py68, _ = get_ball(folder68, fi, False)
    entry = {'px66': px66, 'py66': py66, 'desc': desc or ''}
    w66 = tp([px66, py66], H1)
    entry['wx66'] = float(w66[0]); entry['wy66'] = float(w66[1])
    if px68 is not None:
        w68 = tp([px68, py68], H2)
        pt3d = calc_3d(w66, w68)
        entry.update({'px68': px68, 'py68': py68, 'x': float(pt3d[0]), 'y': float(pt3d[1]), 'z': float(pt3d[2]), 'has_3d': True})
    else:
        entry.update({'x': float(w66[0]), 'y': float(w66[1]), 'z': 0.0, 'has_3d': False})
    mb[fi] = entry

print(f'Match ball: {len(mb)} frames')

# Pre-compute events
bounces = []
net_crossings = []
prev_fi = None; prev_y = None

for fi in sorted(mb.keys()):
    d = mb[fi]
    tags = [t.strip() for t in d['desc'].split(',') if t.strip() and t.strip() != 'match_ball' and not t.startswith('blob') and not t.startswith('conf')]

    if any(t in ('bounce', 'boounce') for t in tags):
        bounces.append({'frame': fi, 'x': d['x'], 'y': d['y'], 'in_court': True})
    elif 'bounce_out' in tags:
        bounces.append({'frame': fi, 'x': d['x'], 'y': d['y'], 'in_court': False})

    if prev_fi is not None and prev_y is not None:
        curr_y = d['wy66']
        if (prev_y < NET_Y and curr_y >= NET_Y) or (prev_y > NET_Y and curr_y <= NET_Y):
            dt = (fi - prev_fi) / 25.0
            if 0 < dt < 2.0:
                dx = d['wx66'] - mb[prev_fi]['wx66']
                dy = curr_y - prev_y
                speed = np.sqrt(dx**2 + dy**2) / dt * 3.6
                if speed < 250:
                    net_crossings.append({'frame': fi, 'speed_kmh': min(speed, 150)})
    prev_fi = fi; prev_y = d['wy66']

print(f'Bounces: {len(bounces)}, Net crossings: {len(net_crossings)}')

# Render
W, H = 1920, 1080; cam_w, cam_h = 840, 472; court_w = 240

def w2p(wx, wy, cw, ch):
    # V2: x=[-HW,+HW], y=[-HL,+HL], origin at center
    px = int((wx + HW) / (2 * HW) * cw)
    py = int((HL - wy) / (2 * HL) * ch)
    return np.clip(px, 0, cw-1), np.clip(py, 0, ch-1)

def draw_court(img):
    ch, cw = img.shape[:2]; img[:] = (30, 80, 30)
    SVC = 6.4
    for x1,y1,x2,y2 in [(-HW,-HL,-HW,HL),(HW,-HL,HW,HL),(-HW,-HL,HW,-HL),
                          (-HW,HL,HW,HL),(-HW,-SVC,HW,-SVC),
                          (-HW,SVC,HW,SVC),(0,-SVC,0,SVC),
                          (-HW-0.3,0,HW+0.3,0)]:
        cv2.line(img, w2p(x1,y1,cw,ch), w2p(x2,y2,cw,ch), (255,255,255), 1)

cap66 = cv2.VideoCapture('uploads/cam66_20260307_173403_2min.mp4')
cap68 = cv2.VideoCapture('uploads/cam68_20260307_173403_2min.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('exports/tracking_gt_full.mp4', fourcc, 25.0, (cam_w*2+court_w, cam_h))

trail66, trail68 = [], []
active_speed = None; speed_fade = 0

for fi in range(3000):
    ret66, f66 = cap66.read(); ret68, f68 = cap68.read()
    if not ret66 or not ret68: break
    r66 = cv2.resize(f66, (cam_w, cam_h)); r68 = cv2.resize(f68, (cam_w, cam_h))
    court = np.zeros((cam_h, court_w, 3), dtype=np.uint8); draw_court(court)

    if fi in mb:
        d = mb[fi]
        sx = int(d['px66']*cam_w/W); sy = int(d['py66']*cam_h/H)
        trail66.append((sx,sy))
        if len(trail66) > 25: trail66.pop(0)
        for i,(tx,ty) in enumerate(trail66):
            a=(i+1)/len(trail66); cv2.circle(r66,(tx,ty),max(1,int(3*a)),(0,int(200*a),int(255*a)),-1)
        cv2.circle(r66,(sx,sy),10,(0,255,255),2); cv2.circle(r66,(sx,sy),3,(0,255,255),-1)

        if d.get('px68'):
            sx68=int(d['px68']*cam_w/W); sy68=int(d['py68']*cam_h/H)
            trail68.append((sx68,sy68))
            if len(trail68)>25: trail68.pop(0)
            for i,(tx,ty) in enumerate(trail68):
                a=(i+1)/len(trail68); cv2.circle(r68,(tx,ty),max(1,int(3*a)),(0,int(200*a),int(255*a)),-1)
            cv2.circle(r68,(sx68,sy68),10,(0,255,255),2); cv2.circle(r68,(sx68,sy68),3,(0,255,255),-1)

    # Net crossing speed
    for nc in net_crossings:
        if nc['frame'] == fi:
            active_speed = nc['speed_kmh']; speed_fade = 50
    if speed_fade > 0:
        alpha = speed_fade / 50
        cv2.putText(r66, f"{active_speed:.0f} km/h", (cam_w//2-80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,int(255*alpha),int(255*alpha)), 3)
        speed_fade -= 1

    # Bounces on minimap (last 5)
    recent = [b for b in bounces if b['frame'] <= fi][-5:]
    for i, b in enumerate(recent):
        bpx, bpy = w2p(b['x'], b['y'], court_w, cam_h)
        color = (0,0,255) if not b['in_court'] else (0,255,0)
        a = (i+1) / len(recent) if recent else 1
        if b['in_court']:
            cv2.circle(court, (bpx,bpy), int(6*a), color, -1)
        else:
            cv2.circle(court, (bpx,bpy), int(6*a), color, 2)
        cv2.putText(court, f"f{b['frame']}", (bpx+8,bpy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

    cv2.putText(r66,'cam66',(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(r68,'cam68',(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(r66,f'F{fi} {fi/25:.1f}s',(10,cam_h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

    out.write(np.hstack([r66,r68,court]))
    if fi%500==0: print(f'  {fi}/3000')

out.release(); cap66.release(); cap68.release()
print(f'Saved: exports/tracking_gt_full.mp4')
for b in bounces:
    print(f'  Bounce f{b["frame"]}: ({b["x"]:.2f},{b["y"]:.2f}) {"IN" if b["in_court"] else "OUT"}')
for nc in net_crossings:
    print(f'  Net crossing f{nc["frame"]}: {nc["speed_kmh"]:.0f} km/h')
