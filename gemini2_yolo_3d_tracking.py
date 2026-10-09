#!/usr/bin/env python3
"""
3D Object Tracking với Orbbec Gemini 2 + YOLO (Ultralytics) + pyorbbecsdk

Pipeline:
  Gemini 2 (Color + Depth, align D2C)
    -> YOLO + ByteTrack (bbox, class, track ID trên ảnh màu)
    -> lấy depth ổn định trong bbox/mask (median)
    -> back-projection (u, v, Z) -> (X, Y, Z) mét, hệ toạ độ camera
    -> Kalman filter 3D (constant velocity) cho mỗi track ID
    -> hiển thị: ảnh màu + depth + bản đồ nhìn từ trên xuống (BEV)

Cài đặt:
    pip install ultralytics opencv-python numpy
    # pyorbbecsdk: build/cài theo https://github.com/orbbec/pyorbbecsdk
    # (Linux cần cài udev rules cho camera)

Chạy:
    python gemini2_yolo_3d_tracking.py                       # YOLOv8n, mọi class
    python gemini2_yolo_3d_tracking.py --classes 0           # chỉ người
    python gemini2_yolo_3d_tracking.py --model yolov8n-seg.pt  # dùng mask để lấy depth chính xác hơn
    Nhấn 'q' hoặc ESC để thoát.
"""
import argparse
from pathlib import Path
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode,
)

# ----------------------------------------------------------------------------
# Tham số
# ----------------------------------------------------------------------------
DEPTH_MIN_MM = 150      # Gemini 2 đo tốt từ ~0.15 m
DEPTH_MAX_MM = 8000
MAX_MISSED = 15         # số frame mất dấu tối đa trước khi xoá track
TRAIL_LEN = 40


# ----------------------------------------------------------------------------
# Tiện ích Orbbec
# ----------------------------------------------------------------------------
def frame_to_bgr(frame):
    """Chuyển color frame của Orbbec sang ảnh BGR (OpenCV)."""
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.I420:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
    print(f"[WARN] Định dạng màu chưa hỗ trợ: {fmt}")
    return None


def depth_to_mm(depth_frame):
    """Chuyển depth frame sang mảng float32 đơn vị mm."""
    w, h = depth_frame.get_width(), depth_frame.get_height()
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
    scale = depth_frame.get_depth_scale()
    return raw.astype(np.float32) * scale


def _select_video_profile(profile_list, preferred_formats, width, height, fps):
    """Choose a supported profile, preferring the requested size/rate/format."""
    candidates = []
    for i in range(len(profile_list)):
        profile = profile_list[i]
        try:
            fmt = profile.get_format()
            w = profile.get_width()
            h = profile.get_height()
            rate = profile.get_fps()
        except AttributeError:
            continue
        if fmt not in preferred_formats:
            continue
        # Prefer the requested format, then the closest size and frame rate.
        score = (
            abs(w - width) + abs(h - height),
            abs(rate - fps),
            preferred_formats.index(fmt),
        )
        candidates.append((score, profile))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def open_camera(align_mode="sw", color_w=640, color_h=480, fps=30):
    pipeline = Pipeline()
    config = Config()

    # --- Color ---
    color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = _select_video_profile(
        color_list, (OBFormat.RGB, OBFormat.BGR, OBFormat.MJPG),
        color_w, color_h, fps,
    )
    if color_profile is None:
        raise RuntimeError("Orbbec device has no supported RGB/BGR/MJPG color profile")
    config.enable_stream(color_profile)

    # --- Depth ---
    depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = _select_video_profile(
        depth_list, (OBFormat.Y16, OBFormat.Y14, OBFormat.Y11),
        color_w, color_h, fps,
    )
    if depth_profile is None:
        raise RuntimeError("Orbbec device has no supported Y16/Y14/Y11 depth profile")
    config.enable_stream(depth_profile)

    # --- Align depth -> color (D2C) ---
    try:
        config.set_align_mode(
            OBAlignMode.HW_MODE if align_mode == "hw" else OBAlignMode.SW_MODE)
    except Exception as e:
        print(f"[WARN] Không đặt được align mode: {e}")

    pipeline.start(config)
    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass
    return pipeline


def get_color_intrinsics(pipeline, img_w, img_h):
    """Lấy fx, fy, cx, cy của camera màu, scale theo độ phân giải ảnh đang dùng."""
    param = pipeline.get_camera_param()
    intr = param.rgb_intrinsic  # sau D2C, depth cùng hệ toạ độ với màu
    sx = img_w / float(intr.width)
    sy = img_h / float(intr.height)
    return intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy


# ----------------------------------------------------------------------------
# Hình học 3D
# ----------------------------------------------------------------------------
def deproject(u, v, z_m, fx, fy, cx, cy):
    """Pixel + depth(m) -> điểm 3D (X phải, Y xuống, Z hướng ra trước) theo mét."""
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return np.array([x, y, z_m], dtype=np.float64)


def robust_depth_mm(depth_mm, box, mask=None):
    """Depth đại diện cho vật thể: median của các pixel hợp lệ."""
    h, w = depth_mm.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    if mask is not None:
        roi = depth_mm[mask > 0]
    else:
        # chỉ lấy vùng trung tâm bbox để tránh dính background
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw = max(int((x2 - x1) * 0.15), 2)
        rh = max(int((y2 - y1) * 0.15), 2)
        roi = depth_mm[max(cy - rh, 0):cy + rh + 1, max(cx - rw, 0):cx + rw + 1]

    vals = roi[(roi > DEPTH_MIN_MM) & (roi < DEPTH_MAX_MM)]
    if vals.size < 10:
        return None
    return float(np.median(vals))


# ----------------------------------------------------------------------------
# Kalman filter 3D (constant velocity)
# state = [x, y, z, vx, vy, vz]
# ----------------------------------------------------------------------------
class KalmanTrack3D:
    def __init__(self, pos, t, q=1.0, r=0.03):
        self.x = np.zeros(6)
        self.x[:3] = pos
        self.P = np.diag([0.05] * 3 + [1.0] * 3)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.R = np.eye(3) * r ** 2
        self.q = q
        self.t = t
        self.missed = 0
        self.trail = deque(maxlen=TRAIL_LEN)
        self.trail.append(pos.copy())

    def _predict(self, dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        # nhiễu quá trình theo mô hình gia tốc ngẫu nhiên
        g = np.vstack([np.eye(3) * 0.5 * dt ** 2, np.eye(3) * dt])
        Q = g @ g.T * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def step(self, t, meas=None):
        dt = max(t - self.t, 1e-3)
        self.t = t
        self._predict(dt)
        if meas is not None:
            y = meas - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P
            self.missed = 0
        else:
            self.missed += 1
        self.trail.append(self.x[:3].copy())

    @property
    def pos(self):
        return self.x[:3]

    @property
    def vel(self):
        return self.x[3:]


# ----------------------------------------------------------------------------
# Vẽ
# ----------------------------------------------------------------------------
def id_color(tid):
    rng = np.random.RandomState(int(tid) * 7919 % 2 ** 31)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_bev(tracks, labels, size=420, x_range=3.0, z_range=6.0):
    """Bản đồ nhìn từ trên xuống: trục ngang = X, trục dọc = Z."""
    bev = np.full((size, size, 3), 30, np.uint8)
    # lưới 1 m
    for m in range(0, int(z_range) + 1):
        y = size - int(m / z_range * size)
        cv2.line(bev, (0, y), (size, y), (60, 60, 60), 1)
        cv2.putText(bev, f"{m}m", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    for m in range(-int(x_range), int(x_range) + 1):
        x = int((m + x_range) / (2 * x_range) * size)
        cv2.line(bev, (x, 0), (x, size), (60, 60, 60), 1)
    # camera
    cv2.drawMarker(bev, (size // 2, size - 4), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)

    def to_px(p):
        px = int((p[0] + x_range) / (2 * x_range) * size)
        py = size - int(p[2] / z_range * size)
        return px, py

    for tid, trk in tracks.items():
        col = id_color(tid)
        pts = [to_px(p) for p in trk.trail]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(bev, a, b, col, 1)
        cv2.circle(bev, to_px(trk.pos), 6, col, -1)
        cv2.putText(bev, f"{labels.get(tid, '')}#{tid}", (to_px(trk.pos)[0] + 8, to_px(trk.pos)[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.putText(bev, "Top-down (BEV)", (size - 130, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return bev


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt / yolov8n-seg.pt / yolo11n.pt ...")
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="lọc class, vd: 0 = person")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml hoặc botsort.yaml")
    ap.add_argument("--align", choices=["sw", "hw"], default="sw")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="vd: 0 (GPU) hoặc cpu")
    args = ap.parse_args()

    # Resolve local weights relative to this script as well as the launch
    # directory. VS Code's working directory can differ from the script's
    # folder, which otherwise makes the default yolov8n.pt look missing.
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        launch_path = Path.cwd() / model_path
        script_path = Path(__file__).resolve().parent / model_path
        if launch_path.is_file():
            model_path = launch_path
        elif script_path.is_file():
            model_path = script_path
    model = YOLO(str(model_path))
    names = model.names

    pipeline = open_camera(args.align, args.width, args.height, args.fps)
    tracks: dict[int, KalmanTrack3D] = {}
    labels: dict[int, str] = {}
    intr = None
    fps_hist = deque(maxlen=20)

    print("Đang chạy... nhấn 'q' hoặc ESC để thoát.")
    try:
        while True:
            t0 = time.time()
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            color = frame_to_bgr(color_frame)
            if color is None:
                continue
            depth_mm = depth_to_mm(depth_frame)
            H, W = color.shape[:2]
            if depth_mm.shape != (H, W):  # phòng khi chưa align cùng kích thước
                depth_mm = cv2.resize(depth_mm, (W, H), interpolation=cv2.INTER_NEAREST)

            if intr is None:
                intr = get_color_intrinsics(pipeline, W, H)
                print(f"Intrinsics (fx, fy, cx, cy) = {intr}")
            fx, fy, cx, cy = intr

            now = time.time()

            # ---------------- YOLO + tracker ----------------
            res = model.track(color, persist=True, tracker=args.tracker, conf=args.conf,
                              classes=args.classes, imgsz=args.imgsz, device=args.device,
                              verbose=False)[0]

            seen = set()
            vis = color.copy()

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.int().cpu().numpy()
                clss = res.boxes.cls.int().cpu().numpy()
                polys = res.masks.xy if res.masks is not None else None

                for i, (box, tid, c) in enumerate(zip(boxes, ids, clss)):
                    tid = int(tid)
                    mask = None
                    if polys is not None and len(polys[i]) >= 3:
                        mask = np.zeros((H, W), np.uint8)
                        cv2.fillPoly(mask, [polys[i].astype(np.int32)], 1)

                    z_mm = robust_depth_mm(depth_mm, box, mask)
                    if z_mm is None:
                        continue

                    # pixel đại diện: tâm bbox (hoặc trọng tâm mask)
                    if mask is not None:
                        ys, xs = np.nonzero(mask)
                        u, v = float(xs.mean()), float(ys.mean())
                    else:
                        u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                    p3d = deproject(u, v, z_mm / 1000.0, fx, fy, cx, cy)

                    if tid not in tracks:
                        tracks[tid] = KalmanTrack3D(p3d, now)
                    else:
                        tracks[tid].step(now, p3d)
                    labels[tid] = names[int(c)]
                    seen.add(tid)

                    # ---- vẽ ----
                    col = id_color(tid)
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                    if mask is not None:
                        overlay = vis.copy()
                        overlay[mask > 0] = col
                        vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
                    trk = tracks[tid]
                    speed = float(np.linalg.norm(trk.vel))
                    X, Y, Z = trk.pos
                    txt1 = f"{labels[tid]} #{tid}"
                    txt2 = f"XYZ=({X:+.2f},{Y:+.2f},{Z:.2f})m v={speed:.2f}m/s"
                    cv2.putText(vis, txt1, (x1, max(y1 - 18, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                    cv2.putText(vis, txt2, (x1, max(y1 - 4, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                    cv2.circle(vis, (int(u), int(v)), 4, col, -1)

            # track bị mất: chỉ dự đoán, xoá nếu mất quá lâu
            for tid in list(tracks.keys()):
                if tid not in seen:
                    tracks[tid].step(now, None)
                    if tracks[tid].missed > MAX_MISSED:
                        del tracks[tid]
                        labels.pop(tid, None)

            # ---------------- Hiển thị ----------------
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(np.clip(depth_mm, 0, 5000), alpha=255.0 / 5000), cv2.COLORMAP_JET)
            depth_vis[depth_mm == 0] = 0

            fps_hist.append(1.0 / max(time.time() - t0, 1e-6))
            cv2.putText(vis, f"FPS: {np.mean(fps_hist):.1f}  tracks: {len(tracks)}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            bev = draw_bev(tracks, labels, size=H)
            canvas = np.hstack([vis, depth_vis, bev])
            cv2.imshow("Gemini 2 - YOLO 3D Tracking  [Color | Depth | BEV]", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
