"""Realtime single-object 3D tracker for Orbbec Gemini 2.

Run from this directory's parent: python -m tracker_3d.tracker_3d [--hw]
Mouse-select the initial ROI, then press R to select a replacement, M to toggle
RGB/depth view, or Q/Esc to quit.
"""
import argparse
import time
import cv2
import numpy as np
from .camera import OrbbecCamera
from .depth_utils import MIN_DEPTH_M, MAX_DEPTH_M
from .tracking import Tracker3D


def main():
    parser = argparse.ArgumentParser(description="Orbbec Gemini 2 optical-flow 3D tracker")
    parser.add_argument("--hw", action="store_true", help="Use hardware depth-to-color alignment")
    args = parser.parse_args()
    camera = None
    try:
        camera = OrbbecCamera(hardware_align=args.hw)
        tracker = Tracker3D()
        cv2.namedWindow("Gemini 2 3D Tracker", cv2.WINDOW_NORMAL)
        show_depth = False
        previous = time.perf_counter(); fps = 0.0
        while True:
            pair = camera.read()
            if pair is None:
                continue
            color, depth = pair
            now = time.perf_counter()
            instant = 1.0/max(now-previous, 1e-6); previous = now
            fps = .9*fps + .1*instant if fps else instant
            if tracker.status == "REINITIALIZING":
                # First frame: explicit user selection.
                roi = cv2.selectROI("Gemini 2 3D Tracker", color, False, False)
                tracker.initialize(color, depth, roi, camera.intrinsics, now)
                continue
            if tracker.status != "LOST":
                count, flow_error = tracker.update(color, depth, camera.intrinsics, now)
            else:
                count, flow_error = 0, float("inf")
            canvas = color.copy()
            if tracker.box:
                x1,y1,x2,y2 = tracker.box
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0) if tracker.status == "TRACKING" else (0,165,255), 2)
            p, v = tracker.motion.position, tracker.motion.velocity
            if p is not None:
                cv2.putText(canvas, f"XYZ: {p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f} m", (15,30), cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
                cv2.putText(canvas, f"V: {v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f} m/s", (15,58), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.putText(canvas, f"{tracker.status} | {fps:.1f} FPS", (15,88), cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            cv2.putText(canvas, f"features {count} | flow error {flow_error:.1f}", (15,116), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            if show_depth:
                valid = np.isfinite(depth) & (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)
                normalized = np.zeros(depth.shape, np.uint8)
                normalized[valid] = (255*(depth[valid]-MIN_DEPTH_M)/(MAX_DEPTH_M-MIN_DEPTH_M)).astype(np.uint8)
                view = cv2.applyColorMap(255-normalized, cv2.COLORMAP_TURBO)
                if tracker.box:
                    x1,y1,x2,y2=tracker.box; cv2.rectangle(view,(x1,y1),(x2,y2),(255,255,255),2)
                cv2.imshow("Gemini 2 3D Tracker", view)
            else:
                cv2.imshow("Gemini 2 3D Tracker", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            if key == ord('m'): show_depth = not show_depth
            if key == ord('r'):
                roi = cv2.selectROI("Gemini 2 3D Tracker", color, False, False)
                tracker.status = "REINITIALIZING"
                tracker.initialize(color, depth, roi, camera.intrinsics, time.perf_counter())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"3D tracker stopped: {exc}")
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
