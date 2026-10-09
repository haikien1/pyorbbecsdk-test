"""User-selected ROI optical flow and a small constant-velocity 3D tracker."""
from typing import Optional, Tuple
import cv2
import numpy as np
from .depth_utils import Intrinsics, deproject_pixel_to_3d, robust_depth_m


class FeatureTracker2D:
    def __init__(self, max_features=120):
        self.max_features = max_features
        self.prev_gray = None
        self.points = None
        self.box = None

    def initialize(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = map(int, box)
        if w < 2 or h < 2:
            return False
        self.box = (x, y, x+w, y+h)
        self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(self.prev_gray.shape, np.uint8)
        mask[max(y, 0):min(y+h, mask.shape[0]), max(x, 0):min(x+w, mask.shape[1])] = 255
        self.points = cv2.goodFeaturesToTrack(self.prev_gray, self.max_features, 0.01, 5, mask=mask)
        return self.points is not None and len(self.points) >= 4

    def update(self, image: np.ndarray) -> Tuple[bool, Optional[Tuple[int,int,int,int]], int, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.points is None or self.prev_gray is None or self.box is None:
            return False, self.box, 0, float("inf")
        nxt, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.points, None,
                                                   winSize=(21,21), maxLevel=3,
                                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if nxt is None or status is None:
            self.points = None
            return False, self.box, 0, float("inf")
        good_old = self.points[status.ravel() == 1].reshape(-1, 2)
        good_new = nxt[status.ravel() == 1].reshape(-1, 2)
        errors = err[status.ravel() == 1].reshape(-1) if err is not None else np.zeros(len(good_new))
        acceptable = np.isfinite(errors) & (errors < 35)
        good_old, good_new = good_old[acceptable], good_new[acceptable]
        mean_error = float(np.mean(errors[acceptable])) if np.any(acceptable) else float("inf")
        if len(good_new) >= 4:
            # Robust translation from median feature displacement; preserve the selected box size.
            delta = np.median(good_new - good_old, axis=0)
            x1, y1, x2, y2 = self.box
            h, w = gray.shape
            x1 = int(np.clip(round(x1 + delta[0]), 0, w-1)); y1 = int(np.clip(round(y1 + delta[1]), 0, h-1))
            bw, bh = x2-self.box[0], y2-self.box[1]
            x2, y2 = min(x1+bw, w), min(y1+bh, h)
            self.box = (x1,y1,x2,y2)
            self.points = good_new.reshape(-1,1,2).astype(np.float32)
            # Replenish features while keeping track inside the current ROI.
            if len(self.points) < self.max_features // 2:
                mask = np.zeros(gray.shape, np.uint8); mask[y1:y2, x1:x2] = 255
                extra = cv2.goodFeaturesToTrack(gray, self.max_features-len(self.points), .01, 5, mask=mask)
                if extra is not None:
                    self.points = np.concatenate([self.points, extra], axis=0)
            self.prev_gray = gray
            return True, self.box, len(good_new), mean_error
        self.points = good_new.reshape(-1,1,2) if len(good_new) else None
        self.prev_gray = gray
        return False, self.box, len(good_new), mean_error


class MotionEstimator:
    """Constant velocity XYZ model with smoothed measured velocity."""
    def __init__(self, alpha=0.35):
        self.position = None
        self.velocity = np.zeros(3, dtype=np.float64)
        self.alpha = alpha
        self.last_time = None

    def update(self, measurement: Optional[np.ndarray], timestamp: float):
        if measurement is None:
            if self.position is not None and self.last_time is not None:
                dt = max(timestamp-self.last_time, 0.0)
                self.position = self.position + self.velocity*dt
            self.last_time = timestamp
            return
        if self.position is None or self.last_time is None:
            self.position = measurement.astype(np.float64); self.last_time = timestamp; return
        dt = max(timestamp-self.last_time, 1e-3)
        measured_velocity = (measurement-self.position)/dt
        self.velocity = (1-self.alpha)*self.velocity + self.alpha*measured_velocity
        self.position = measurement.astype(np.float64)
        self.last_time = timestamp


class DepthEstimator:
    def measure(self, depth_m, box):
        return robust_depth_m(depth_m, box)


class Tracker3D:
    def __init__(self):
        self.features = FeatureTracker2D()
        self.depth = DepthEstimator()
        self.motion = MotionEstimator()
        self.status = "REINITIALIZING"
        self.lost_frames = 0
        self.box = None

    def initialize(self, color, depth, roi, intrinsics, timestamp):
        ok = self.features.initialize(color, roi)
        if not ok:
            self.status = "LOST"; return False
        self.box = self.features.box
        z = self.depth.measure(depth, self.box)
        if z is None:
            self.status = "LOST"; return False
        x1,y1,x2,y2 = self.box
        p = deproject_pixel_to_3d((x1+x2)/2, (y1+y2)/2, z, intrinsics)
        self.motion = MotionEstimator()
        self.motion.update(p, timestamp)
        self.status = "TRACKING"; self.lost_frames = 0
        return True

    def update(self, color, depth, intrinsics, timestamp):
        ok, box, count, error = self.features.update(color)
        if box is not None: self.box = box
        z = self.depth.measure(depth, box) if ok and box is not None else None
        measurement = None
        if z is not None and box is not None:
            x1,y1,x2,y2 = box
            measurement = deproject_pixel_to_3d((x1+x2)/2, (y1+y2)/2, z, intrinsics)
            old = self.motion.position
            if old is not None and np.linalg.norm(measurement-old) > 1.0:
                measurement = None
        if measurement is None:
            self.lost_frames += 1
            self.status = "LOST" if self.lost_frames >= 5 else "LOW CONFIDENCE"
        else:
            self.lost_frames = 0; self.status = "TRACKING"
        self.motion.update(measurement, timestamp)
        return count, error
