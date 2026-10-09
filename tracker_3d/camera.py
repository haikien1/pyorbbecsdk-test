"""Orbbec Gemini RGB/depth acquisition via pyorbbecsdk."""
from typing import Optional, Tuple

import cv2
import numpy as np
from pyorbbecsdk import (AlignFilter, Config, Context, OBAlignMode, OBFormat,
                         OBSensorType, OBStreamType, Pipeline)

from .depth_utils import Intrinsics


def _choose_profile(profiles, formats, width=640, height=480, fps=30):
    candidates = []
    for i in range(len(profiles)):
        p = profiles[i]
        try:
            if p.get_format() in formats:
                score = (abs(p.get_width()-width)+abs(p.get_height()-height), abs(p.get_fps()-fps))
                candidates.append((score, p))
        except (AttributeError, RuntimeError):
            continue
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def frame_to_bgr(frame) -> np.ndarray:
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    conversions = {OBFormat.YUYV: cv2.COLOR_YUV2BGR_YUY2, OBFormat.UYVY: cv2.COLOR_YUV2BGR_UYVY,
                   OBFormat.I420: cv2.COLOR_YUV2BGR_I420, OBFormat.NV12: cv2.COLOR_YUV2BGR_NV12,
                   OBFormat.NV21: cv2.COLOR_YUV2BGR_NV21}
    if fmt in conversions:
        shape = (h, w, 2) if fmt in (OBFormat.YUYV, OBFormat.UYVY) else (h * 3 // 2, w)
        return cv2.cvtColor(data.reshape(shape), conversions[fmt])
    raise RuntimeError(f"Unsupported color frame format: {fmt}")


class OrbbecCamera:
    def __init__(self, hardware_align: bool = False, timeout_ms: int = 1000):
        self.timeout_ms = timeout_ms
        self.pipeline = Pipeline()
        self.software_align = not hardware_align
        self.align_filter = None
        self.started = False
        context = Context()
        if context.query_devices().get_count() == 0:
            raise RuntimeError("No Orbbec camera found. Connect the Gemini 2 and retry.")
        config = Config()
        color_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color = _choose_profile(color_list, (OBFormat.RGB, OBFormat.BGR, OBFormat.MJPG,
                                             OBFormat.YUYV, OBFormat.UYVY, OBFormat.NV12, OBFormat.NV21, OBFormat.I420))
        depth_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth = _choose_profile(depth_list, (OBFormat.Y16, OBFormat.Y14, OBFormat.Y11))
        if color is None:
            raise RuntimeError("Gemini has no supported color stream profile.")
        if depth is None:
            raise RuntimeError("Gemini has no supported depth stream profile.")
        config.enable_stream(color)
        if hardware_align:
            aligned_profiles = self.pipeline.get_d2c_depth_profile_list(color, OBAlignMode.HW_MODE)
            if len(aligned_profiles) == 0:
                raise RuntimeError("Hardware D2C alignment is not supported for this profile. Run without --hw.")
            config.disable_stream(OBSensorType.DEPTH_SENSOR)
            config.enable_stream(aligned_profiles[0])
            config.set_align_mode(OBAlignMode.HW_MODE)
        else:
            config.enable_stream(depth)
        self.pipeline.start(config)
        self.started = True
        if self.software_align:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        try:
            self.pipeline.enable_frame_sync()
        except (AttributeError, RuntimeError):
            pass
        param = self.pipeline.get_camera_param()
        intr = param.rgb_intrinsic
        self.intrinsics = Intrinsics(float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy))

    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None
        if self.align_filter is not None:
            frames = self.align_filter.process(frames)
            if frames is None:
                return None
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None
        color = frame_to_bgr(color_frame)
        w, h = depth_frame.get_width(), depth_frame.get_height()
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
        # SDK example/API depth scale is millimeters per raw unit. Convert to meters.
        depth_m = raw.astype(np.float32) * float(depth_frame.get_depth_scale()) * 0.001
        if depth_m.shape != color.shape[:2]:
            raise RuntimeError(f"Aligned depth {depth_m.shape} does not match color {color.shape[:2]}.")
        return color, depth_m

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False
