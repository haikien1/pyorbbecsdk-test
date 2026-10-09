import sys

import cv2
import numpy as np
import argparse
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import frame_to_bgr_image, is_dabai_a_series_device

from pyorbbecsdk import OBAlignMode
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
    unDistortionFilter,
)

# Constants

ESC_KEY = 27
MIN_DEPTH = 20
MAX_DEPTH = 10000

# Hardware d2c helpers (used only with --hw)

def get_hw_stream_config(pipeline: Pipeline):
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        for i in range(len(profile_list)):
            color_profile = profile_list[i]
            if color_profile.get_format() != OBFormat.RGB:
                continue
            hw_depth_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
            if len(hw_depth_list) == 0:
                continue
            config.enable_stream(hw_depth_list[0])
            config.enable_stream(color_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)
            return config
    except Exception as e:
        print(f"HW D2C config error: {e}")
    return None

def swtich_hw_d2c(pipeline: Pipeline, config: Config, enable: bool):
    # Stop the pipeline, flip the HW D2C align mode, and restart.
    pipeline.stop()
    time.sleep(0.1)
    config.set_align_mode(OBAlignMode.HW_MODE if enable else OBAlignMode.DISABLE)
    print(f"Hardware D2C: {'Enable' if enable else 'Disable'}")
    pipeline.start()

# Main

def main():
    # Check if device is connected
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("No device found.")
        return

    parser = argparse.ArgumentParser(description="Color + Depth aligned viewer")
    parser.add_argument(
        "--hw",
        action="store_true",
        help="Use hardware D2C alignment instead of software AlignFilter"
    )
    args = parser.parse_args()

    window_name = "Color + Depth Aligned | Q/ESC = quit"
    