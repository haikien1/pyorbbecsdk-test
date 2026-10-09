import cv2
import numpy as np
from utils import frame_to_bgr_image, resize_to_fit

from pyorbbecsdk import OBError, OBFormat, Pipeline #type: ignore

#Constants
ESC_KEY = 27
MIN_DEPTH = 20
MAX_DEPTH = 5000

WINDOW_NAME = "QuickStart Viewer"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

def render_depth_3d(depth_mn: np.ndarray) -> np.ndarray:
    """
    Convert a float32 depth-in-mm array into a 3D-looking BGR image.

    Pipeline:
      clip [MIN, MAX] → gamma 0.8 → uint8 → Scharr gradient lighting → colormap
    """

    # 1. Clip to fixed range (keeps colors stable acroos frames)
    depth_clipped = np.clip(depth_mn, MIN_DEPTH, MAX_DEPTH)

    # 2. Normalize to [0, 1] and apply gamma correction
    # y < 1 stretches near-field gradients for better detail

    depth_norm = (depth_clipped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
    depth_gamma = np.power(depth_norm, 0.8)

    # 3. Convert to uint8
    
    depth_8bit = np.uint8(depth_gamma * 255).astype(np.uint8)

    # 4. Compute Scharr gradient for lighting effect
    #   Simulates a directional light from top-left for 3D relief

    grad_x = cv2.Scharr(depth_8bit, cv2.CV_32F,1 , 0)
    grad_y = cv2.Scharr(depth_8bit, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(grad_x, grad_y) + 1.0

    lighting = -0.707 * (grad_x + grad_y) / mag
    lighting = lighting * 0.15 + 0.85
    np.clip(lighting, 0.7, 1.0, out=lighting)

    # 5. Apply colormap then multiply by lighting to get final BGR image
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    depth_colored = (depth_colored * lighting[..., np.newaxis]).astype(np.uint8)

    return depth_colored

def main():
    #Step 1: Create a pipeline and start with the default configuration

    try:
        pipeline = Pipeline()
        pipeline.start()
    except OBError as e:
        print(f"Error: {e}")
        print("Make sure the Orbbec Astra camera is connected and try again.")
        return

    print("Pipeline started (default configuration). Press ESC or 'Q' to exit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        try:
            #Step 2: Wait for a synchronized frame set (depth + color)

            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            #Step 3: Get the Color frame and convert to BGR image

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)

            #Step 4: Get the Depth frame and apple scale and range filter

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_mn = depth_data.astype(np.float32) * scale

            #Step 5: Render the depth frame into a 3D-looking BGR image
            
            depth_image = render_depth_3d(depth_mn)

            half_w = WINDOW_WIDTH // 2
            color_resized = resize_to_fit(color_image, half_w, WINDOW_HEIGHT)
            depth_resized = resize_to_fit(depth_image, half_w, WINDOW_HEIGHT)
            combined = np.hstack((color_resized, depth_resized))

            cv2.imshow(WINDOW_NAME, combined)

            if cv2.waitKey(1) in (ord("q"), ord("Q"), ESC_KEY):
                break

        except KeyboardInterrupt:
            break

    #Step 6: Clean up

    cv2.destroyAllWindows()
    pipeline.stop()
    print("Pipeline stopped. Exiting.")

if __name__ == "__main__":
    main()
