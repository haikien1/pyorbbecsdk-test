# Orbbec Gemini 2 --- Two-Week Learning Progress

## Overview

This repository documents my two-week learning progress with the Orbbec
Gemini 2 RGB-D camera and the `pyorbbecsdk` library, alongside initial
computer vision experiments in object detection and tracking.

The main goal was to understand RGB-D data acquisition and processing,
explore the fundamentals of camera geometry and 3D representation, and
compare different approaches to tracking objects in video.

## Week 1: Camera Setup and RGB-D Data Processing

### 1. Hello Camera

-   Explored the basic workflow for connecting to the Orbbec Gemini 2
    camera.
-   Learned about camera pipelines, stream profiles, and frame
    acquisition.
-   Studied how to configure camera streams and retrieve frames through
    the SDK.

**Outcome:** Built foundational understanding of how to acquire data
from the camera for computer vision tasks.

### 2. Depth Visualization

-   Explored depth frame acquisition and processing.
-   Learned how depth data represents distance information.
-   Studied depth visualization using OpenCV.
-   Investigated valid depth measurements and their role in distance
    estimation.

**Outcome:** Developed an understanding of how depth information can
help estimate the distance between the camera and an observed object.

### 3. Color and Depth Alignment

-   Explored the acquisition of color and depth frames.
-   Studied Depth-to-Color (D2C) alignment.
-   Learned why color-depth alignment matters when associating object
    detections with depth measurements.
-   Investigated the relationship between 2D image coordinates and
    corresponding depth data.

**Outcome:** Understood why color and depth data need to be spatially
aligned when combining 2D detections with depth measurements.

## Week 2: Camera Geometry, 3D Representation, and Tracking

### 4. Camera Calibration

-   Studied the purpose of camera calibration in computer vision.
-   Explored camera intrinsic parameters, including focal lengths and
    the principal point.
-   Learned how camera geometry supports the conversion from image
    coordinates to 3D camera coordinates.
-   Investigated the importance of camera parameters for 3D position
    estimation.

**Outcome:** Developed foundational understanding of how camera
parameters relate 2D image coordinates to 3D positions.

### 5. Point Cloud
-   Studied point clouds as representations of 3D scenes.
-   Explored the relationship between depth images and 3D spatial data.
-   Learned how RGB-D information can represent spatial locations.
-   Investigated the potential use of point clouds for understanding
    scene geometry.

**Outcome:** Gained foundational knowledge of 3D spatial representation
using depth-camera data.

### 6. YOLO-Based Object Detection and Tracking

**Approach:** Automatic detection followed by tracking.

-   Explored using YOLO to detect objects in video frames.
-   Used bounding boxes and class labels to represent detected objects.
-   Worked with a tracking workflow to associate detections across
    consecutive frames and maintain tracking IDs.
-   Studied the distinction between object detection and object
    tracking.

**Outcome:** Learned how automatic detection can be combined with a
tracking algorithm to follow objects over time.

### 7. User-Selected Bounding Box with a Custom Tracker

**Approach:** Manually select a target and track it using a
self-implemented tracking algorithm.

-   Implemented a workflow in which the user selects the target object
    by specifying a bounding box in the video.
-   Developed a custom tracker to follow the selected target across
    subsequent frames.
-   Explored how the target's location can be updated over time based on
    visual information.
-   Considered tracking challenges such as object movement, appearance
    changes, occlusion, and loss of visual features.

**Outcome:** Gained practical experience with ROI/bounding-box
initialization and implementing a tracking method independently, rather
than relying on an OpenCV built-in tracker.

## Comparison of the Tracking Approaches

  -----------------------------------------------------------------------
  Aspect                  YOLO-Based Tracking     User-Selected Custom
                                                  Tracking
  ----------------------- ----------------------- -----------------------
  Target initialization   Automatic detection by  User selects a bounding
                          YOLO                    box

  Main purpose            Track objects detected  Follow a specific
                          by the model            user-selected target

  Core components         YOLO detection plus a   Custom-implemented
                          tracking algorithm      tracking logic

  Main advantage          Can detect supported    Allows the user to
                          object classes and      select a target
                          track detections        directly

  Main challenge          Depends on detection    May lose the target
                          quality and data        under occlusion,
                          association             appearance changes, or
                                                  weak visual features
  -----------------------------------------------------------------------

The two approaches serve different purposes: YOLO-based tracking begins
with automatic object detection, while the custom tracking experiment
begins with a manually selected target. Their behavior depends on the
details of each implementation and the test conditions.

## Relevance to 3D Object Tracking

The concepts studied in the SDK examples and tracking experiments
provide a foundation for an RGB-D-based 3D tracking pipeline:

1.  Acquire color and depth frames from the Orbbec Gemini 2.
2.  Align depth data with the color image.
3.  Detect objects using YOLO or initialize a target using a
    user-selected bounding box.
4.  Associate detections or update the selected target across frames.
5.  Estimate representative depth for the target.
6.  Use camera intrinsics and depth to convert image coordinates into 3D
    camera coordinates.
7.  Apply state estimation, such as a 3D Kalman filter, if required.
8.  Visualize tracking results and evaluate performance.

The SDK examples and the two tracking experiments are learning and
development steps toward this pipeline; they do not by themselves
establish that the complete pipeline has been fully validated.

## Key Learning Outcomes

-   Camera initialization and frame acquisition using `pyorbbecsdk`.
-   Depth data processing and visualization.
-   Color-depth alignment for RGB-D perception.
-   Fundamentals of camera calibration and intrinsic parameters.
-   Point cloud concepts and 3D spatial representation.
-   YOLO-based object detection and tracking workflows.
-   User-selected ROI/bounding-box initialization.
-   Implementing a custom visual tracker rather than relying on a
    built-in OpenCV tracker.
-   Understanding how 2D tracking and depth measurements can contribute
    to 3D position estimation.

## Next Steps

-   Test depth measurements at different distances and assess their
    reliability.
-   Validate the conversion from image coordinates and depth values to
    3D camera coordinates.
-   Evaluate the stability of YOLO-based tracking and the custom tracker
    under different conditions.
-   Measure runtime performance, including frame rate and processing
    latency.
-   Explore how 3D tracking outputs could support future robot
    perception and control tasks.

## Summary

During these two weeks, I studied the fundamentals of the Orbbec Gemini
2 RGB-D camera and the `pyorbbecsdk` library through five example
programs. I also explored two object-tracking approaches: YOLO-based
automatic detection and tracking, and a custom tracker initialized by a
user-selected bounding box.

These activities helped establish a foundation in RGB-D data processing,
camera geometry, 3D representation, and visual tracking. Further
implementation, testing, and performance evaluation remain ongoing
tasks.
