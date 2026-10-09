# Gemini 2: prototype theo dõi vật thể 3D

Đây là tracker đơn vật thể tự triển khai, dùng RGB, depth đã căn chỉnh, optical flow và mô hình vận tốc không đổi. Người dùng chọn ROI bằng chuột trên frame đầu tiên. Không dùng YOLO hay tracker dựng sẵn của OpenCV.

## Cài đặt và chạy

Trong PowerShell, từ thư mục cha chứa `tracker_3d`:

```powershell
python -m pip install -r tracker_3d/requirements.txt
python -m tracker_3d.tracker_3d
python -m tracker_3d.tracker_3d --hw
```

`pyorbbecsdk` là package phụ thuộc vào hệ điều hành/Python và cần được cài theo hướng dẫn Orbbec cho thiết bị. Chương trình chọn profile màu và depth khả dụng gần 640×480, 30 FPS. Phím `R` chọn lại ROI, `M` đổi giữa ảnh màu và depth, `Q`/Esc thoát.

## Pipeline và phép đo 3D

Camera stream màu và depth được mở bằng `Pipeline`/`Config`. Mặc định, `AlignFilter(OBStreamType.COLOR_STREAM)` đưa depth sang lưới pixel màu bằng software D2C. `--hw` dùng depth profile D2C do `get_d2c_depth_profile_list(..., OBAlignMode.HW_MODE)` trả về và bật hardware alignment. Profile depth có scale SDK; mã đổi raw depth thành mét.

ROI ban đầu cung cấp vùng ảnh và các feature. `goodFeaturesToTrack` tìm feature trong vùng; `calcOpticalFlowPyrLK` ghép vị trí feature giữa hai frame. Độ dịch chuyển trung vị của các feature tốt cập nhật bounding box. Logic chọn điểm, loại flow yếu, duy trì box và đánh giá trạng thái do module này quản lý.

Depth estimator lấy median pixel hợp lệ trong 50% trung tâm của box. Giá trị 0, NaN/inf và ngoài 1–5 m bị bỏ qua; MAD loại bỏ các giá trị ngoại lai. Từ tâm box `(u,v)` và độ sâu `Z`, deprojection pinhole dùng intrinsics RGB:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
```

Tọa độ camera Orbbec dùng `+X` sang phải, `+Y` hướng xuống và `+Z` hướng ra trước. Tất cả vị trí là mét.

## Motion, trạng thái và giới hạn

Motion estimator giữ state `[X,Y,Z,Vx,Vy,Vz]`. Vận tốc lấy từ chênh lệch vị trí chia cho `dt`, rồi làm mượt EMA. Khi thiếu phép đo, nó dự đoán vị trí bằng `position += velocity * dt`; đây là mô hình vận tốc không đổi đơn giản, không phải Kalman filter.

Tracker đánh dấu confidence thấp khi flow/depth không tạo được phép đo, chuyển `LOST` sau 5 frame lỗi liên tiếp, và chấp nhận chọn ROI mới bằng `R`. Một bước nhảy 3D trên 1 m/frame cũng bị coi là measurement bất thường. Prototype phù hợp để học và thử nghiệm: optical flow có thể trôi khi vật thể ít texture, bị che khuất, hoặc ROI chứa nền; kiểm tra calibration/alignment và vùng đo thực tế trước khi dùng làm hệ thống an toàn hoặc điều khiển.
