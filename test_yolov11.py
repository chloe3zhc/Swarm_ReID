from ultralytics import YOLO
import cv2
import os

# Load an official or custom model
model = YOLO("./yolo_weights/yolo11n.pt")  # Load an official Detect model

# 创建保存文件夹
original_frames_dir = "original_frames"
annotated_frames_dir = "annotated_frames"
cropped_images_dir = "cropped_images"  # 用于保存裁剪的图像
os.makedirs(original_frames_dir, exist_ok=True)
os.makedirs(annotated_frames_dir, exist_ok=True)
os.makedirs(cropped_images_dir, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture("E:\ZHC\FusionReID-master\yolo_input\zhc_try.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 保存原始帧
    original_frame_path = os.path.join(original_frames_dir, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(original_frame_path, frame)

    # 运行跟踪推理
    results = model.track(frame, show=False, classes=[0], persist=True)  # 不显示结果，但保留跟踪ID

    # 获取带标注的帧
    annotated_frame = results[0].plot()  # 获取绘制了检测结果的帧

    # 保存带标注的帧
    annotated_frame_path = os.path.join(annotated_frames_dir, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(annotated_frame_path, annotated_frame)

    # 裁剪检测到的边界框区域并保存
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # 裁剪图像
            cropped_img = frame[y1:y2, x1:x2]
            
            # 生成唯一文件名，包含帧号和边界框ID
            if box.id is not None:
                track_id = int(box.id.cpu().numpy())
                cropped_img_path = os.path.join(cropped_images_dir, f"frame_{frame_count:06d}_id_{track_id}.jpg")
            else:
                cropped_img_path = os.path.join(cropped_images_dir, f"frame_{frame_count:06d}_det_{i}.jpg")
            
            # 保存裁剪的图像
            cv2.imwrite(cropped_img_path, cropped_img)

    frame_count += 1

    # 显示带标注的帧（可选）
    cv2.imshow('Annotated Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break

cap.release()
cv2.destroyAllWindows()

# results = model.track("E:\ZHC\FusionReID-master\yolo_input\ccc.mp4", show=True, classes=[0])  # Tracking with default tracker
# results = model.track("E:\ZHC\yolov11-main\ccc.mp4", show=True, tracker="bytetrack.yaml", classes=[0])  # with ByteTrack
