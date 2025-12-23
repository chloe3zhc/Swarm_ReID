'''
from ultralytics import YOLO

# 加载预训练的 YOLOv11n 模型
model = YOLO('yolo11x.pt')
source = 'person.jpg' #更改为自己的图片路径
# 运行推理，并附加参数
model.predict(source, save=True)
'''

from ultralytics import YOLO

# Load an official or custom model
model = YOLO("./yolo_input/yolo11n.pt")  # Load an official Detect model

# Perform tracking with the model
results = model.track("E:\ZHC\yolov11-main\ccc.mp4", show=True, classes=[0])  # Tracking with default tracker
# results = model.track("E:\ZHC\yolov11-main\ccc.mp4", show=True, tracker="bytetrack.yaml", classes=[0])  # with ByteTrack
