from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from config import cfg
from torchvision import transforms as T
from modeling import make_model
import time
import psutil

# Load YOLO model
model = YOLO("./yolo_weights/yolo11n.pt")  # Load an official Detect model

# Load ReID model
cfg.merge_from_file("configs/MSMT17/msmt_vitb12_res50_layer2.yml")  # Load your config file
reid_model = make_model(cfg, num_class=1, camera_num=15, view_num=0)  # Initialize model with dummy values
reid_model.load_param_ignore_classifier("E:\ZHC\FusionReID\FusionReID_180.pth")  # Load trained weights
reid_model.eval()  # Set to evaluation mode

# 确定设备并移动模型到相应设备
device = "cuda" if torch.cuda.is_available() else "cpu"
reid_model.to(device)  # Move to GPU if available

# 打开视频文件并获取输入帧率
cap = cv2.VideoCapture("E:\ZHC\FusionReID-master\yolo_input\zhc1_10.mp4")
input_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Input video FPS: {input_fps}")
print(f"Total frames: {total_frames}")

# 创建保存文件夹
original_frames_dir = "original_frames"
annotated_frames_dir = "annotated_frames"
os.makedirs(original_frames_dir, exist_ok=True)
os.makedirs(annotated_frames_dir, exist_ok=True)

# 存储已知特征向量和ID
known_features = []  # 已知特征向量列表
known_ids = []  # 对应的ID列表
next_id = 0  # 下一个可用的ID
threshold = 0.75  # 特征相似度阈值，可从配置中获取

def calculate_features_memory(features_list):
    """计算特征向量列表占用的内存大小（KB）"""
    total_memory = 0
    for feature in features_list:
        if isinstance(feature, np.ndarray):
            # 计算NumPy数组占用的内存
            total_memory += feature.nbytes
        else:
            # 对于其他类型，尝试获取其大小
            total_memory += len(feature) * 8  # 假设每个元素是float64，占8字节
    return total_memory / 1024  # 转换为KB

def extract_reid_features_batch(images, reid_model, cfg, device):
    """批量提取重识别特征，使用与项目一致的处理方式"""

    # 记录批量处理时间
    batch_start_time = time.time()
    # 预处理图像
    val_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(cfg.INPUT.SIZE_TEST),  # 调整图像大小
        T.ToTensor(),  # 转换为Tensor
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)  # 标准化
    ])

    # 批量处理图像
    batch_tensors = []
    for img in images:
        img_tensor = val_transforms(img)
        batch_tensors.append(img_tensor)

    # 堆叠成批次张量
    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        # 使用与inference_processor.py和test_net.py一致的特征提取方式
        batch_size = batch_tensor.size(0)
        cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)
        feats = reid_model(batch_tensor, cam_label=cam_label, view_label=None)
        feats_np = feats.cpu().numpy()

    # 返回特征列表
    batch_end_time = time.time()
    batch_process_time = batch_end_time - batch_start_time
    print(f"Batch processing time for {len(images)} images: {batch_process_time:.4f} seconds")
    print(f"Average time per image in batch: {batch_process_time / len(images):.4f} seconds")

    # 返回特征列表
    return [feat.flatten() for feat in feats_np]


def assign_id_to_feature(feature, known_features, known_ids, threshold=0.9):
    """
    根据特征向量相似度为新特征分配ID，使用与inference_processor.py一致的逻辑
    """
    if len(known_features) == 0:
        return None

    # 计算与所有已知特征的余弦相似度，与inference_processor.py保持一致
    similarities = []
    for known_feat in known_features:
        # 计算余弦相似度
        dot_product = np.dot(feature, known_feat)
        norm_product = np.linalg.norm(feature) * np.linalg.norm(known_feat)
        if norm_product == 0:
            similarity = 0
        else:
            similarity = dot_product / norm_product
        similarities.append(similarity)

    # 找到最大相似度
    max_similarity = max(similarities)
    max_index = similarities.index(max_similarity)

    # 如果最大相似度超过阈值，则认为是同一个ID
    if max_similarity >= threshold:
        return known_ids[max_index]
    else:
        return None


# 缓冲区存储待处理的图像信息
image_buffer = []  # 存储待处理的图像信息

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 保存原始帧
    original_frame_path = os.path.join(original_frames_dir, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(original_frame_path, frame)

    # 运行跟踪推理
    results = model.track(frame, show=False, classes=[0], persist=True, verbose=False)  # 不显示结果，但保留跟踪ID

    # 为每个检测框分配重识别ID
    reid_results = []
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
            # print(f"Cropped image shape before preprocessing: {cropped_img.shape}")

            # 将裁剪图像信息添加到缓冲区
            image_buffer.append({
                'cropped_img': cropped_img,
                'bbox': (x1, y1, x2, y2),
                'confidence': box.conf[0].cpu().numpy(),
                'frame_num': frame_count
            })

            # 当缓冲区中有更多图像时，执行重识别
            if len(image_buffer) > 0:
                # 提取缓冲区中所有图像的特征
                images = [img_info['cropped_img'] for img_info in image_buffer]
                features = extract_reid_features_batch(images, reid_model, cfg, device)

                # 为每张图像分配ID并处理
                for i, img_info in enumerate(image_buffer):
                    reid_feature = features[i]
                    bbox = img_info['bbox']
                    confidence = img_info['confidence']

                    # 分配重识别ID，使用项目原有的匹配方法
                    reid_id = assign_id_to_feature(reid_feature, known_features, known_ids, threshold)

                    # 如果是新ID，分配新ID
                    if reid_id is None:
                        reid_id = next_id
                        next_id += 1
                        # 添加到已知特征列表
                        known_features.append(reid_feature)
                        known_ids.append(reid_id)

                    # 保存重识别结果
                    reid_results.append({
                        'bbox': bbox,
                        'reid_id': reid_id,
                        'confidence': confidence
                    })

                # 清空缓冲区
                image_buffer = []

    # 处理剩余在缓冲区中的图像（视频结束时）
    if not ret and len(image_buffer) > 0:
        # 提取缓冲区中所有图像的特征
        images = [img_info['cropped_img'] for img_info in image_buffer]
        features = extract_reid_features_batch(images, reid_model, cfg, device)

        # 为每张图像分配ID并处理
        for i, img_info in enumerate(image_buffer):
            reid_feature = features[i]
            bbox = img_info['bbox']
            confidence = img_info['confidence']

            # 分配重识别ID，使用项目原有的匹配方法
            reid_id = assign_id_to_feature(reid_feature, known_features, known_ids, threshold)

            # 如果是新ID，分配新ID
            if reid_id is None:
                reid_id = next_id
                next_id += 1
                # 添加到已知特征列表
                known_features.append(reid_feature)
                known_ids.append(reid_id)

            # 保存重识别结果
            reid_results.append({
                'bbox': bbox,
                'reid_id': reid_id,
                'confidence': confidence
            })

        # 清空缓冲区
        image_buffer = []

    # 在帧上绘制重识别ID而不是跟踪ID
    annotated_frame = frame.copy()
    for result in reid_results:
        x1, y1, x2, y2 = result['bbox']
        reid_id = result['reid_id']
        confidence = result['confidence']

        # 绘制边界框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 在边界框上方显示重识别ID
        label = f"ReID: {reid_id}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保存带标注的帧
    annotated_frame_path = os.path.join(annotated_frames_dir, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(annotated_frame_path, annotated_frame)

    frame_count += 1

    # 每50帧打印一次内存占用情况
    if frame_count % 5 == 0:
        # 计算特征向量占用的内存大小
        features_memory = calculate_features_memory(known_features)
        print(f"Frame {frame_count}: Features memory usage: {features_memory:.2f} KB")
        features_count = len(known_features)
        print(f"Frame {frame_count}: Number of feature vectors: {features_count}")

    # 显示带标注的帧（可选）
    cv2.imshow('Annotated Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break
'''
    # 计算并显示输出帧率
    if frame_count % 50 == 0:  # 每50帧计算一次
        elapsed_time = time.time() - start_time
        output_fps = frame_count / elapsed_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds. Output FPS: {output_fps:.2f}")
'''

# 计算最终的输出帧率
end_time = time.time()
total_processing_time = end_time - start_time
output_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

print(f"Input video FPS: {input_fps}")
print(f"Output video FPS: {output_fps:.2f}")
print(f"Total processed frames: {frame_count}")
print(f"Total processing time: {total_processing_time:.2f} seconds")

cap.release()
cv2.destroyAllWindows()
