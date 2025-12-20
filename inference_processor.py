################################################################################
# 基于项目文件 “E:\ZHC\FusionReID-master\engine\processor.py” 中的 do_inference() 函数修改
# 函数功能：测试模型，实现模型提取的特征的本地保存
#################################################################################

import logging
import os
import time
import torch
import torch.nn as nn
# from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
# from torch.cuda import amp
# import torch.distributed as dist
import numpy as np

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("FusionReID.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    # 用于保存特征向量
    features_list = []
    pid_list = []
    camid_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

            # 保存特征向量用于后续保存到文件
            features_list.append(feat.cpu().numpy())
            # 修改：检查类型并正确处理
            if isinstance(pid, torch.Tensor):
                pid_list.append(pid.numpy())
            else:
                pid_list.append(pid)

            if isinstance(camid, torch.Tensor):
                camid_list.append(camid.numpy())
            else:
                camid_list.append(camid)

    try:
        cmc, mAP, all_AP, all_CMC, indices, good_index, junk_index = evaluator.compute()
        print("YES")

        # 打印查询ID匹配到的ID以及正确ID
        logger.info("Query Matching Results:")
        logger.info("Format: Query_ID -> Matched_IDs (Correct_ID)")

        # 获取查询集和画廊集的标签
        labels = evaluator.labels if hasattr(evaluator, 'labels') else None
        if labels is not None:
            query_labels = labels[:num_query]
            gallery_labels = labels[num_query:]

            # 打印前几个查询结果作为示例
            num_samples_to_print = min(10, len(query_labels))
            for i in range(num_samples_to_print):
                query_id = query_labels[i]
                # 获取匹配的索引（这里打印top-5匹配结果）
                if i < len(indices):
                    matched_indices = indices[i][:5]  # Top-5 matches
                    matched_ids = [gallery_labels[idx] for idx in matched_indices]
                    print(f"Query ID: {query_id} -> Matched IDs: {matched_ids} (Correct ID: {query_id})")

        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))

        for r in [1, 5, 10]:
            if r <= len(cmc):
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            else:
                logger.info("CMC curve, Rank-{:<3}: N/A (insufficient data)".format(r))

    except AssertionError as e:
        if "all query identities do not appear in gallery" in str(e):
            print("NO")
            logger.error("Error: All query identities do not appear in gallery")
        else:
            raise e
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e

    # 保存特征向量到本地txt文件
    if cfg.TEST.SAVE_FEATURES:
        save_features_to_txt(features_list, pid_list, camid_list, img_path_list, cfg.OUTPUT_DIR)

    return cmc[0] if 'cmc' in locals() and len(cmc) > 0 else 0


def save_features_to_txt(features_list, pid_list, camid_list, img_path_list, output_dir):
    """
    将特征向量保存到txt文件中

    Args:
        features_list: 特征向量列表
        pid_list: 行人ID列表
        camid_list: 摄像头ID列表
        img_path_list: 图像路径列表
        output_dir: 输出目录
    """
    import os
    import numpy as np

    # 合并所有批次的数据
    all_features = np.vstack(features_list)
    all_pids = np.hstack(pid_list)
    all_camids = np.hstack(camid_list)

    # 创建特征保存目录
    features_dir = os.path.join(output_dir, 'extracted_features')
    os.makedirs(features_dir, exist_ok=True)

    # 保存特征向量到txt文件
    features_file = os.path.join(features_dir, 'features.txt')
    with open(features_file, 'w') as f:
        # 写入头部信息
        f.write(f"# Total samples: {len(all_features)}\n")
        f.write(f"# Feature dimension: {all_features.shape[1]}\n")
        f.write("# PID\tCAMID\tFEATURE_VECTOR\tIMAGE_PATH\n")

        # 写入每个样本的特征
        for i in range(len(all_features)):
            pid = all_pids[i]
            camid = all_camids[i]
            feature_vec = ' '.join([f'{x:.6f}' for x in all_features[i]])
            img_path = img_path_list[i] if i < len(img_path_list) else 'N/A'
            f.write(f"{pid}\t{camid}\t{feature_vec}\t{img_path}\n")

    print(f"Features saved to {features_file}")

    # 可选：分别保存PID、CAMID和图像路径到单独的文件
    # np.savetxt(os.path.join(features_dir, 'pids.txt'), all_pids, fmt='%d')
    # np.savetxt(os.path.join(features_dir, 'camids.txt'), all_camids, fmt='%d')

    print(f"Additional info saved to {features_dir}")


def do_real_time_inference(cfg, model, image_loader):
    """
    实时推理函数，用于处理无标签的新采集数据

    Args:
        cfg: 配置对象
        model: 训练好的模型
        image_loader: 新采集图片的数据加载器

    Returns:
        assigned_ids: 每张图片分配的ID列表
    """
    device = "cuda"
    logger = logging.getLogger("FusionReID.real_time")
    logger.info("Enter real-time inferencing")

    # 初始化模型
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    # 存储已分配的特征向量和ID
    known_features = []  # 已知特征向量列表
    known_ids = []  # 对应的ID列表
    next_id = 0  # 下一个可用的ID

    assigned_ids = []  # 为每张图片分配的ID

    # 特征保存目录
    features_dir = os.path.join(cfg.OUTPUT_DIR, 'real_time_features')
    os.makedirs(features_dir, exist_ok=True)
    features_file = os.path.join(features_dir, 'features.txt')

    # 如果存在历史特征文件，则加载
    if os.path.exists(features_file):
        known_features, known_ids, next_id = load_known_features(features_file)
        logger.info(f"Loaded {len(known_features)} known features, next ID: {next_id}")

    # 遍历数据加载器中的批次
    for n_iter, (imgs, img_paths) in enumerate(image_loader):
        with torch.no_grad():
            # 将图像移到设备上
            imgs = imgs.to(device)

            # 提取特征
            feats = model(imgs)  # 提取特征
            # logger.info(f"Feats already exists,Batch {n_iter}: Extracted features shape: {feats.shape}")

            # 转换为numpy数组
            feats_np = feats.cpu().numpy()

            # 为批次中的每张图像分配ID
            for i in range(len(imgs)):
                feat_np = feats_np[i].flatten()
                img_path = img_paths[i] if isinstance(img_paths, list) else img_paths

                # 判断是否为新ID
                assigned_id = assign_id_to_feature(feat_np, known_features, known_ids, cfg.TEST.REID_THRESHOLD)

                # 如果是新ID，分配新ID
                if assigned_id is None:
                    assigned_id = next_id
                    next_id += 1
                    # 添加到已知特征列表
                    known_features.append(feat_np)
                    known_ids.append(assigned_id)
                    logger.info(f"New ID {assigned_id} assigned to image {img_path}")
                # else:
                    # logger.info(f"Existing ID {assigned_id} assigned to image {img_path}")

                assigned_ids.append(assigned_id)

                # 保存特征到文件（注意这里要放在循环内部）
                save_real_time_feature(feat_np, assigned_id, img_path, features_file)

    logger.info("Real-time inferencing completed")
    return assigned_ids


def load_known_features(features_file):
    """
    从文件加载已知特征向量

    Args:
        features_file: 特征文件路径

    Returns:
        known_features: 已知特征向量列表
        known_ids: 对应的ID列表
        next_id: 下一个可用的ID
    """
    known_features = []
    known_ids = []
    next_id = 0

    with open(features_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pid = int(parts[0])
                feature_str = parts[2]
                feature = np.array([float(x) for x in feature_str.split()])

                known_features.append(feature)
                known_ids.append(pid)

                if pid >= next_id:
                    next_id = pid + 1

    return known_features, known_ids, next_id


def assign_id_to_feature(feature, known_features, known_ids, threshold=0.9):
    """
    根据特征向量相似度为新特征分配ID

    Args:
        feature: 新特征向量
        known_features: 已知特征向量列表
        known_ids: 对应的ID列表
        threshold: 相似度阈值

    Returns:
        assigned_id: 分配的ID，如果为新ID则返回None
    """
    start_time = time.time()

    if len(known_features) == 0:
        return None

    # 计算与所有已知特征的余弦相似度
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

    # 打印相似度信息用于调试
    elapsed_time = time.time() - start_time
    # logging.getLogger("FusionReID.real_time").info(f"Elapsed time: {elapsed_time:.4f} seconds")
    logging.getLogger("FusionReID.real_time").info(f"Feature similarities: {similarities}")
    logging.getLogger("FusionReID.real_time").info(f"Max similarity: {max_similarity}, Threshold: {threshold}")

    # 如果最大相似度超过阈值，则认为是同一个ID
    if max_similarity >= threshold:
        return known_ids[max_index]
    else:
        return None


def save_real_time_feature(feature, pid, img_path, features_file):
    """
    保存实时特征到文件

    Args:
        feature: 特征向量
        pid: 分配的ID
        img_path: 图片路径
        features_file: 特征文件路径
    """
    # 将特征向量转换为字符串
    feature_str = ' '.join([f'{x:.6f}' for x in feature])

    # 构造行数据
    line = f"{pid}\t0\t{feature_str}\t{img_path}\n"  # CAMID设为0

    # 追加写入文件
    with open(features_file, 'a') as f:
        f.write(line)
