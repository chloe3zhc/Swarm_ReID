import os
import glob
from torch.utils.data import Dataset
import torch
from PIL import Image
from .bases import read_image


class RealTimeImageDataset(Dataset):
    """
    实时图像数据集类，用于处理无标签的新采集图像
    """

    def __init__(self, image_folder, transform=None):
        """
        初始化实时图像数据集

        Args:
            image_folder (str): 包含待处理图像的文件夹路径
            transform (callable, optional): 图像变换函数
        """
        self.image_folder = image_folder
        self.transform = transform

        # 获取文件夹中所有图像文件
        self.image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        # 支持递归搜索子文件夹
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root, file))

        # 按文件名排序以保证一致性
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        获取指定索引的图像

        Args:
            index (int): 图像索引

        Returns:
            tuple: (image_tensor, image_path) 如果有变换函数
                   (image, image_path) 如果没有变换函数
        """
        img_path = self.image_paths[index]

        # 读取图像
        try:
            img = read_image(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个默认图像或跳过
            img = Image.new('RGB', (256, 128))  # 创建一个默认图像
            img_path = "error_image"

        # 应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, img_path


def collate_fn(batch):
    """
    自定义批次合并函数

    Args:
        batch: 批次数据

    Returns:
        tuple: 合并后的批次数据
    """
    # 正确处理批次数据，将图像堆叠成张量，保持路径为列表
    imgs, img_paths = zip(*batch)
    # 将图像列表堆叠成一个张量
    imgs_tensor = torch.stack(imgs, dim=0)
    return imgs_tensor, list(img_paths)


def make_real_time_dataloader(cfg, image_folder):
    """
    创建用于实时推理的数据加载器

    Args:
        cfg (CfgNode): 配置对象
        image_folder (str): 包含待处理图像的文件夹路径

    Returns:
        DataLoader: 实时推理数据加载器
    """
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    # 定义图像变换
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),  # 调整图像大小
        T.ToTensor(),  # 转换为Tensor
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)  # 标准化
    ])

    # 创建数据集
    dataset = RealTimeImageDataset(image_folder, val_transforms)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,  # 使用测试批次大小
        shuffle=False,  # 不打乱数据
        num_workers=cfg.DATALOADER.NUM_WORKERS,  # 工作进程数
        collate_fn=collate_fn
    )

    return dataloader