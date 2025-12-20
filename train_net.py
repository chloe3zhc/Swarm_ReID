from utils.logger import setup_logger  # 日志设置
from data import make_dataloader  # 数据加载器
from modeling import make_model  # 模型构建
from solver.make_optimizer import make_optimizer  # 优化器
from solver.scheduler_factory import create_scheduler  # 学习率调度器
from layers.make_loss import make_loss  # 损失函数
from engine.processor import do_train  # 训练引擎
import random
import torch
import numpy as np
import os
import argparse
from config import cfg  # 配置文件


# 设置随机种子以确保实验可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确定性卷积算法
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="FusionReID Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)  # 分布式训练参数
    args = parser.parse_args()

    # 加载配置文件
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)  # 从文件加载配置
    cfg.merge_from_list(args.opts)  # 从命令行参数加载配置
    cfg.TEST.FEAT = args.fea_cft  # 设置测试特征类型
    cfg.freeze()  # 冻结配置，防止后续修改

    # 设置随机种子
    set_seed(cfg.SOLVER.SEED)

    # 根据特征选择打印相应信息
    if cfg.TEST.FEAT == 0:
        print('All features used in test')
    elif cfg.TEST.FEAT == 1:
        print('Original_r used in test')
    elif cfg.TEST.FEAT == 2:
        print('Original_f used in test')
    elif cfg.TEST.FEAT == 3:
        print('LRU_r used in test')
    elif cfg.TEST.FEAT == 4:
        print('LRU_f used in test')
    elif cfg.TEST.FEAT == 5:
        print('HTM_r used in test')
    elif cfg.TEST.FEAT == 6:
        print('HTM_f used in test')

    # 分布式训练设置
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    # 创建输出目录
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置日志记录器
    logger = setup_logger("FusionReID", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    # 记录配置信息
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r', encoding='utf-8') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 初始化分布式训练
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 创建数据加载器
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print("data is ready")
    # 创建模型
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 创建损失函数
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # 创建优化器
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # 创建学习率调度器
    scheduler = create_scheduler(cfg, optimizer)

    # 开始训练
    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
