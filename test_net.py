import os
from config import cfg
import argparse
from data import make_dataloader
from data.datasets.real_time_dataset import make_real_time_dataloader
from modeling import make_model
# from engine.processor import do_inference
from inference_processor import do_inference, do_real_time_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionReID Testing")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int)
    parser.add_argument("--real_time_folder", default="", help="Path to folder containing real-time images", type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.FEAT = args.fea_cft
    cfg.freeze()
    if cfg.TEST.FEAT == 0:
        print('All features used in test')

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("FusionReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 检查是否提供了实时图像文件夹
    if args.real_time_folder and os.path.exists(args.real_time_folder):
        # 使用实时推理数据加载器
        image_loader = make_real_time_dataloader(cfg, args.real_time_folder)
        num_query = len(image_loader.dataset)
        num_classes = 0  # 实时推理中不需要类别数
        camera_num = 0   # 实时推理中不需要摄像头数
        view_num = 0     # 实时推理中不需要视角数
    else:
        # 使用默认的测试数据加载器
        image_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=15, view_num=view_num)
    model.load_param_ignore_classifier("E:\ZHC\FusionReID\FusionReID_180.pth")
    
    # 调用实时推理函数
    do_real_time_inference(cfg, model, image_loader)