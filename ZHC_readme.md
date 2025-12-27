# 项目使用说明文档

### 文件说明

1. FusionReID模型配置文件在`configs`目录下，针对不同数据集有不用的配置文件。采用的配置文件路径如下：
- `DukeMTMC`数据集`configs/DukeMTMC/duke_vitb12_res50_layer2.yml`
- `Market1501`数据集`configs/Market1501/market_vitb12_res50_layer2.yml`
- `MSMT17`数据集`configs/MSMT17/msmt_vitb12_res50_layer2.yml`

2. 新增Yolo11模型
- `ultralytics`文件来自于Yolo11官方项目中
- `yolo_input`文件为输入Yolo11模型的图像和视频
- `yolo_weights`文件为Yolo11模型的权重文件

### 修改记录

1. `data/datasets/make_dataloader.py`190行后面的注释为原始代码，修改后为适配于只使用测试集，数据集中只有测试集
```
# 适配于 只使用测试集，数据集中只有测试集
def make_dataloader(cfg):
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),  # 调整图像大小（通常使用固定大小）
        T.ToTensor(),  # 转换为Tensor
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)  # 标准化
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,  # 测试批次大小
        shuffle=False,  # 不打乱数据
        num_workers=num_workers,  # 数据加载工作进程数
        collate_fn=val_collate_fn  # 自定义批次合并函数（验证版本）
    )

    # 由于只有测试数据，返回简化版的结果
    # 如果数据集中有num_query_pids属性，则使用它；否则使用默认值0
    num_query_pids = getattr(dataset, 'num_query_pids', 0)
    num_classes = num_query_pids  # 对于测试模式，类别数等于查询集ID数
    cam_num = getattr(dataset, 'num_query_cams', 0)  # 查询集摄像头数
    view_num = getattr(dataset, 'num_query_vids', 0)  # 查询集视图数

    return val_loader, len(dataset.query), num_classes, cam_num, view_num
```
2. `data/datasets/msmt17.py`


3. 实时推理功能：新增了实时推理功能，可以在无标签的情况下对新采集的图像进行人员重识别。主要特性包括：
   - 自动为新图像分配ID
   - 基于特征相似度判断是否为同一人
   - 支持配置相似度阈值（TEST.REID_THRESHOLD）
   - 实时保存特征向量供后续使用
   
### 指令：
```bash
   python test_net.py --config_file configs/MSMT17/msmt_vitb12_res50_layer2.yml --real_time_folder path/to/your/images
   ```


### 20251227更新
`test_yolo_reid_0.py`
该文件是一个结合了Yolo和ReID的视频处理脚本，主要功能：
1. 使用Yolov11检测视频中的人并进行跟踪
2. 对检测到的人进行重识别，判断是否为同一人
3. 保存原始视频帧，并展示带有重识别ID的视频帧

