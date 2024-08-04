import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
# import YOLODataset
from ultralytics import YOLOv10
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build_dataloader
# from yolov10.ultralytics.models.yolo.model import YOLO
from compressai.models import TinyLIC
# # 实例化数据集
# with open('/home/fumchin/work/baseline/vcod/VOC.yaml', 'r') as f:
#     data_config = yaml.safe_load(f)
# # 获取根路径
# base_path = Path(data_config['path'])

# # 获取相对路径，并拼接为绝对路径
# train_img_paths = [base_path / p for p in data_config['train']]
# val_img_paths = [base_path / p for p in data_config['val']]

# # 实例化 YOLODataset
# train_dataset = YOLODataset(img_path=train_img_paths, data=data_config, task="detect")
# val_dataset = YOLODataset(img_path=val_img_paths, data=data_config, task="detect")

# # 创建 DataLoader
# train_loader = build_dataloader(train_dataset, batch=16, workers=4, shuffle=True, rank=-1)
# val_loader = build_dataloader(val_dataset, batch=16, workers=4, shuffle=False, rank=-1)

# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


yolo_model = YOLOv10('/home/fumchin/work/baseline/vcod/jameslahm/yolov10n.pt')
yolo_model = yolo_model.to(device)
tinylic_model = TinyLIC()
tinylic_model = tinylic_model.to(device)
optimizer = torch.optim.Adam(list(tinylic_model.parameters()) + list(yolo_model.parameters()))
# optimizer = torch.optim.Adam(list(yolo_model.parameters()))
# 示例损失计算函数
def compute_yolo_loss(outputs, labels):
    return yolo_model.compute_loss(outputs, labels)  # 使用模型的 compute_loss 方法

# optimizer = torch.optim.Adam(list(yolo_model.parameters()) + list(tinylic_model.parameters()), lr=0.001)


def train_merge():
    yolo_model.train(compress_model=tinylic_model, data='/home/fumchin/work/baseline/vcod/VOC.yaml', epochs=500, imgsz=640, batch=32, workers=0, pretrained=True)

def train_one_epoch(loader, yolo_model, tinylic_model, optimizer, device):
    # yolo_model.train()
    tinylic_model.train()
    for data in loader:
        # 解包数据
        input_images = data['img']
        labels = {
            'cls': data['cls'],
            'bboxes': data['bboxes']
        }
        
        input_images = input_images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        # # 将输入图像转换为 uint8 类型
        # input_images_byte = input_images.to(torch.uint8)

        # # TinyLIC 前向传播
        # compressed_images = tinylic_model(input_images_byte)['x_hat']  # TinyLIC 返回一个字典，需要取出 'x_hat'

        # YOLO v10 前向传播
        detections = yolo_model(input_images)

        # 计算 YOLO 的损失
        detection_loss = compute_yolo_loss(detections, labels)
        compressed_images = detection_loss
        # 计算 TinyLIC 的 MSE 损失
        mse_loss = torch.nn.functional.mse_loss(compressed_images, input_images)
        
        # 总损失
        total_loss = detection_loss + mse_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

# # 训练循环
# num_epochs = 100
# for epoch in range(num_epochs):
#     train_one_epoch(train_loader, yolo_model, None, optimizer, device)
#     # validate_one_epoch(val_loader, yolo_model, tinylic_model, device)  # 如果你有验证函数的话

# # 保存模型
# torch.save({
#     'yolo_model_state_dict': yolo_model.state_dict(),
#     # 'tinylic_model_state_dict': tinylic_model.state_dict()
# }, 'combined_model.pth')

train_merge()