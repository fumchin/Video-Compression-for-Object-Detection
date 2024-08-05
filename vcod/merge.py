import math
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import torch.nn as nn
import torch.optim as optim
# import YOLODataset
from ultralytics import YOLOv10
from compressai.models import TinyLIC

def configure_optimizers(net, lr=1e-4):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=lr,
    )
    return optimizer, aux_optimizer




class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out

def train_merge():
    # yolo_model.train(compress_model=tinylic_model, data='/home/fumchin/work/baseline/vcod/VOC.yaml', epochs=500, imgsz=640, batch=32, workers=0, pretrained=True)
    yolo_model.train_with_compression(compression_model=tinylic_model, compression_optimizer=compression_optimizer, data='/home/fumchin/work/baseline/vcod/VOC.yaml', epochs=500, imgsz=640, batch=2, workers=8, pretrained=True)

# def train_one_epoch(loader, yolo_model, tinylic_model, optimizer, device):
#     # yolo_model.train()
#     tinylic_model.train()
#     for data in loader:
#         # 解包数据
#         input_images = data['img']
#         labels = {
#             'cls': data['cls'],
#             'bboxes': data['bboxes']
#         }
        
#         input_images = input_images.to(device)
#         labels = {k: v.to(device) for k, v in labels.items()}

#         # # 将输入图像转换为 uint8 类型
#         # input_images_byte = input_images.to(torch.uint8)

#         # # TinyLIC 前向传播
#         # compressed_images = tinylic_model(input_images_byte)['x_hat']  # TinyLIC 返回一个字典，需要取出 'x_hat'

#         # YOLO v10 前向传播
#         detections = yolo_model(input_images)

#         # 计算 YOLO 的损失
#         detection_loss = compute_yolo_loss(detections, labels)
#         compressed_images = detection_loss
#         # 计算 TinyLIC 的 MSE 损失
#         mse_loss = torch.nn.functional.mse_loss(compressed_images, input_images)
        
#         # 总损失
#         total_loss = detection_loss + mse_loss

#         # 反向传播
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    yolo_model = YOLOv10('/home/fumchin/work/baseline/vcod/jameslahm/yolov10n.pt')
    yolo_model = yolo_model.to(device)
    tinylic_model = TinyLIC()
    tinylic_model = tinylic_model.to(device)
    compression_optimizer, aux_optimizer = configure_optimizers(tinylic_model)
    compression_criterion = RateDistortionLoss(lmbda=1e-2)
    # train_merge()
    
    yolo_model.train_with_compression(compression_model=tinylic_model, compression_optimizer=compression_optimizer, aux_optimizer=aux_optimizer, compression_criterion=compression_criterion, data='/home/fumchin/work/baseline/vcod/VOC.yaml', epochs=500, imgsz=320, batch=8, workers=4, pretrained=True)