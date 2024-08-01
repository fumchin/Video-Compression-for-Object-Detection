import torch
import sys
from compressai.models import TinyLIC
from yolov10.ultralytics.models.yolo import YOLO


tinylic_model = TinyLIC()
yolo_model = YOLO()

optimizer = torch.optim.Adam(list(tinylic_model.parameters()) + list(yolo_model.parameters()))
num_epochs = 200
for epoch in range(num_epochs):
    for data in dataloader:
        # 获取输入图像和标签
        input_images, labels = data

        # TinyLIC 前向传播
        compressed_images = tinylic_model(input_images)

        # YOLO v10 前向传播
        detections = yolo_model(compressed_images)

        # 计算 TinyLIC 的 MSE 损失
        mse_loss = torch.nn.functional.mse_loss(compressed_images, input_images)

        # 计算 YOLO v10 的目标检测损失
        detection_loss = yolo_model.compute_loss(detections, labels)

        # 计算联合损失
        total_loss = mse_loss + detection_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}")

print("Training complete.")