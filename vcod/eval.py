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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    yolo_model = YOLOv10('/home/englishassignment123/work/baseline/vcod/checkpoints_q3/checkpoint_best_loss_yolo.pth.tar')
    yolo_model = yolo_model.to(device)
    
    checkpoint = torch.load('/home/englishassignment123/work/baseline/vcod/checkpoints_q3/checkpoint_best_loss_compression.pth.tar')
    tinylic_model = TinyLIC() 
    tinylic_model = tinylic_model.to(device)
    
    tinylic_model.load_state_dict(checkpoint["state_dict"], strict=False)
    # compression_optimizer.load_state_dict(checkpoint["optimizer"])
    # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    # train_merge()
    
    # yolo_model.train_with_compression(compression_model=tinylic_model, compression_optimizer=compression_optimizer, aux_optimizer=aux_optimizer, compression_criterion=compression_criterion, data='/home/englishassignment123/work/baseline/vcod/VOC.yaml', epochs=500, imgsz=256, batch=8, workers=4, pretrained=True)
    yolo_model.val(tinylic_model, device, data='/home/englishassignment123/work/baseline/vcod/VOC.yaml', imgsz=256, batch=8, workers=4, pretrained=True)