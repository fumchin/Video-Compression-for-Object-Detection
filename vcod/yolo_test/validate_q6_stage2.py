import argparse

import torch
from ultralytics import YOLOv10

# Set up argument parser
parser = argparse.ArgumentParser(description='Run YOLOv10 q6 test stage2 e-6.')
parser.add_argument('--yaml', type=str, required=True, help='Path to the YAML file for validation')

# Parse arguments
args = parser.parse_args()

# Load model
model = YOLOv10('/home/englishassignment123/work/baseline/runs/detect/train7/weights/best.pt')
# model = YOLOv10('/home/englishassignment123/work/baseline/vcod/both_pretrained/checkpoints_q6_pretrained_2_e-6/checkpoint_yolo_0.pt');
# checkpoint = torch.load('/home/englishassignment123/work/baseline/vcod/checkpoints_q3/checkpoint_best_loss_yolo.pth.tar')
# model.load_state_dict(checkpoint["state_dict"])



# Run validation with the specified YAML file
model.val(data=args.yaml, batch=8)