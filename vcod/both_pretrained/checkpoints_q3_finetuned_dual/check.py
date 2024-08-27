import torch
# read and print information from the checkpoint
checkpoint = torch.load('/home/englishassignment123/work/baseline/runs/detect/train11/weights/best.pt')
print(checkpoint['epoch'])
