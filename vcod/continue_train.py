from ultralytics import YOLOv10

yolo_model = YOLOv10('/home/fumchin/work/baseline/object-detection/yolov10/runs/detect/train12/weights/last.pt')
yolo_model.train(data='VOC.yaml', epochs=500, imgsz=640, batch=32, workers=0, pretrained=True, resume=True)