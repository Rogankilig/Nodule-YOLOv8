import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics\cfg\models/v8/Nodule-YOLOv8.yaml")
    model.train(data=r'ultralytics\cfg\datasets\nodule.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                resume=True,
                workers=0,
                device=0,
                optimizer='SGD',  # using SGD
                amp=False,  # close amp
                )