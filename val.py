from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    pth_path = r"D:\PycharmProjects\YOLOv8.2\ultralytics\runs\detect\train\weights\best.pt"
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model
    conf=0.001


    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
