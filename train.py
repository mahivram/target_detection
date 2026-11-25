from ultralytics import YOLO


def main():
    model = YOLO("yolo26s.pt")

    results = model.train(
        data="./dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project="runs/detect",
        name="target_detection",
    )

    print("Training completed!")


if __name__ == "__main__":
    main()