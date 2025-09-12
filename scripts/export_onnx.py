import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="đường dẫn .pt (vd: yolov8n-face.pt)")
    p.add_argument("--device", type=str, default="cuda")  # cpu|cuda
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--half", action="store_true")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--simplify", action="store_true", default=True)
    args = p.parse_args()

    model = YOLO(args.weights)
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        verbose=True,
    )
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()