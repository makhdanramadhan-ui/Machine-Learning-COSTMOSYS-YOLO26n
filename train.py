import argparse
import os

import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--data", default=os.path.join(HERE, "data.yaml"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--project", default=os.path.join(HERE, "runs"))
    ap.add_argument("--name", default="yolo26n_640_s0")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=0,
        device=a.device,
        seed=a.seed,
        deterministic=True,
        patience=a.patience,
        cache=False,
        project=a.project,
        name=a.name,
        exist_ok=True,
        plots=True,
        val=True,
        verbose=True,
        amp=True,
        cls=0.5,
    )
    print("BEST", model.trainer.save_dir)


if __name__ == "__main__":
    main()
