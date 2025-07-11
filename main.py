from ultralytics import YOLO
import os

def train_yolo():
    # ✅ 设置保存目录，TensorBoard 会读取这里的日志
    project_dir = "runs/tb_logs"
    experiment_name = "exp1"

    # ✅ 加载模型（可以是 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt）
    model = YOLO("yolov8s.pt")  # 你也可以换成 yolov8n.pt 等

    # ✅ 训练
    model.train(
        data="data.yaml",         # 你的数据配置文件路径
        epochs=100,                # 训练轮数
        imgsz=448,                # 输入图像尺寸
        project=project_dir,      # 日志保存目录
        name=experiment_name,     # 子目录名
        device=0,                 # 使用 GPU，如果你要用 CPU 就写 device='cpu'
    )

    print(f"✅ Training complete. Logs saved in: {os.path.join(project_dir, experiment_name)}")

if __name__ == "__main__":
    train_yolo()
