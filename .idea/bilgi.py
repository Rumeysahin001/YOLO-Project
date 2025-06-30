from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="C:/Users/rmysh/OneDrive/Desktop/PythonProject/dataset/data.yaml")  # kendi data.yaml dosyanın yolu

# Detaylı yazdırma
print("Model Performans Bilgisi:")
print(f"🔹 Precision       : {metrics.box.p:.2f}")
print(f"🔹 Recall          : {metrics.box.r:.2f}")
print(f"🔹 mAP@0.5         : {metrics.box.map50:.2f}")
print(f"🔹 mAP@0.5:0.95    : {metrics.box.map:.2f}")
