from ultralytics import YOLO
import cv2
from pathlib import Path
import shutil
import os

# 🔹 1. Önceki tahmin klasörlerini sil
predict_dir1 = Path("runs/detect/test_output1")
predict_dir2 = Path("runs/detect/test_output2")
shutil.rmtree(predict_dir1, ignore_errors=True)
shutil.rmtree(predict_dir2, ignore_errors=True)

# 🔹 2. YOLO modellerini yükle
model1 = YOLO("best.pt")  # Birinci model
model2 = YOLO("bestik.pt")  # İkinci model (dosya adını senin belirlemene göre değiştir)

# 🔹 3. Görselin yolu
image_path = "d4.jpg"  # Görsel dosyası

# 🔹 4. Model 1 ile tahmin
model1.predict(
    source=image_path,
    imgsz=640,
    conf=0.25,
    save=True,
    save_txt=True,
    project="runs/detect",
    name="test_output1",
    exist_ok=True
)

# 🔹 5. Model 2 ile tahmin
model2.predict(
    source=image_path,
    imgsz=640,
    conf=0.1,
    save=True,
    save_txt=True,
    project="runs/detect",
    name="test_output2",
    exist_ok=True
)

# 🔹 6. Çıktı görsellerini sırayla göster
for i, predict_dir in enumerate([predict_dir1, predict_dir2], start=1):
    result_images = list(predict_dir.glob("*.jpg"))
    if result_images:
        result_image = result_images[0]
        img = cv2.imread(str(result_image))
        if img is not None:
            cv2.imshow(f"Model {i} Tahmin Sonucu", img)
        else:
            print(f"❌ Model {i} çıktısı okunamadı (OpenCV).")
    else:
        print(f"❌ Model {i} için çıktı görseli bulunamadı.")

cv2.waitKey(0)
cv2.destroyAllWindows()

# 🔹 7. Klasör içeriklerini yazdır (debug için)
for i, predict_dir in enumerate([predict_dir1, predict_dir2], start=1):
    print(f"\n📂 'test_output{i}' klasöründeki dosyalar:")
    for file in predict_dir.glob("*"):
        print(f" - {file.name}")