import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO("best.pt")  # kendi eğittiğin modelin adı

# Kamera başlat (0 -> varsayılan webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO ile nesne tespiti yap
    results = model(frame, conf=0.5)  # İstersen conf ayarını değiştir

    # Sonuçları çiz
    result_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow("YOLO - Canlı Kamera", result_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
