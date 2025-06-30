import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

# BU FONKSİYONU YOLO MODELİNLE DOLDURACAKSIN
def detect_objects(image_path):
    from ultralytics import YOLO

    model = YOLO("bestik.pt")  # Kendi modelin varsa buraya yaz
    results = model(image_path, conf=0.05)

    result = results[0]
    result_image = result.plot()  # kutular çizilmiş görüntü

    return result_image

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detected_image = detect_objects(file_path)
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(detected_image)
        img_tk = ImageTk.PhotoImage(image=img)
        label.config(image=img_tk)
        label.image = img_tk

# ARAYÜZ TASARIMI
root = tk.Tk()
root.title("YOLO Nesne Tanıma")
root.geometry("800x600")

btn = tk.Button(root, text="Resim Seç", command=browse_image)
btn.pack(pady=10)

label = tk.Label(root)
label.pack()

root.mainloop()
