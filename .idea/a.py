import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2

# Giriş bilgileri
KULLANICI_ADI = "admin"
SIFRE = "1234"

# YOLO tespiti
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

# Giriş ekranı
def giris_ekrani():
    giris = tk.Tk()
    giris.withdraw()  # Küçük giriş penceresi görünmesin

    kullanici = simpledialog.askstring("Giriş", "Kullanıcı adı:")
    sifre = simpledialog.askstring("Giriş", "Şifre:", show="*")

    if kullanici == KULLANICI_ADI and sifre == SIFRE:
        giris.destroy()
        baslat_arayuz()
    else:
        messagebox.showerror("Hata", "Geçersiz kullanıcı adı veya şifre!")
        giris.destroy()

# Ana arayüz
def baslat_arayuz():
    global label

    root = tk.Tk()
    root.title("YOLO Nesne Tanıma")
    root.geometry("800x600")

    btn = tk.Button(root, text="Resim Seç", command=browse_image)
    btn.pack(pady=10)

    label = tk.Label(root)
    label.pack()

    root.mainloop()

# Uygulamayı başlat
giris_ekrani()
