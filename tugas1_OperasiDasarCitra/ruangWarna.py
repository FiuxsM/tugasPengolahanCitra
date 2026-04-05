import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_operasi_warna():
    print("Menjalankan Operasi Warna...")

    # baca gambar
    img = cv2.imread('images/foto.jpg')

    # cek kalau gambar ga ada
    if img is None:
        print("Gambar tidak ditemukan")
        return

    # ubah dari BGR ke RGB biar sesuai matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # tambah brightness
    bright = cv2.convertScaleAbs(img, alpha=1, beta=50)

    # tambah contrast
    contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    # negatif image
    negative = 255 - img

    # ambil channel merah
    R = img.copy()
    R[:,:,1] = 0
    R[:,:,2] = 0

    # ambil channel hijau
    G = img.copy()
    G[:,:,0] = 0
    G[:,:,2] = 0

    # ambil channel biru
    B = img.copy()
    B[:,:,0] = 0
    B[:,:,1] = 0

    # tampilkan semua hasil
    plt.figure(figsize=(15,10))

    images = [img, bright, contrast, negative, R, G, B]
    titles = ["Original", "Brightness +50", "Contrast x1.5",
              "Negative", "Red Channel", "Green Channel", "Blue Channel"]

    for i in range(len(images)):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_operasi_warna()