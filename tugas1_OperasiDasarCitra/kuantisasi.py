import numpy as np
import matplotlib.pyplot as plt

# fungsi buat kuantisasi gambar
def quantize(img, levels):
    # kalau gambar masih float (0-1), ubah ke 0-255
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # hitung step tiap level
    step = 256 // levels

    # proses kuantisasi
    return (img // step) * step

def run_kuantisasi():
    print("Menjalankan Kuantisasi...")

    # baca gambar
    img = plt.imread('images/foto.jpg')

    # kuantisasi jadi 16 level
    img_quant = quantize(img, 16)

    # tampilkan sebelum & sesudah
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Before")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_quant)
    plt.title("After (16 Levels)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_kuantisasi()