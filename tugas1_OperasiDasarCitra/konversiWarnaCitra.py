import cv2
import matplotlib.pyplot as plt

def run_konversi_citra():
    print("Menjalankan Konversi Citra Warna...")

    # baca gambar
    img = cv2.imread('images/foto.jpg')

    # cek kalau gambar ga ada
    if img is None:
        print("Gambar tidak ditemukan")
        return

    # ubah ke RGB (default opencv itu BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # konversi ke grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # konversi ke HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # konversi ke LAB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # balik dari HSV ke RGB
    rgb_from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # tampilkan hasil
    plt.figure(figsize=(15,10))

    images = [img_rgb, gray, hsv, lab, rgb_from_hsv]
    titles = ["Original (RGB)", "Grayscale", "HSV", "LAB", "HSV to RGB"]

    for i in range(len(images)):
        plt.subplot(2,3,i+1)

        # kalau grayscale pakai cmap
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i])

        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_konversi_citra()