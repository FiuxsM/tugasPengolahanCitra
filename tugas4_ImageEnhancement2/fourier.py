import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar grayscale
img = cv2.imread('images/foto.jpg', 0)

if img is None:
    print("File gambar tidak ditemukan!")
    exit()

# ======================
# FFT (ke domain frekuensi)
# ======================
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# magnitude spectrum (biar kelihatan)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# ======================
# LOW PASS FILTER
# ======================
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # titik tengah

# bikin mask (lingkaran tengah = frekuensi rendah)
mask = np.zeros((rows, cols), np.uint8)
r = 50  # radius (atur sendiri, kecil = lebih blur)
mask[crow-r:crow+r, ccol-r:ccol+r] = 1

# apply filter
fshift_filtered = fshift * mask

# ======================
# Balik ke gambar (inverse FFT)
# ======================
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# ======================
# Visualisasi
# ======================
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(132)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_back, cmap='gray')
plt.title('Hasil Low Pass Filter (Blur)')
plt.axis('off')

plt.suptitle('Transformasi Fourier + Filtering', fontsize=14)
plt.show()