import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Load gambar grayscale
img = cv2.imread('images/foto.jpg', 0)

if img is None:
    print("File gambar tidak ditemukan!")
    exit()

# ======================
# DWT (Wavelet Haar)
# ======================
coeffs2 = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeffs2

# ======================
# DENOISING (threshold detail)
# ======================
threshold = 20  # bisa diatur (semakin besar = makin halus)

LH_d = pywt.threshold(LH, threshold, mode='soft')
HL_d = pywt.threshold(HL, threshold, mode='soft')
HH_d = pywt.threshold(HH, threshold, mode='soft')

# ======================
# Inverse DWT (balik ke gambar)
# ======================
coeffs2_denoised = (LL, (LH_d, HL_d, HH_d))
img_denoised = pywt.idwt2(coeffs2_denoised, 'haar')

# ======================
# Visualisasi
# ======================
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(132)
plt.imshow(LL, cmap='gray')
plt.title('LL (Aproksimasi)')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_denoised, cmap='gray')
plt.title('Hasil Denoising Wavelet')
plt.axis('off')

plt.suptitle('Wavelet Denoising Multi-Resolusi (Haar)', fontsize=14)
plt.show()