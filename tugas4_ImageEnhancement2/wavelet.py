import cv2
import pywt
import matplotlib.pyplot as plt
# Load gambar dalam mode grayscale
img = cv2.imread('images/foto.jpg', 0)
if img is None:
	print("File gambar tidak ditemukan!")
else:
# --- Proses Wavelet Transform (Metode Haar) ---
# Melakukan dekomposisi 1 level
# LL: Aproksimasi, (LH, HL, HH): Detail Horizontal, Vertikal, Diagonal
 coeffs2 = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeffs2
# --- Visualisasi Before & After ---
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Sumber (Citra Asli)')
plt.axis('off')
plt.subplot(122)
# Kita tampilkan LL sebagai hasil transformasi 'After'
plt.imshow(LL, cmap='gray')
plt.title('AFTER (Wavelet LL - Aproksimasi)')
plt.axis('off')
plt.suptitle('Transformasi Wavelet (Discrete Wavelet Transform)',
			 fontsize=16)
plt.show()