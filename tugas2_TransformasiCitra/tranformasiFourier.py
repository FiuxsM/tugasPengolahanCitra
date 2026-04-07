import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load gambar dalam mode grayscale
img = cv2.imread('images/foto.jpg', 0)
if img is None:
	print("File gambar tidak ditemukan!")
else:
	# Mengubah ke domain frekuensi
	f = np.fft.fft2(img)
	# Geser komponen frekuensi rendah ke tengah (agar visualisasi lebih intuitif)
	fshift = np.fft.fftshift(f)
# Hitung Magnitude Spectrum (dalam skala Log agar terlihat jelas)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
# --- Visualisasi Before & After ---
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Sumber (Citra Asli)')
plt.axis('off')
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('AFTER (Magnitude Spectrum)')
plt.axis('off')
plt.suptitle('Transformasi Fourier 2D', fontsize=16)
plt.show()