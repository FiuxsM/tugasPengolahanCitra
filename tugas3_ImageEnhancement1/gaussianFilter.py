import cv2
import numpy as np
# Load gambar
image = cv2.imread('foto.jpg')
if image is None:
	print("Gambar tidak ditemukan!")
else:
# 1. Smoothing (Gaussian Blur)
 smoothed = cv2.GaussianBlur(image, (5, 5), 0)
# 2. Sharpening
kernel_sharpen = np.array([[0,
-1, 0],
[-1, 5,
-1],
[0,
-1, 0]])
sharpened = cv2.filter2D(smoothed,
-1, kernel_sharpen)
# Menggabungkan gambar Before dan After secara horizontal
# Pastikan kedua gambar memiliki dimensi yang sama
comparison = np.hstack((image, sharpened))
# Tampilkan hasil
cv2.imshow('Soal 1: Before (Left) vs After Smooth-Sharpen (Right)',
comparison)
cv2.imwrite('perbandingan_soal_1.jpg', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()