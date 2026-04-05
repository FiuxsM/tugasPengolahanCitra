import cv2
import numpy as np
# Load gambar
image = cv2.imread('foto.jpg')
if image is None:
	print("Gambar tidak ditemukan!")
	
else:
	# Langsung Sharpening
 kernel_sharpen = np.array([[0, -1, 0],
[-1, 5, -1],
[0, -1, 0]])
sharpened_direct = cv2.filter2D(image, -1, kernel_sharpen)
# Menggabungkan gambar Before dan After secara horizontal
comparison_direct = np.hstack((image, sharpened_direct))
# Tampilkan hasil
cv2.imshow('Soal 2: Before (Left) vs After Direct Sharpen (Right)',
comparison_direct)
cv2.imwrite('perbandingan_soal_2.jpg', comparison_direct)
cv2.waitKey(0)
cv2.destroyAllWindows()