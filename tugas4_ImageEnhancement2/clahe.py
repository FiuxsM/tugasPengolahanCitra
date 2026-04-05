import cv2
import matplotlib.pyplot as plt
# Baca gambar grayscale
img = cv2.imread('images/foto.jpg', 0)
if img is None:
	print("Gambar tidak ditemukan!")
	exit()
#low contrast
low_contrast = cv2.normalize(img, None, alpha=50, beta=150,
norm_type=cv2.NORM_MINMAX)
#clahe
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
out = clahe.apply(low_contrast)
#hasil
cv2.imwrite('cache/low_contrast.png', low_contrast)
cv2.imwrite('cache/clahe.png', out)
#output
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Sumber')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.title('Low Contrast')
plt.imshow(low_contrast, cmap='gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.title('Hasil Clahe')
plt.imshow(out, cmap='gray')
plt.axis('off')
plt.show()