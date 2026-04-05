import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# folder cache
os.makedirs('cache', exist_ok=True)
img = cv2.imread('images/foto.jpg', 0)
if img is None:
	print("gambar ga ketemu bro")
	exit()
# blur = less noise
blur = cv2.GaussianBlur(img, (0, 0), 1.2)
# morfo
kernel = np.ones((3, 3), np.uint8)
# opening: hapus noise kecil
opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
# closing: nutup lubang kecil
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# gradient = edge detection
gx = cv2.Sobel(closing, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(closing, cv2.CV_32F, 0, 1, ksize=3)
# combine gradien x and y
grad = cv2.magnitude(gx, gy)
# normalize gradien ke range 0-255
grad_norm = cv2.normalize(grad, None, 0, 255,
cv2.NORM_MINMAX).astype('uint8')
# combine: gambar + edge
out = np.clip(closing + 0.7 * grad_norm, 0,
255).astype('uint8')
# simpan hasil
cv2.imwrite('cache/1_input.png', img)
cv2.imwrite('cache/2_before_final.png', grad_norm)
cv2.imwrite('cache/3_final_output.png', out)
# ===== output pakai plt =====
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.title('Input')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.title('Gradient')
plt.imshow(grad_norm, cmap='gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.title('Final Output')
plt.imshow(out, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()