import cv2
import os
import matplotlib.pyplot as plt
from skimage import exposure
# Buat folder cache jika belum ada
os.makedirs('cache', exist_ok=True)
reference = cv2.imread('images/foto.jpg', 0)
if reference is None:
	print("File tidak ditemukan")
	exit()
source = cv2.normalize(reference, None, alpha=150, beta=300,
norm_type=cv2.NORM_MINMAX)
if source is None or reference is None:
	print("File tidak ditemukan")
	exit()
# histogram matching
matched = exposure.match_histograms(source, reference,
channel_axis=None)
# Konversi ke uint8 agar bisa disimpan oleh OpenCV
matched = matched.astype('uint8')
# Simpan hasil
cv2.imwrite('cache/source.png', source)
cv2.imwrite('cache/reference.png', reference)
cv2.imwrite('cache/matched.png', matched)
# output
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Sumber')
plt.imshow(source, cmap='gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.title('Refrensi')
plt.imshow(reference, cmap='gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.title('Hasil Matching')
plt.imshow(matched, cmap='gray')
plt.axis('off')
plt.show()