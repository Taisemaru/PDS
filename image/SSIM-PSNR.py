import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np


img_num=3 #画像枚数
SIZE=256 #画像サイズ

#原画像を読み込み
img=np.zeros((SIZE,SIZE))
img = cv2.imread("0.bmp",cv2.IMREAD_GRAYSCALE)


est=np.zeros((SIZE,SIZE))
est = cv2.imread("0(est)1-gam2.png",cv2.IMREAD_GRAYSCALE)

psnr_value = psnr(img, est)
ssim_value = ssim(img, est, data_range=255)

print("PSNR:", psnr_value)
print("SSIM:", ssim_value)
"""
# 画像ファイルのパス
original_image_path = '/Users/katotaisei/Library/CloudStorage/OneDrive-個人用/program/M1_3~/PDS-latest/image/0.bmp'
restored_image_path = '/Users/katotaisei/Library/CloudStorage/OneDrive-個人用/program/M1_3~/PDS-latest/image/0(est)3-gam1.png'

# 画像を読み込む
original_image = cv2.imread(original_image_path)
restored_image = cv2.imread(restored_image_path)

# PSNRを算出する
psnr_value = psnr(original_image, restored_image)

# SSIMを算出する
ssim_value = ssim(original_image, restored_image, data_range=255)

print("PSNR:", psnr_value)
print("SSIM:", ssim_value)
"""