##########PSNR,SSIM計算用ファイル##########

import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_ssim(image1_path, image2_path):
    # 画像をグレースケールで読み込む
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    print(image1_path)
    
    if img1 is None or img2 is None:
        print("Error: 画像が読み込めません。")
        return None, None
    
    if img1.shape != img2.shape:
        print("Error: 画像のサイズが一致しません。")
        return None, None
    
    # PSNRの計算
    psnr_value = psnr(img1, img2, data_range=255)
    
    # SSIMの計算
    ssim_value = ssim(img1, img2, data_range=255)
    
    return psnr_value, ssim_value

if __name__ == "__main__":
    path = "/Users/katotaisei/Library/CloudStorage/OneDrive-個人用/2024年研究/program/M1_3~/PDS-latest/"
    image1_path = os.path.join(path, "0.bmp")  # 比較する画像1のパス
    image2_path = os.path.join(path, "0(noise).png")  # 比較する画像2のパス
    
    psnr_value, ssim_value = calculate_psnr_ssim(image1_path, image2_path)
    
    if psnr_value is not None and ssim_value is not None:
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
