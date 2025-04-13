import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

# フォルダのパス
original_image_dir = '/Users/katotaisei/Library/CloudStorage/OneDrive-個人用/2025年研究/PDS/image'
restored_image_dir = '/Users/katotaisei/Library/CloudStorage/OneDrive-個人用/2025年研究/PDS/PDS_result'

# 対象となる画像の種類（suffix）
suffixes = ['(circuit_noise)', '(no_circuit_noise)', '(noise)']

# 画像の数
num_images = 20

# ループで処理
for i in range(num_images):
    # 原画像ファイル名
    original_filename = f"{i}.bmp"
    original_path = os.path.join(original_image_dir, original_filename)
    
    # 原画像を読み込み
    original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Original image {original_filename} not found.")
        continue

    for suffix in suffixes:
        # 劣化画像ファイル名
        degraded_filename = f"γ=10_{i}{suffix}.png"
        degraded_path = os.path.join(restored_image_dir, degraded_filename)
        
        # 劣化画像を読み込み
        degraded_image = cv2.imread(degraded_path, cv2.IMREAD_GRAYSCALE)
        if degraded_image is None:
            print(f"Degraded image {degraded_filename} not found.")
            continue
        
        # PSNRとSSIMの計算
        psnr_value = psnr(original_image, degraded_image)
        ssim_value = ssim(original_image, degraded_image, data_range=255)
        
        # 結果の表示
        print(f"[{i}] {suffix}")
        print(f"   PSNR: {psnr_value:.2f}")
        print(f"   SSIM: {ssim_value:.4f}")
