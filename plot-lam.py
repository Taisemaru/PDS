import matplotlib.pyplot as plt
import numpy as np
import cv2

color_array=['blue','red','violet','turquoise','green']
plt.rcParams["font.size"] = 11
plt.tight_layout()

index_array=[0, 1, 2, 3, 4]
LAM_array=np.array([0.02, 0.03, 0.04, 0.05, 0.06])
EP=50

PSNR=np.loadtxt('PSNR-lam.txt')
SSIM=np.loadtxt('SSIM-lam.txt')
PSNR_noise = np.loadtxt('PSNR-lam-optnoise.txt')
SSIM_noise = np.loadtxt('SSIM-lam-optnoise.txt')

#PSNRの推移を表示
x=np.linspace(0, EP+1, EP+1)

#fig=plt.figure(figsize=(8.6))
#ax=fig.add_subplot(1,1,1)

for i in index_array:
  print(i)
  plt.plot(x, PSNR[i], color=color_array[i], label=f'noiseless(λ={LAM_array[i]})',linestyle='--')
  plt.plot(x, PSNR_noise[i], color=color_array[i], label=f'noisy(λ={LAM_array[i]})')
#plt.title('ADMM_TV')
plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Number of iterations')
plt.ylabel('PSNR')
#plt.xticks([10,20,30,40,50],fontsize=9,color='black')
plt.legend(loc='lower left', fontsize=11, bbox_to_anchor=(1, 0))
#plt.ylim(21.5,35)
#plt.savefig('plot(PSNR-lambda-256).pdf')
plt.show()

#SSIMの推移を表示
x=np.linspace(0, EP+1, EP+1)
for i in index_array:
  plt.plot(x, SSIM[i], color=color_array[i], label=f'noiseless(λ={LAM_array[i]})',linestyle='--')
  plt.plot(x, SSIM_noise[i], color=color_array[i], label=f'noisy(λ={LAM_array[i]})')
#plt.title('ADMM_TV')
plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Number of iterations')
plt.ylabel('SSIM')
plt.legend(loc='lower left', fontsize=12, bbox_to_anchor=(1, 0))
#plt.ylim(0.35,0.95)
#plt.savefig('plot(SSIM-lambda-256).pdf')
plt.show()

