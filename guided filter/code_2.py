import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def guided_filter(I, p, r, eps):
    I = np.float32(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY))
    p = np.float32(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY))

    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    q = a * I + b

    # 归一化像素值到 [0, 255] 范围内
    q_normalized = cv2.normalize(q, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return q_normalized

def calculate_metrics(original, processed):
    mse = mean_squared_error(original, processed)
    psnr = peak_signal_noise_ratio(original, processed)
    ssim = structural_similarity(original, processed)
    return mse, psnr, ssim

image = cv2.imread(r"lena.jpg")
image_1 = cv2.imread(r"image.png")
image_2 = cv2.imread(r"moon.jpg")
image_list = [image, image_1, image_2]

window_size = 5
r = 16
eps = 1000

for img in image_list:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用高斯滤波平滑图像
    gaussian_smoothed_image = cv2.GaussianBlur(gray_img, (window_size, window_size), 0)

    # 使用箱形滤波平滑图像
    box_smoothed_image = cv2.blur(gray_img, (window_size, window_size))

    # 使用引导滤波器平滑图像
    guided_filtered_image = guided_filter(img, img, r, eps)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(gaussian_smoothed_image, cmap='gray')
    plt.title('Gaussian Smoothed Image')

    plt.subplot(1, 4, 3)
    plt.imshow(box_smoothed_image, cmap='gray')
    plt.title('Box Smoothed Image')

    plt.subplot(1, 4, 4)
    plt.imshow(guided_filtered_image, cmap='gray')
    plt.title('Guided Filtered Image')

    # 计算指标
    mse_original, psnr_original, ssim_original = calculate_metrics(gray_img, gray_img)
    mse_gaussian, psnr_gaussian, ssim_gaussian = calculate_metrics(gray_img, gaussian_smoothed_image)
    mse_box, psnr_box, ssim_box = calculate_metrics(gray_img, box_smoothed_image)
    mse_guided, psnr_guided, ssim_guided = calculate_metrics(gray_img, guided_filtered_image)

    # 打印指标数据
    print("Original Image Metrics: ")
    print(f"MSE: {mse_original:.2f}, PSNR: {psnr_original:.2f}, SSIM: {ssim_original:.2f}")
    print("Gaussian Smoothed Image Metrics: ")
    print(f"MSE: {mse_gaussian:.2f}, PSNR: {psnr_gaussian:.2f}, SSIM: {ssim_gaussian:.2f}")
    print("Box Smoothed Image Metrics: ")
    print(f"MSE: {mse_box:.2f}, PSNR: {psnr_box:.2f}, SSIM: {ssim_box:.2f}")
    print("Guided Filtered Image Metrics: ")
    print(f"MSE: {mse_guided:.2f}, PSNR: {psnr_guided:.2f}, SSIM: {ssim_guided:.2f}")

    plt.show()
