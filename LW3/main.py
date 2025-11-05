import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def rotate_and_crop(image, angle):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    rotation_matrix[0, 2] += (new_width / 2) - center_x
    rotation_matrix[1, 2] += (new_height / 2) - center_y
    rotated = cv.warpAffine(image, rotation_matrix, (new_width, new_height),
                            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    cropped = crop_black_borders_aggressive(rotated)
    return cropped


def crop_black_borders_aggressive(image, threshold=10):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    mask = gray > threshold
    if not np.any(mask):
        return image
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    padding = 2
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_max = min(image.shape[1], x_max + padding)
    cropped = image[y_min:y_max + 1, x_min:x_max + 1]
    return cropped


def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2)
    s1 = np.float32(s1)
    s1 = s1 * s1
    sse = s1.sum()
    if sse <= 1e-10:
        return 0
    else:
        shape = I1.shape
        p = 1
        for i in shape:
            p *= i
        mse = 1.0 * sse / p
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr


def getSSIM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    I1 = np.float32(i1)
    I2 = np.float32(i2)
    I2_2 = I2 * I2
    I1_2 = I1 * I1
    I1_I2 = I1 * I2
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2
    ssim_map = cv.divide(t3, t1)
    ssim = cv.mean(ssim_map)
    ssim = ssim[:3]
    return np.mean(ssim)


if __name__ == "__main__":
    color_image = cv.imread("original.jpg")
    aligned_color = rotate_and_crop(color_image, 260)
    aligned_bw = cv.cvtColor(aligned_color, cv.COLOR_BGR2GRAY)
    aligned_color_rgb = cv.cvtColor(aligned_color, cv.COLOR_BGR2RGB)
    original_color_rgb = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)

    kernel55 = np.ones((5, 5), np.float32) / 25
    kernel77 = np.ones((7, 7), np.float32) / 49

    filtered_color_55 = cv.filter2D(aligned_color_rgb, -1, kernel55)
    filtered_color_77 = cv.filter2D(aligned_color_rgb, -1, kernel77)
    gaussian_color_7 = cv.GaussianBlur(aligned_color_rgb, (7, 7), 0)
    gaussian_color_15 = cv.GaussianBlur(aligned_color_rgb, (15, 15), 0)
    median_color_5 = cv.medianBlur(aligned_color_rgb, 5)

    plt.figure(figsize=(15, 12))
    plt.suptitle('Цветная фильтрация (выровненное изображение)', fontsize=16, fontweight='bold')
    color_images = [
        (aligned_color_rgb, 'Выровненное цветное изображение'),
        (filtered_color_55, f'Средняя линейная 5x5\nPSNR = {getPSNR(aligned_color_rgb, filtered_color_55):.3f}\nSSIM = {getSSIM(aligned_color_rgb, filtered_color_55):.3f}'),
        (filtered_color_77, f'Средняя линейная 7x7\nPSNR = {getPSNR(aligned_color_rgb, filtered_color_77):.3f}\nSSIM = {getSSIM(aligned_color_rgb, filtered_color_77):.3f}'),
        (gaussian_color_7, f'Гауссовская 7x7\nPSNR = {getPSNR(aligned_color_rgb, gaussian_color_7):.3f}\nSSIM = {getSSIM(aligned_color_rgb, gaussian_color_7):.3f}'),
        (gaussian_color_15, f'Гауссовская 15x15\nPSNR = {getPSNR(aligned_color_rgb, gaussian_color_15):.3f}\nSSIM = {getSSIM(aligned_color_rgb, gaussian_color_15):.3f}'),
        (median_color_5, f'Медианная 5x5\nPSNR = {getPSNR(aligned_color_rgb, median_color_5):.3f}\nSSIM = {getSSIM(aligned_color_rgb, median_color_5):.3f}')
    ]
    for i, (image, title) in enumerate(color_images, 1):
        plt.subplot(2, 3, i)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('color.jpg')
    plt.show()

    filtered_bw_55 = cv.filter2D(aligned_bw, -1, kernel55)
    filtered_bw_77 = cv.filter2D(aligned_bw, -1, kernel77)
    gaussian_bw_7 = cv.GaussianBlur(aligned_bw, (7, 7), 0)
    gaussian_bw_15 = cv.GaussianBlur(aligned_bw, (15, 15), 0)
    median_bw_5 = cv.medianBlur(aligned_bw, 5)

    plt.figure(figsize=(15, 12))
    plt.suptitle('Черно-белая фильтрация (выровненное изображение)', fontsize=16, fontweight='bold')
    bw_images = [
        (aligned_bw, 'Выровненное ЧБ изображение'),
        (filtered_bw_55, f'Средняя линейная 5x5\nPSNR = {getPSNR(aligned_bw, filtered_bw_55):.3f}\nSSIM = {getSSIM(aligned_bw, filtered_bw_55):.3f}'),
        (filtered_bw_77, f'Средняя линейная 7x7\nPSNR = {getPSNR(aligned_bw, filtered_bw_77):.3f}\nSSIM = {getSSIM(aligned_bw, filtered_bw_77):.3f}'),
        (gaussian_bw_7, f'Гауссовская 7x7\nPSNR = {getPSNR(aligned_bw, gaussian_bw_7):.3f}\nSSIM = {getSSIM(aligned_bw, gaussian_bw_7):.3f}'),
        (gaussian_bw_15, f'Гауссовская 15x15\nPSNR = {getPSNR(aligned_bw, gaussian_bw_15):.3f}\nSSIM = {getSSIM(aligned_bw, gaussian_bw_15):.3f}'),
        (median_bw_5, f'Медианная 5x5\nPSNR = {getPSNR(aligned_bw, median_bw_5):.3f}\nSSIM = {getSSIM(aligned_bw, median_bw_5):.3f}')
    ]
    for i, (image, title) in enumerate(bw_images, 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gray.jpg')
    plt.show()