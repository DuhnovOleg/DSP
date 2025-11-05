import cv2
import matplotlib.pyplot as plt

def remove_lines(image_path, output_path1='output1.png', output_path2='output2.png'):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(gray)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    horizontal_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    result = cv2.inpaint(gray, horizontal_lines, 3, cv2.INPAINT_TELEA)

    _, thresh1 = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    thresh3 = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    cv2.imwrite(output_path1, result)
    cv2.imwrite(output_path2, thresh2)

    plt.figure(figsize=(15, 10))

    images = [
        (gray, 'Оригинал (grayscale)'),
        (inverted, 'Инвертированный'),
        (horizontal_lines, 'Горизонтальные линии'),
        (result, 'После удаления линий'),
        (thresh1, 'Бинаризация Otsu'),
        (thresh2, 'Adaptive Mean'),
        (thresh3, 'Adaptive Gaussian')
    ]

    for i, (image, title) in enumerate(images, 1):
        plt.subplot(3, 3, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return thresh1

if __name__ == "__main__":
    remove_lines('original.png')