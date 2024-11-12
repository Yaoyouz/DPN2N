import cv2
import numpy as np


def bilateral_filter(images, diameter=5, sigma_color=0.3, sigma_space=0.3):
    """
    双边滤波函数
    :param image: 输入图像
    :param diameter: 滤波核直径，控制双边滤波器的范围
    :param sigma_color: 控制颜色空间的高斯滤波器的标准差
    :param sigma_space: 控制坐标空间的高斯滤波器的标准差
    :return: 滤波后的图像
    """
    if images.dtype != np.float32:
        images = images.astype(np.float32)
    n_images = images.shape[0]
    images_filted = []
    for i in range(n_images):
        image = np.squeeze(images[i])
        image_filted = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        images_filted.append(image_filted)
    images_filted = np.array(images_filted)
    images_filted = np.expand_dims(images_filted, axis=1)
    return images_filted