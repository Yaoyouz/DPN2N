import numpy as np
import matplotlib.pyplot as plt

def plot_grayscale_image(npy_file_path):
    # 读取npy文件
    data = np.load(npy_file_path)

    # 确保数据是二维的（灰度图）
    if len(data.shape) == 2:
        plt.imshow(data, cmap='gray')
        plt.title('Grayscale Image')
        plt.colorbar()
        plt.show()
    else:
        print("输入的数据不是灰度图。")

if __name__ == "__main__":
    npy_file_path = r'C:\Users\Administrator\Desktop\OPUNAKE3D_13220.npy'
    plot_grayscale_image(npy_file_path)
