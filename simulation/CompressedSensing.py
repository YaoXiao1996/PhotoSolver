import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack as spfft
import pandas as pd

image = np.array(Image.open("..\lena.jpg").convert('L'))

m, n = image.shape
block_size = 16
sample_rate = 0.8

data_length = block_size * block_size
# measurement matrix
Phi = np.random.randn(int(sample_rate * data_length), data_length)

# Transformation matrix
I_mn = np.eye(data_length, data_length)
Psi = spfft.dct(I_mn, norm='ortho')

# Sensing matrix
Theta = Phi @ Psi

plt.ion()
x_recover = np.zeros((m, n))
for block_row in range(16):
    for block_column in range(16):
        print("正在重建第{}行，第{}列".format(block_row, block_column))
        x = image[block_size * block_row:block_size * (block_row + 1), block_size * block_column:block_size * (block_column + 1)].reshape((-1, 1))
        y = Phi @ x
        theory_s = Psi.T @ x
        s = np.zeros((data_length, 1))
        learning_rate = 0.001
        tao = 0.01 * np.max(Theta.T @ y)
        loss_list = []
        for iteration in range(2000):
            y_hat = Theta @ s
            loss1 = 0.5 * np.sum(np.power(y_hat - y, 2))
            loss2 = tao * np.sum(np.abs(s))
            loss = loss1 + loss2
            partial_1 = Theta.T @ (y_hat - y)
            partial_2 = tao * np.sign(s)
            s = s - learning_rate * (partial_1 + partial_2)
            loss_list.append(loss)
            # if iteration % 10 == 0:
            #     plt.clf()
            #     plt.title("column = {}, iteration = {}".format(column, iteration))
            #     plt.plot(theory_s, label="theory s", alpha=0.5)
            #     plt.plot(s, label="s", alpha=0.5)
            #     plt.legend()
            #     plt.show()
            #     plt.pause(0.1)
            if iteration % 200 == 199:
                learning_rate = learning_rate * 0.5
                tao = tao * 0.5
        x_re = Psi @ s
        x_recover[block_size * block_row:block_size * (block_row + 1), block_size * block_column:block_size * (block_column + 1)] = x_re.reshape((block_size,block_size))
plt.ioff()

fig = plt.figure()
plt.matshow(image, cmap="gray")
plt.matshow(x_recover, cmap="gray")
diff = image - x_recover
MSE = np.mean(np.square(diff))
PSNR = 10 * np.log10(255 * 255 / MSE)
print(PSNR)
plt.show()