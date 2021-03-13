import cv2
import numpy as np


def conv(img, kernel, strides):
    Fx = kernel.shape[0]
    Fy = kernel.shape[1]
    Nx = img.shape[0]
    Ny = img.shape[1]

    xOutput = int(((Nx - Fx) / strides) + 1)
    yOutput = int(((Ny - Fy) / strides) + 1)
    res = np.zeros((xOutput, yOutput), np.float32)

    for y in range(Ny - Fy + 1):
        if y % strides == 0:
            for x in range(Nx - Fx + 1):
                if x % strides == 0:
                    res[x][y] = (kernel * img[x: x + Fx, y: y + Fy]).sum()

    return res


def relu(input_arr):
    res = np.zeros((input_arr.shape[0], input_arr.shape[1], input_arr.shape[2]), np.float32)
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            for k in range(input_arr.shape[2]):
                res[i, j, k] = max(0, input_arr[i, j, k])
    return res


def max_pooling(input_arr):
    stride = 2
    res = np.zeros((int(input_arr.shape[0] / 2), int(input_arr.shape[1] / 2), input_arr.shape[2]), np.float32)
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            for k in range(res.shape[2]):
                origI = i * stride
                origJ = j * stride
                res[i, j, k] = np.max(input_arr[origI:origI + stride, origJ:origJ + stride, k])
    return res


def main():
    img = cv2.imread("Lena.png")
    Nx = img.shape[0]
    Ny = img.shape[0]
    F = 3
    C = 3
    S = 1
    M = 5

    xOutput = int(((Nx - F) / S) + 1)
    yOutput = int(((Ny - F) / S) + 1)
    output = np.zeros((xOutput, yOutput, M), dtype=np.float32)
    # Conv
    for m in range(M):
        kernel = np.random.uniform(-5.0, 5.0, (F, F, C))
        output[:, :, m] = conv(img, kernel, S)

    # Relu
    output = relu(output)
    # Max pooling
    output = max_pooling(output)

    print(output)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
