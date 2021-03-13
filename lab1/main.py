import cv2
import numpy as np

# 1 read image
img = cv2.imread('lena.png')
cv2.imshow("1 input", img)
# 2 gray image
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("n2 Gray", imgGray)
# 3 equalizehist
imgEqu = cv2.equalizeHist(imgGray)
cv2.imshow("3 Good contrast", imgEqu)
# 4 Canny
imgEdge = cv2.Canny(imgGray, 100, 200)
cv2.imshow("4 Edges", imgEdge)

# 5 Corners
cornersImg = imgGray.copy()
points = cv2.goodFeaturesToTrack(cornersImg, 1000, 0.01, 10)
for i in points:
    x, y = i.ravel()
    cv2.circle(cornersImg, (x, y), 2, 255)
cv2.imshow("5 Corners", cornersImg)
# 6 distance map
distImg = cv2.bitwise_not(imgEdge)
distImg = cv2.distanceTransform(distImg, distanceType=cv2.DIST_C, maskSize=cv2.DIST_MASK_3)
disMapImg = cv2.normalize(distImg, None, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow("6 distance map", disMapImg)
cv2.waitKey(0)

# # 7 filtering
height = imgGray.shape[0]
width = imgGray.shape[1]
size_filter = 50

imgGrayF = np.zeros((height + size_filter, width + size_filter), dtype=np.uint8)
imgGrayF[0:height, 0:width] = imgGray.copy()

imgGrayF[height:, :] = imgGrayF[height - 1, :]
for j in range(width, imgGrayF.shape[0]):
    imgGrayF[:, j] = imgGrayF[:, width - 1]

imgF = imgGray
imgF = imgF.astype("float32")
coeff7 = 0.4
for i in range(height):
    for j in range(width):
        print(i, j)
        dis_filter_size = round(distImg[i][j] * coeff7)
        dist_kernel = np.ones((dis_filter_size, dis_filter_size), np.float32) / (dis_filter_size * dis_filter_size)
        try:
            if dis_filter_size != 0:
                imgF[i][j] = (imgGrayF[i:i + dis_filter_size, j:j + dis_filter_size] * dist_kernel).sum()
        except IndexError:
            continue
        except ValueError:
            print("error")

imgF = cv2.normalize(imgF, None, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow("filter", imgF)

# 8

integral = cv2.integral(imgGray)
imgInt = imgGray.copy().astype("float32")

kernelsize = 5
coeff8 = 0.4

for i in range(height):
    for j in range(width):
        print(i, j)
        dis_filter_size = round(distImg[i][j] * coeff8)

        im = max(0, i - dis_filter_size)
        jm = max(0, j - dis_filter_size)
        ip = min(height - 1, i + dis_filter_size)
        jp = min(width - 1, j + dis_filter_size)

        A = integral[ip][jp]
        C = integral[im][jm]
        B = integral[ip][jm]
        D = integral[im][jp]
        if dis_filter_size != 0:
            imgInt[i][j] = (A + C - B - D) / ((ip - im) * (jp - jm))

cv2.normalize(imgInt, imgInt, 0, 1, cv2.NORM_MINMAX)
cv2.imshow("integralF", imgInt)
cv2.waitKey(0)
cv2.destroyAllWindows()
