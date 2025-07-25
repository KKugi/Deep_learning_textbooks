import numpy as np
import cv2

img = cv2.imread(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/source/cleansing_data/sample.JPG"
)
size = img.shape

# 트리밍: 이미지의 일부를 잘라내는 작업
# 리사이즈는 이미지의 크기를 변경(확대, 축소)하는 작업

trim_img = img[: size[0] // 2, : size[1] // 3]

cv2.imwrite(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/trimming.jpg",
    trim_img,
)

resize_img = cv2.resize(trim_img, (trim_img.shape[1] * 2, trim_img.shape[0] * 2))

cv2.imwrite(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/resize.jpg",
    resize_img,
)

img1 = cv2.imread(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/trimming.jpg"
)
img2 = cv2.imread(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/resize.jpg"
)
# print("trimming.jpg size:", img1.shape)
# print("resize.jpg size:", img2.shape)

# 연습 문제

# sample.jpg의 폭과 높이를 각각 1/3로 리사이즈

practice_img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))

cv2.imwrite(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/Practice_img.jpg",
    practice_img,
)

img3 = cv2.imread(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/Practice_img.jpg"
)
print("Practice_img size:", img3.shape)
