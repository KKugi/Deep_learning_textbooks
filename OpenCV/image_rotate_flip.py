import numpy as np
import cv2

img = cv2.imread(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/dog.JPG"
)

# cv2.imwrite(
#     "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/original_img.jpg",
#     img,
# )

# warpAffine() 함수 사용에 필요한 행렬을 만든다.
# 첫 번째 인수는 회전의 중심(여기서는 이미지의 중심을 설정)
# 두 번째 인수는 회전 각도(여기서는 180도를 설정)
# 세 번째 인수는 배율(여기서는 2배 확대로 설정)

center = (img.shape[1] / 2, img.shape[0] / 2)
mat = cv2.getRotationMatrix2D(center, 180, 2.0)


# 아핀 변환을 한다.
# 첫 번째 인수는 변환하려는 이미지
# 두 번째 인수는 위에서 생성한 행렬(mat)
# 세 번째 인수는 사이즈

my_img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))


cv2.imwrite(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/rotate_img.jpg",
    my_img,
)


# 연습 문제
# cv2.flip() 함수를 사용하여 이미지를 x축 중심으로 반전

flip_img = cv2.flip(img, 0)
cv2.imwrite(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/flip_dog_img.jpg",
    flip_img,
)
