import cv2
import numpy as np


img_size = (512, 512)

img_list = []

for i in range(img_size[0]):
    row = []
    for j in range(img_size[1]):
        row.append([0, 0, 255])
    img_list.append(row)


img = np.array(img_list, dtype="uint8")
print(img.shape)

cv2.imwrite(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/my_red_img.jpg",
    img,
)
# # 창을 수동 조정 가능하도록 설정
# cv2.namedWindow("Resizable Window", cv2.WINDOW_NORMAL)
# cv2.imshow("Resizable Window", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 연습 문제

img_list = []
for i in range(img_size[0]):
    row = []
    for j in range(img_size[1]):
        row.append([0, 255, 0])
    img_list.append(row)

g_img = np.array(img_list, dtype="uint8")
cv2.imwrite(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/OpenCV/img/my_green_img.jpg",
    g_img,
)
