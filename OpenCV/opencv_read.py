import numpy as np
import cv2

img = cv2.imread(
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/source/cleansing_data/sample.JPG"
)

# 창을 수동 조정 가능하도록 설정
cv2.namedWindow("Resizable Window", cv2.WINDOW_NORMAL)
cv2.imshow("Resizable Window", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

