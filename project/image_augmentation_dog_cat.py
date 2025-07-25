import numpy as np
import cv2
import os
import re

input_dir = "C:/Users/samsung-user/Documents/Deep_learning_textbooks/project/original"
output_dir = (
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/project/rotated_multiple"
)
os.makedirs(output_dir, exist_ok=True)


# 폴더 내 jpg, jpeg, png 파일만 처리
valid_exts = (".jpg", ".jpeg", ".png")
angles = [90, 180, 270, 360]

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(valid_exts):
        continue

    label = re.sub(r"\d+\.(jpg|jpeg|png)$", "", filename, flags=re.IGNORECASE)
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 로드 실패: {filename}")
        continue
    current_img = img.copy()

    # 라벨별 폴더 경로 생성
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    for i in range(1, 5):  # 90도, 180도, 270도, 360도 (원본)
        center = (current_img.shape[1] / 2, current_img.shape[0] / 2)
        mat = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated_img = cv2.warpAffine(
            current_img, mat, (current_img.shape[1], current_img.shape[0])
        )

        # 저장 경로에 라벨 폴더 포함
        save_path = os.path.join(
            label_dir, f"{os.path.splitext(filename)[0]}_rotated_{90*i}.jpg"
        )
        cv2.imwrite(save_path, rotated_img)
        print(f"저장 완료: {save_path}")

        current_img = rotated_img
