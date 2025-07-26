import torch
from PIL import Image
import os


def predict_image(model, image_path, transform, class_names, DEVICE):
    """
    새로운 이미지에 대해 모델 예측을 수행하고 예측된 클래스 이름을 반환합니다.

    Args:
        model (nn.Module): 예측에 사용할 학습된 모델.
        image_path (str): 예측할 이미지의 경로.
        transform (torchvision.transforms.Compose): 이미지 변환 객체.
        class_names (list): 데이터셋의 클래스 이름 리스트.
        DEVICE (str): 예측을 수행할 장치 ('cuda' 또는 'cpu').

    Returns:
        tuple: (pred_class_name, pred_class_idx) 예측된 클래스 이름과 인덱스.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred_class_idx = output.argmax(dim=1).item()

    pred_class_name = class_names[pred_class_idx]
    return pred_class_name, pred_class_idx
