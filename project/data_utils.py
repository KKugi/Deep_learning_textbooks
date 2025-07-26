import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms():
    """
    데이터 전처리 및 증강을 위한 transforms 객체를 반환합니다.
    """
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # [0,255] -> [0,1]
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet average and std
            ),
        ]
    )


def get_datasets_and_loaders(input_path, transform, batch_size=4, train_ratio=0.8):
    """
    ImageFolder 데이터셋을 로드하고, 학습 및 검증 데이터셋으로 분할하며,
    해당 데이터셋에 대한 DataLoader를 생성합니다.

    Args:
        input_path (str): 이미지 데이터셋의 루트 경로.
        transform (torchvision.transforms.Compose): 이미지 변환 객체.
        batch_size (int): DataLoader의 배치 크기.
        train_ratio (float): 학습 데이터셋의 비율.

    Returns:
        tuple: (train_loader, val_loader, dataset.classes)
    """
    dataset = datasets.ImageFolder(root=input_path, transform=transform)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes
