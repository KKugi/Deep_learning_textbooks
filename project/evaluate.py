import torch


def Test(model, test_DL, DEVICE):
    """
    주어진 모델을 테스트 데이터로 평가하고 정확도를 반환합니다.

    Args:
        model (nn.Module): 평가할 모델.
        test_DL (DataLoader): 테스트 데이터 로더.
        DEVICE (str): 평가를 수행할 장치 ('cuda' 또는 'cpu').

    Returns:
        float: 테스트 정확도 (%).
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        rcorrect = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Inference
            y_hat = model(x_batch)

            # Corrects accumulation
            pred = y_hat.argmax(dim=1)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b

        accuracy_e = rcorrect / len(test_DL.dataset) * 100

    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({accuracy_e:.1f} %)")
    return round(accuracy_e, 1)
