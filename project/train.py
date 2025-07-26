import torch


def Train(model, train_DL, criterion, optimizer, EPOCH, DEVICE):
    """
    주어진 모델을 학습 데이터로 훈련합니다.

    Args:
        model (nn.Module): 학습할 모델.
        train_DL (DataLoader): 학습 데이터 로더.
        criterion (nn.Module): 손실 함수.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        EPOCH (int): 학습할 에폭(epoch) 수.
        DEVICE (str): 학습을 수행할 장치 ('cuda' 또는 'cpu').

    Returns:
        list: 각 에폭(epoch)의 평균 학습 손실 리스트.
    """
    loss_history = []
    NoT = len(train_DL.dataset)  # The number of training data

    model.train()  # Set model to training mode
    for ep in range(EPOCH):
        rloss = 0  # Running loss: accumulate loss per batch
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Inference
            y_hat = model(x_batch)

            # Loss
            loss = criterion(y_hat, y_batch)

            # Update
            optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Weight update

            # Accumulate loss (batch average loss * batch size)
            loss_b = loss.item() * x_batch.shape[0]
            rloss += loss_b  # Accumulate loss for the current epoch

        # Print epoch loss
        loss_e = rloss / NoT  # Average loss for the entire epoch
        loss_history.append(loss_e)
        print(f"Epoch: {ep+1}, train loss: {loss_e:.4f}")
        print("-" * 20)

    return loss_history
