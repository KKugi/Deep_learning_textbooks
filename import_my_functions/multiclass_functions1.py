import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def Train(model, train_DL, criterion, optimizer, EPOCH):
    loss_history = []
    NoT = len(train_DL.dataset)  # The number of training data

    model.train()  # train model로 전환
    ch = 0
    for ep in range(EPOCH):
        rloss = 0  # running loss: 배치 마다 loss 누적해서 전체 데이터에 대해서의 loss값 구하기, # 매 에폭마다 초기화
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # inference
            y_hat = model(x_batch)

            # loss
            loss = criterion(y_hat, y_batch)

            # update
            optimizer.zero_grad()  # gradient 누적을 막기 위한 초기화
            loss.backward()  # backpropagation
            optimizer.step()  # weight update

            # loss 누적 (배치 평균 loss × 배치 크기)
            # loss.item() : 현재 배치에 대한 CrossEntropy 평균값
            loss_b = (
                loss.item() * x_batch.shape[0]
            )  # 배치 평균 loss에 실제 배치 크기를 곱해 전체 loss로 변환
            # 총 배치 개수 = ceil(60000 / 32) = ceil(1875) = 1875 (한 에폭(epoch) 동안 반복문이 도는 횟수)
            rloss += loss_b  # 현재 에폭(epoch) 동안의 누적 loss 합계 (모든 배치의 loss_b 누적)

        # print loss
        loss_e = (
            rloss / NoT
        )  # 현재 에폭(epoch) 전체 데이터에 대한 평균 loss (총합 / 전체 샘플 수), # 위 그림에서 빨간색으로 된 1/50 부분을 나타냄
        loss_history += [loss_e]
        print(f"Epoch: {ep+1}, train loss: {loss_e:.4f}")
        print("-" * 20)

    return loss_history


def Test(model, test_DL):
    model.eval()  # test mode로 전환
    with torch.no_grad():
        rcorrect = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # inference
            y_hat = model(x_batch)
            # print(y_hat)
            # print(y_hat.shape)
            # break

            # corrects accumulation
            pred = y_hat.argmax(dim=1)
            # print(y_hat.shape)
            # print(pred.shape)
            # break

            corrects_b = torch.sum(
                pred == y_batch
            ).item()  # torch.eq(pred, y_batch).sum().item()
            # print(pred) # 예측
            # print(y_batch) # 정답
            # print(pred == y_batch)
            # print(torch.sum(pred == y_batch).item())
            # break
            rcorrect += corrects_b

        accuracy_e = rcorrect / len(test_DL.dataset) * 100

    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({accuracy_e:.1f} %)")
    return round(accuracy_e, 1)


def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))  # 하나의 배치만 가져옴
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)
        # print(pred)

    x_batch = x_batch.to("cpu")

    """
    imshow()는 (Height, Width, Channel) 형식인 (H, W, C)을 요구

    x_batch[0].shape = (1, 28, 28)  # (채널 수=1, 높이, 너비)
    흑백 이미지	shape = (1, H, W) → .squeeze() → (H, W)로 변경
    RGB 이미지	shape = (3, H, W) → .permute(1, 2, 0) → (H, W, 3)로 변경
    """
    # print(x_batch[0].shape) # torch.Size([1, 28, 28])
    # print(x_batch[0].permute(1,2,0).shape) # torch.Size([28, 28, 1])
    # print(x_batch[0].permute(1,2,0).squeeze().shape) # torch.Size([28, 28])

    plt.figure(figsize=(8, 4))
    for idx in range(6):
        plt.subplot(2, 3, idx + 1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1, 2, 0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(
            f"{pred_class} ({true_class})",
            color="b" if pred_class == true_class else "r",
        )


# 파라미터 수 구하기
def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num


def get_conf(model, test_DL):
    N = len(test_DL.dataset.classes)
    model.eval()
    with torch.no_grad():
        confusion = torch.zeros(N, N)
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # inference
            y_hat = model(x_batch)

            # accuracy
            pred = y_hat.argmax(dim=1)
            """
            1. N: 클래스 수 (예: 10)
            2. y_batch.cpu(): 실제 레이블을 CPU 텐서로 변환
            3. pred.cpu(): 예측 레이블을 CPU 텐서로 변환
            4. N * y_batch.cpu() + pred.cpu():
            - (실제 클래스, 예측 클래스) 쌍을 단일 인덱스로 변환 (예: 실제=2, 예측=1 → 21)
            5. torch.bincount(..., minlength=N**2):
            - 각 인덱스 별 등장 횟수를 세어, 길이 N^2인 1차원 텐서 생성(길이 100인 1차원 배열)
            - 예를 들어, 인덱스 12는 혼동 행렬에서 1행(실제 클래스 1), 2열(예측 클래스 2)을 의미
            - minlength 옵션으로 특정 조합이 없더라도 크기를 N^2로 고정
            6. .reshape(N, N):
            - 1차원 텐서를 N×N 크기의 혼동 행렬 형태로 변환
            7. confusion += ...:
            - 이전 배치까지 계산된 혼동 행렬에 이번 배치 결과를 누적 합산

            결과적으로 이 코드는 배치 단위로 나온 (실제, 예측) 쌍의 개수를 모아서
            전체 데이터에 대한 혼동 행렬을 누적 계산하는 역할을 한다
            """
            confusion += torch.bincount(
                N * y_batch.cpu() + pred.cpu(), minlength=N**2
            ).reshape(N, N)
            # confusion matrix는 무조건 10X10이 되어야 하는데 만약 마지막 label에 대해서 예측을 모두 실패하면 100개보다 작아질 수 있기 때문에 minlength를 설정
    confusion = confusion.numpy()
    return confusion


def plot_confusion_matrix(confusion, classes=None):
    N = confusion.shape[0]
    print(N)

    # np.trace()은 행렬의 대각 원소들의 합을 계산
    accuracy = np.trace(confusion) / np.sum(confusion) * 100
    print(accuracy)

    # confusion = confusion/np.sum(confusion, axis=1)
    plt.figure(figsize=(10, 7))
    plt.imshow(confusion, cmap="Blues")
    plt.title("confusion matrix")
    plt.colorbar()

    for i in range(N):
        for j in range(N):
            plt.text(
                j,
                i,
                round(confusion[i, j]),
                horizontalalignment="center",
                fontsize=10,
                color="white" if confusion[i, j] > np.max(confusion) / 1.5 else "black",
            )

    if classes is not None:
        plt.xticks(range(N), classes)
        plt.yticks(range(N), classes)
    else:
        plt.xticks(range(N))
        plt.yticks(range(N))

    plt.xlabel(f"Predicted label \n accuracy = {accuracy:.1f} %")
    plt.ylabel("True label")
