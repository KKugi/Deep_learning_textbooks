import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

LR = 1e-3
EPOCH = 10
criterion = nn.CrossEntropyLoss()
new_model_train = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

input_path = (
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/project/rotated_multiple"
)

# 2. 데이터 변환 정의 (전처리 및 증강)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # [0,255] -> [0,1]
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # 이미지넷 평균
        ),  # 이미지넷 std
    ]
)

# 3. 데이터셋 로드
dataset = datasets.ImageFolder(root=input_path, transform=transform)

# 4. 데이터 분할 (train:validation = 80:20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# 5. 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

x_batch, y_batch = next(iter(train_loader))
print(type(x_batch))
print(x_batch.shape)


class CNN_deep(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.Maxpool1 = nn.MaxPool2d(2)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.Maxpool2 = nn.MaxPool2d(2)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.Maxpool3 = nn.MaxPool2d(2)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512), nn.ReLU(), nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.Maxpool1(x)
        x = self.conv_block2(x)
        x = self.Maxpool2(x)
        x = self.conv_block3(x)
        x = self.Maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


num_class = len(dataset.classes)
model = CNN_deep(num_class)
print(model)

x_batch, _ = next(iter(train_loader))
print(model(x_batch.to(DEVICE)).shape)

optimizer = optim.Adam(model.parameters(), lr=LR)


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


loss_history = Train(model, train_loader, criterion, optimizer, EPOCH)
# return loss_history
plt.plot(range(1, EPOCH + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.savefig("loss_plot.png")

save_model_path = (
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/project/model/CNN_deep.pt"
)

# 저장 코드
torch.save(model.state_dict(), save_model_path)

load_model = CNN_deep(num_class).to(DEVICE)
load_model.load_state_dict(torch.load(save_model_path, map_location=DEVICE))

Test(load_model, val_loader)


from PIL import Image

# 예측할 이미지 경로
new_image_path = (
    "C:/Users/samsung-user/Documents/Deep_learning_textbooks/project/test.jpg"
)

# 이미지 열고 전처리
img = Image.open(new_image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1, 3, 128, 128]

# 모델 예측
load_model.eval()
with torch.no_grad():
    output = load_model(img_tensor)
    pred_class_idx = output.argmax(dim=1).item()

# 클래스 이름 출력
class_names = dataset.classes
pred_class_name = class_names[pred_class_idx]

print(f"예측된 클래스: {pred_class_name} (index: {pred_class_idx})")
