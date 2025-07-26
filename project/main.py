import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from model import CNN_deep
from data_utils import get_transforms, get_datasets_and_loaders
from train import Train
from evaluate import Test
from predict import predict_image

# --- Configuration ---
LR = 1e-3
EPOCH = 10
BATCH_SIZE = 4
TRAIN_RATIO = 0.8
NEW_MODEL_TRAIN = False  # True to train a new model, False to load a saved model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Define paths (adjust as needed)
BASE_PATH = "C:/Users/samsung-user/Documents/Deep_learning_textbooks/project"
INPUT_PATH = os.path.join(BASE_PATH, "rotated_multiple")
SAVE_MODEL_DIR = os.path.join(BASE_PATH, "model")
SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR, "CNN_deep.pt")
LOSS_PLOT_PATH = "loss_plot.png"
TEST_IMAGE_PATH = os.path.join(BASE_PATH, "test.jpg")

# Ensure model save directory exists
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# --- Data Preparation ---
transform = get_transforms()
train_loader, val_loader, class_names = get_datasets_and_loaders(
    INPUT_PATH, transform, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO
)

num_class = len(class_names)
print(f"Number of classes: {num_class}")
print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(val_loader.dataset)}")


# --- Model Initialization ---
model = CNN_deep(num_class).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\n--- Model Summary ---")
print(model)
print("---------------------\n")

# --- Training or Loading Model ---
if NEW_MODEL_TRAIN:
    print("--- Starting Training ---")
    loss_history = Train(model, train_loader, criterion, optimizer, EPOCH, DEVICE)

    # Plotting and saving loss
    plt.plot(range(1, EPOCH + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()  # Close the plot to free memory
    print(f"Loss plot saved to {LOSS_PLOT_PATH}")

    # Save the trained model
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")
else:
    print(f"--- Loading pre-trained model from {SAVE_MODEL_PATH} ---")
    if not os.path.exists(SAVE_MODEL_PATH):
        print(
            f"Error: Model file not found at {SAVE_MODEL_PATH}. Set NEW_MODEL_TRAIN to True to train a new model."
        )
        exit()  # Exit if model not found and not training new
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
    print("Model loaded successfully.")

# --- Evaluation ---
print("\n--- Starting Evaluation ---")
val_accuracy = Test(model, val_loader, DEVICE)
print(f"Validation Accuracy: {val_accuracy:.1f} %")

# --- Prediction on a New Image ---
print("\n--- Starting Prediction on a New Image ---")
try:
    pred_class_name, pred_class_idx = predict_image(
        model, TEST_IMAGE_PATH, transform, class_names, DEVICE
    )
    print(
        f"Predicted class for '{os.path.basename(TEST_IMAGE_PATH)}': {pred_class_name} (index: {pred_class_idx})"
    )
except FileNotFoundError as e:
    print(f"Prediction error: {e}")
