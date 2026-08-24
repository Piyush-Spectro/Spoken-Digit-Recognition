import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DigitAudioDataset
from model import get_resnet18_model

def train():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Paths
    BASE_DIR = "/kaggle/input/competitions/digitrecognition-ee708"
    TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
    TRAIN_AUDIO_DIR = os.path.join(BASE_DIR, "train_audio", "train_audio")

    # Load and split dataset
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_data, val_data = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=42)

    train_dataset = DigitAudioDataset(train_data, TRAIN_AUDIO_DIR, is_train=True)
    val_dataset = DigitAudioDataset(val_data, TRAIN_AUDIO_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model, loss function, optimizer, and scheduler
    model = get_resnet18_model(num_classes=10, in_channels=1, device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(DEVICE), labels.data.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        train_acc = train_correct.double() / len(train_dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        val_acc = val_correct.double() / len(val_dataset)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_digit_model.pth')
            print(f"--> Saved new best model weights (Val Acc: {val_acc:.4f})")

if __name__ == "__main__":
    train()
