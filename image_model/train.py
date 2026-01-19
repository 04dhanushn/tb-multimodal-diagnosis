import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler

def train_image_model(model, train_loader, criterion, optimizer, scheduler, epochs, device):
    scaler = GradScaler(device=device)

    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device).long()
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(1)
            loss_sum += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            loop.set_postfix(loss=loss_sum/total, acc=100*correct/total)

        scheduler.step()
