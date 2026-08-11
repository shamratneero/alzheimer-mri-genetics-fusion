import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from dataset import OASISBinaryDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'checkpoints')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = OASISBinaryDataset('train', transform=transform)
    test_dataset = OASISBinaryDataset('test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # --- Freeze everything first ---
    for param in model.parameters():
        param.requires_grad = False

    # --- Unfreeze only the last block (layer4) + classifier head ---
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, 2)
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # Confirm how many parameters are actually trainable now
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    class_weights = torch.tensor([1.0, 121/71]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimize the unfrozen parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    EPOCHS = 10
    best_test_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct/total)

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f}")

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [eval]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1}/{EPOCHS} - Test Acc: {test_acc:.4f}")

        scheduler.step()

        checkpoint_path = os.path.join(OUTPUT_DIR, f'frozen_resnet18_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_path = os.path.join(OUTPUT_DIR, 'frozen_resnet18_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"New best test acc: {test_acc:.4f} - saved as best checkpoint")

    print(f"\nTraining complete. Best test accuracy: {best_test_acc:.4f}")


if __name__ == '__main__':
    main()