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
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    class_weights = torch.tensor([1.0, 121/71]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 3

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
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
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f}")

        checkpoint_path = os.path.join(OUTPUT_DIR, f'baseline_resnet18_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    final_path = os.path.join(OUTPUT_DIR, 'baseline_resnet18_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


if __name__ == '__main__':
    main()