
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Map original class folders to binary labels
CLASS_TO_BINARY = {
    'NonDemented': 0,
    'VeryMildDemented': 1,
    'MildDemented': 1,
    'ModerateDemented': 1,
}

class OASISBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []  # list of (filepath, label)
        self.transform = transform

        for class_name, binary_label in CLASS_TO_BINARY.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} not found, skipping")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), binary_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = OASISBinaryDataset('train', transform=transform)
    test_dataset = OASISBinaryDataset('test', transform=transform)

    print(f"Train images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # quick check: load one sample
    img, label = train_dataset[0]
    print(f"Sample image shape: {img.shape}, label: {label}")