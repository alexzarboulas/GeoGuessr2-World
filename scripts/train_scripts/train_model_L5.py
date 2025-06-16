import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

#Custom dataset to traverse 8-level hierarchy
class HierarchicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = {}
        self._build_index()

    def _build_index(self):
        label_id = 0
        for dirpath, _, filenames in os.walk(self.root_dir):
            image_files = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue
            label = os.path.relpath(dirpath, self.root_dir)
            if label not in self.label_map:
                self.label_map[label] = label_id
                label_id += 1
            for img_file in image_files:
                self.samples.append((os.path.join(dirpath, img_file), self.label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    # --- Configuration ---
    #DATA_DIR = "../../data/hierarchy_L1"
    #DATA_DIR = "../../data/hierarchy_L3"
    DATA_DIR = "../../data/hierarchy_L5"
    #DATA_DIR = "../../data/raw_images"
    BATCH_SIZE = 64
    NUM_EPOCHS = 6
    LEARNING_RATE = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Transforms (data augmentation) for better generalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for faster training

    #Dataset loading
    dataset = HierarchicalImageDataset(DATA_DIR, transform=transform)
    num_classes = len(dataset.label_map)
    compute_top10 = num_classes >= 10
    print(f"Total images: {len(dataset)} | Total classes: {num_classes}")
    torch.save(dataset.label_map, "label_map_L5.pth")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    #Using pretrained weights on ResNet50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #Training
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

        train_acc = correct / len(train_set)
        print(f"Epoch {epoch+1} - Loss: {running_loss:.4f}, Train Accuracy (Top-1): {train_acc:.2%}")

        #Validation-
        model.eval()
        correct_top1 = 0
        correct_top10 = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                #Top 1 accuracies
                _, top1_preds = torch.topk(outputs, 1, dim=1)
                correct_top1 += (top1_preds.squeeze() == labels).sum().item()

                #Top 10 accuracies (if applicable)
                if compute_top10:
                    _, top10_preds = torch.topk(outputs, 10, dim=1)
                    for label, topk in zip(labels, top10_preds):
                        correct_top10 += int(label in topk)

                total += labels.size(0)

        print(f"Validation Accuracy (Top-1): {correct_top1 / total:.2%}")
        if compute_top10: print(f"Validation Accuracy (Top-10): {correct_top10 / total:.2%}")

        #Checkpoint saving (no longer necessary)
        """
        checkpoint_path = f"../../models/checkpoints/epoch_{epoch+1}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': running_loss,
            'train_accuracy': train_acc,
            'val_accuracy_top1': correct_top1 / total,
            'val_accuracy_top10': correct_top10 / total,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        """


    #Save model-
    os.makedirs("../../models", exist_ok=True)
    torch.save(model.state_dict(), "../../models/hierarchy_L5.pth")
    print("Training complete, model saved to ../../models/hierarchy_L5.pth")


#Windows multiprocessing fix
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
