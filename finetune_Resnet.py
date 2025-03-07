import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from dataset_class import ImageTitleDatasetResNet

from tqdm import tqdm

def show_images(dataloader, dataset):
    images, labels = next(iter(dataloader))
    images = images.permute(0, 2, 3, 1)
    text_labels = dataset.label_encoder.inverse_transform(labels.numpy())
    plt.figure(figsize=(12, 6))
    for i in range(len(images)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(text_labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

from tqdm import tqdm 

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f'{phase.capitalize()} Phase Progress',
                leave=True,
                unit='batch'
            )

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                current_loss = running_loss / (len(progress_bar) * inputs.size(0))
                progress_bar.set_postfix(
                    loss=current_loss,
                    acc=(running_corrects.double() / (len(progress_bar) * inputs.size(0))).item()
                )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    loss_plot_path = os.path.join("plots/", "big-surround_resnet_loss_per_epoch.png")
    plt.savefig(loss_plot_path)
    plt.close()

    return model


def main():
    # Dataset Path
    image_path = r""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageTitleDatasetResNet(root_dir=image_path, filter=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False)
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    show_images(dataloaders['train'], train_dataset.dataset)

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    label_mapping = dataset.label_encoder.classes_
    model_ft.fc = nn.Linear(num_ftrs, len(label_mapping))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    model_ft = train_model(
        model=model_ft,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer_ft,
        device=device,
        num_epochs=15
    )

    torch.save(model_ft.state_dict(), "")

if __name__ == "__main__":
    main()
