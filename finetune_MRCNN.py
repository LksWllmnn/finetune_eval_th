import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from references.engine import train_one_epoch  # Importiere deine Trainingsfunktion
from final_tools.dataset_class import ImageTitleDatasetMRCNN, COLOR_TO_ID
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# Load the pre-trained Mask R-CNN model
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def evaluate_with_iou(model, data_loader, device):
    model.eval()
    total_iou = 0
    count = 0

    for images, targets in tqdm(data_loader, desc="Evaluating IoU", ncols=100):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)
        
        for output, target in zip(outputs, targets):
            pred_masks = output['masks'] > 0.5
            target_masks = target['masks']
            
            for pred_mask, target_mask in zip(pred_masks, target_masks):
                total_iou += calculate_iou(pred_mask.cpu().numpy(), target_mask.cpu().numpy())
                count += 1

    mean_iou = total_iou / count if count > 0 else 0
    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou

# IoU
def calculate_iou(pred_mask, target_mask):
    intersection = (pred_mask & target_mask).sum()
    union = (pred_mask | target_mask).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def calculate_segmentation_accuracy(model, data_loader, device):
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                true_mask = target['masks'].cpu().numpy() > 0.5
                pred_mask = output['masks'].cpu().detach().numpy() > 0.5

                correct_pixels += (pred_mask == true_mask).sum()
                total_pixels += true_mask.size

    return correct_pixels / total_pixels

def train_and_evaluate(model, train_loader, val_loader, device, num_epochs, lr, save_path=""):
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_losses, val_losses, iou_scores, accuracies = [], [], [], []
    best_iou = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        train_losses.append(train_loss)
        mean_iou = evaluate_with_iou(model, val_loader, device)
        iou_scores.append(mean_iou / len(val_loader) if val_loader else 0)
        
        # Update Best Model
        if iou_scores[-1] > best_iou:
            best_iou = iou_scores[-1]
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with IoU: {best_iou:.4f}")

        lr_scheduler.step()

    return train_losses, val_losses, iou_scores, accuracies

# Plotting-Funktion
def save_plots_and_logs(train_losses, val_losses, iou_scores, accuracies, log_path="big-surround_mrcnn_training_log.txt", plot_dir="plots"):
    import os
    os.makedirs(plot_dir, exist_ok=True)

    # IoU Plot
    plt.figure(figsize=(10, 5))
    plt.plot(iou_scores, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU per Epoch")
    plt.legend()
    plt.savefig(f"{plot_dir}/big-surround_iou_plot.png")
    plt.close()

    # Logs speichern
    with open(log_path, "w") as log_file:
        log_file.write("IoU\t\n")
        for epoch in range(len(train_losses)):
            log_file.write(f"{iou_scores[epoch]:.4f}\n")
    print(f"Logs und Plots gespeichert in {plot_dir}")

# Hauptprogramm
if __name__ == "__main__":
    # Daten laden
    root_dir = r""
    
    dataset = ImageTitleDatasetMRCNN(root_dir=root_dir, filter=True, transforms=get_transform(train=True))
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))  # 80% ftrain
    val_size = len(dataset) - train_size  # 20% val

    # random split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"trainingsdataset {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

    # Modell instatiate
    num_classes = len(COLOR_TO_ID) 
    model = get_model_instance_segmentation(num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_losses, val_losses, iou_scores, accuracies = train_and_evaluate(
        model, train_loader, val_loader, device, num_epochs=10, lr=0.005, save_path=""
    )

    save_plots_and_logs(train_losses, val_losses, iou_scores, accuracies)
