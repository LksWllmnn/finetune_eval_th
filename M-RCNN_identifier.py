import os
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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

ID_TO_NAME = {
    0: "Background",
    1: "A-Building",
    2: "B-Building",
    3: "C-Building",
    4: "E-Building",
    5: "F-Building",
    6: "G-Building",
    7: "H-Building",
    8: "I-Building",
    9: "L-Building",
    10: "M-Building",
    11: "N-Building",
    12: "O-Building",
    13: "R-Building",
    14: "Z-Building",
}

ID_TO_COLOR = {
    0: "gray",       # Background
    1: "blue",       # A-Building
    2: "green",      # B-Building
    3: "yellow",     # C-Building
    4: "orange",     # E-Building
    5: "cyan",       # F-Building
    6: "purple",     # G-Building
    7: "pink",       # H-Building
    8: "red",        # I-Building
    9: "brown",      # L-Building
    10: "magenta",   # M-Building
    11: "lime",      # N-Building
    12: "turquoise", # O-Building
    13: "gold",      # R-Building
    14: "navy",      # Z-Building
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = len(ID_TO_NAME)
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(r"", map_location=device))
model.to(device)
model.eval()

eval_transform = get_transform(train=False)

def analyze_and_save_images(input_folder, output_folder, confidence_threshold=0.5):
    """
    analyse pictures
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        image = Image.open(image_path).convert("RGB")
        image_tensor = eval_transform(image).to(device).unsqueeze(0)

        with torch.no_grad():
            predictions = model(image_tensor)
            pred = predictions[0]

        image_display = T.ToTensor()(image).mul(255).byte()

        labels = pred["labels"]
        scores = pred["scores"]
        boxes = pred["boxes"]
        masks = (pred["masks"] > confidence_threshold).squeeze(1)

        valid_indices = [
            i for i, (label, score) in enumerate(zip(labels, scores)) 
            if label.item() != 15 and score >= confidence_threshold
        ]
        filtered_boxes = boxes[valid_indices]
        filtered_labels = labels[valid_indices]
        filtered_scores = scores[valid_indices]
        filtered_masks = masks[valid_indices]

        mask_colors = [ID_TO_COLOR[label.item()] for label in filtered_labels]

        pred_labels = [
            f"{ID_TO_NAME[label.item()]}: {score:.3f}"
            for label, score in zip(filtered_labels, filtered_scores)
        ]

        output_image = draw_segmentation_masks(image_display, filtered_masks, alpha=0.5, colors=mask_colors)
        output_image = draw_bounding_boxes(output_image, filtered_boxes.long(), labels=pred_labels, colors="red")

        output_image = output_image.permute(1, 2, 0).cpu().numpy()
        plt.imsave(output_path, output_image)
        print(f"Saved analyzed image to {output_path}")

input_folder = r""
output_folder= r""
analyze_and_save_images(input_folder, output_folder)

