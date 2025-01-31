import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def save_anns(anns, image, output_path, mask_color=[255, 0, 0], transparency=90):
    if len(anns) == 0:
        return

    image_with_mask = image.copy()
    for ann in anns:
        m = ann['segmentation']
        overlay = np.zeros((*m.shape, 4), dtype=np.uint8)
        overlay[m] = np.concatenate([mask_color, [transparency]])
        alpha_mask = overlay[:, :, 3] / 255.0
        for c in range(3):
            image_with_mask[:, :, c] = (1 - alpha_mask) * image_with_mask[:, :, c] + alpha_mask * overlay[:, :, c]

    image_rgb = cv2.cvtColor(image_with_mask, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

def save_mask_only(anns, image_shape, output_path):
    mask_image = np.zeros(image_shape[:2], dtype=np.uint8)
    if len(anns) > 0:
        for ann in anns:
            mask_image[ann['segmentation']] = 255

    cv2.imwrite(output_path, mask_image)

def classify_images(model, label_mapping, images, target_classes, confidence_threshold=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    batch_size = 8
    valid_images_by_class = {target: [] for target in target_classes}
    valid_mask_indices_by_class = {target: [] for target in target_classes}

    target_class_indices = {}
    for target in target_classes:
        idx = next((i for i, name in label_mapping.items() if name == target), None)
        if idx is not None:
            target_class_indices[target] = idx
        else:
            print(f"Warning: Target class '{target}' not found in label mapping. Skipping this class.")
    
    target_classes = [cls for cls in target_classes if cls in target_class_indices]

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_tensors = torch.stack([preprocess(Image.fromarray(img)).to(device) for img in batch])

        with torch.no_grad():
            outputs = model(batch_tensors)
            probs = F.softmax(outputs, dim=1)

        for idx, prob in enumerate(probs):
            for target, target_idx in target_class_indices.items():
                if prob[target_idx].item() * 100 >= confidence_threshold:
                    valid_images_by_class[target].append(batch[idx])
                    valid_mask_indices_by_class[target].append(i + idx)

        del batch_tensors, outputs, probs
        torch.cuda.empty_cache()

    return valid_images_by_class, valid_mask_indices_by_class

def process_image(image_path, output_folder_base, mask_generator, model, label_mapping, target_classes, confidence_threshold):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}, skipping...")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    mask_images = []
    for mask in masks:
        m = mask['segmentation'].astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=m)
        mask_images.append(masked_image)

    valid_images_by_class, valid_mask_indices_by_class = classify_images(
        model=model, label_mapping=label_mapping, images=mask_images,
        target_classes=target_classes, confidence_threshold=confidence_threshold
    )

    for target_class in target_classes:
        qualitativ_folder = os.path.join(output_folder_base, "qualitativ", f"{target_class}")
        mask_only_folder = os.path.join(output_folder_base, "just-mask", f"{target_class}")
        os.makedirs(qualitativ_folder, exist_ok=True)
        os.makedirs(mask_only_folder, exist_ok=True)

        combined_output_path = os.path.join(qualitativ_folder, os.path.basename(image_path).replace(".jpg", "_m.jpg"))
        mask_only_output_path = os.path.join(mask_only_folder, os.path.basename(image_path).replace(".jpg", ".jpg"))

        valid_images = valid_images_by_class[target_class]
        valid_mask_indices = valid_mask_indices_by_class[target_class]

        if valid_images:
            valid_masks = [masks[i] for i in valid_mask_indices]
            save_anns(valid_masks, image, combined_output_path)
            save_mask_only(valid_masks, image.shape, mask_only_output_path)
            print(f"Saved marked image for '{target_class}' to {combined_output_path}")
            print(f"Saved mask-only image for '{target_class}' to {mask_only_output_path}")
        else:
            save_mask_only([], image.shape, mask_only_output_path)
            cv2.imwrite(combined_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"No valid masks for class '{target_class}' in image: {image_path}")

    del masks, mask_images, valid_images_by_class, valid_mask_indices_by_class
    torch.cuda.empty_cache()

def process_folder(image_folder, output_folder_base, mask_generator, model, label_mapping, target_classes, confidence_threshold=10):
    """Verarbeitet einen gesamten Ordner."""
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for image_path in image_paths:
        process_image(image_path, output_folder_base, mask_generator, model, label_mapping, target_classes, confidence_threshold)

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint="chkpts\\sam_vit_h_4b8939.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

model = models.resnet50(pretrained=False)
num_classes = 14
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(r"", map_location=device, weights_only=True))
model.to(device)
model.eval()

label_mapping = {
    0: "A-Building", 1: "B-Building", 2: "C-Building", 3: "E-Building", 4: "F-Building", 5: "G-Building",
    6: "H-Building", 7: "I-Building", 8: "L-Building", 9: "M-Building", 10: "N-Building", 11: "O-Building",
    12: "R-Building", 13: "Z-Building"
}

if __name__ == "__main__":
    image_folder = r""
    output_folder_base = r""
    target_classes = ["A-Building", 
                      "B-Building",
                      "C-Building",
                      "E-Building",
                      "F-Building",
                      "G-Building", 
                      "H-Building",
                      "L-Building",
                      "M-Building",
                      "N-Building",
                      "O-Building", 
                      "I-Building", 
                      "R-Building", 
                      "Z-Building"]
    process_folder(image_folder, output_folder_base, mask_generator, model, label_mapping, target_classes, confidence_threshold=30)
