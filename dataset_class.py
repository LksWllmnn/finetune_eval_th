import os
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import Compose, Resize, ToTensor
import clip
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

BUILDING_COLORS = {
    "A-Building": (0, 0, 255, 255),
    "B-Building": (0, 255, 0, 255),
    "C-Building": (255, 0, 0, 255),
    "E-Building": (255, 255, 255, 255),
    "F-Building": (255, 235, 4, 255),
    "G-Building": (128, 128, 128, 255),
    "H-Building": (255, 32, 98, 255),
    "I-Building": (255, 25, 171, 255),
    "M-Building": (255, 73, 101, 255),
    "N-Building": (145, 255, 114, 255),
    "L-Building": (93, 71, 255, 255),
    "O-Building": (153, 168, 255, 255),
    "R-Building": (64, 0, 75, 255),
    "Z-Building": (18, 178, 0, 255),
}

COLOR_TO_ID = {
    (0, 0, 0, 255): 0,  # Background
    (0, 0, 255, 255): 1,    # A-Building
    (0, 255, 0, 255): 2,    # B-Building
    (255, 0, 0, 255): 3,    # C-Building
    (255, 255, 255, 255): 4,  # E-Building
    (255, 235, 4, 255): 5,  # F-Building
    (128, 128, 128, 255): 6,  # G-Building
    (255, 32, 98, 255): 7,  # H-Building
    (255, 25, 171, 255): 8,  # I-Building
    (255, 73, 101, 255): 9,  # L-Building
    (145, 255, 114, 255): 10,  # M-Building
    (93, 71, 255, 255): 11,  # N-Building
    (153, 168, 255, 255): 12,  # O-Building
    (64, 0, 75, 255): 13,  # R-Building
    (18, 178, 0, 255): 14,  # Z-Building
    #(255, 169, 0, 255): 15,  # Other
}

BACKGROUND_COLOR = (0, 0, 0, 255)

# Preprocessing for images
preprocess = Compose([
    Resize((224, 224)),  # Resize to CLIP model input size
    ToTensor()           # Convert to tensor
])

class ImageTitleDataset:
    def __init__(self, root_dir, device='cuda', transform=None, filter=False):
        self.image_paths = []
        self.labels = []
        self.valid_images = []
        self.invalid_images = []
        self.filter = filter
        self.transform = transform
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Store count of valid images per building
        self.building_counts = Counter()

        # Iterate over building folders
        building_folders = [b for b in BUILDING_COLORS.keys() if os.path.isdir(os.path.join(root_dir, b))]

        for building in tqdm(building_folders, desc="Processing buildings"):
            color = torch.tensor(BUILDING_COLORS[building], device=self.device, dtype=torch.uint8)
            building_dir = os.path.join(root_dir, building)

            building_valid_images = []

            for filename in tqdm(os.listdir(building_dir), desc=f"Filtering images in {building}", leave=False):
                if filename.endswith("camera.png"):
                    base_name = filename.replace(".camera.png", "")
                    seg_path = os.path.join(building_dir, f"{base_name}.camera.semantic segmentation.png")

                    if not os.path.exists(seg_path):
                        continue

                    # Check if the semantic segmentation image matches the conditions
                    full_image_path = os.path.join(building_dir, filename)
                    if self.filter:
                        if self.is_valid_image(seg_path, color):
                            building_valid_images.append(full_image_path)
                            self.building_counts[building] += 1
                        else:
                            self.invalid_images.append(full_image_path)
                    else:
                        building_valid_images.append(full_image_path)
                        self.building_counts[building] += 1


            # Limit to 500 valid images per building
            selected_images = building_valid_images[:500]
            self.image_paths.extend(selected_images)
            self.labels.extend([building] * len(selected_images))
            self.valid_images.extend(selected_images)

    def is_valid_image(self, seg_path, target_color):
        """Check if the segmentation image is valid for the target building."""
        seg_image = Image.open(seg_path).convert("RGBA")
        seg_array = torch.tensor(np.array(seg_image), device=self.device, dtype=torch.uint8)

        # Count pixel occurrences, excluding the background color
        unique, counts = torch.unique(seg_array.view(-1, 4), dim=0, return_counts=True)
        color_counts = {tuple(color.tolist()): count.item() for color, count in zip(unique, counts) if tuple(color.tolist()) != BACKGROUND_COLOR}

        # Exclude if target color is not present
        if tuple(target_color.tolist()) not in color_counts:
            return False

        return True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_paths[idx]).convert("RGB"))
        label = clip.tokenize(self.labels[idx])[0]  # Tokenize the label
        return image, label
    
class ImageTitleDatasetMRCNN:
    def __init__(self, root_dir, transforms=None,device='cuda', filter=False, num_images_per_building=500):
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.masks = []
        self.imgs = []
        self.invalid_images = []
        self.filter = filter
        self.transforms = transforms
        self.num_images_per_building = num_images_per_building
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Store count of valid images per building
        self.building_counts = Counter()

        # Iterate over building folders
        building_folders = [b for b in BUILDING_COLORS.keys() if os.path.isdir(os.path.join(root_dir, b))]

        for building in tqdm(building_folders, desc="Processing buildings"):
            color = torch.tensor(BUILDING_COLORS[building], device=self.device, dtype=torch.uint8)
            building_dir = os.path.join(root_dir, building)

            building_valid_images = []
            building_valid_masks = []

            for filename in tqdm(os.listdir(building_dir), desc=f"Filtering images in {building}", leave=False):
                if filename.endswith("camera.png"):
                    base_name = filename.replace(".camera.png", "")
                    seg_path = os.path.join(building_dir, f"{base_name}.camera.semantic segmentation.png")

                    if os.path.exists(seg_path):
                        full_image_path = os.path.join(building_dir, filename)
                        if self.filter:
                            if self.is_valid_image(seg_path, color):
                                building_valid_images.append(full_image_path)
                                building_valid_masks.append(seg_path)
                                self.building_counts[building] += 1
                            else:
                                self.invalid_images.append(full_image_path)
                        else:
                            building_valid_images.append(full_image_path)
                            building_valid_masks.append(seg_path)
                            self.building_counts[building] += 1
                    else:
                        self.invalid_images.append(os.path.join(building_dir, filename))

            # Limit to num_images_per_building valid images per building
            self.imgs.extend(building_valid_images[:self.num_images_per_building])
            self.masks.extend(building_valid_masks[:self.num_images_per_building])

        num_valid_images = min(len(self.imgs), len(self.masks))
        selected_indices = list(range(num_valid_images))[:self.num_images_per_building]
        print(f"Building: {building}, Total images: {len(self.imgs)}, Total masks: {len(self.masks)}")
        if len(self.imgs) != len(self.masks):
            print(f"Warning: Mismatch detected for building {building}")

        selected_images = [self.imgs[i] for i in selected_indices]
        selected_masks = [self.masks[i] for i in selected_indices]

        self.image_paths.extend(selected_images)
        self.mask_paths.extend(selected_masks)
        self.labels.extend([building] * len(selected_images))
        

    def is_valid_image(self, seg_path, target_color):
        """Check if the segmentation image is valid for the target building."""
        seg_image = Image.open(seg_path).convert("RGBA")
        seg_array = torch.tensor(np.array(seg_image), device=self.device, dtype=torch.uint8)

        # Count pixel occurrences, excluding the background color
        unique, counts = torch.unique(seg_array.view(-1, 4), dim=0, return_counts=True)
        color_counts = {tuple(color.tolist()): count.item() for color, count in zip(unique, counts) if tuple(color.tolist()) != BACKGROUND_COLOR}

        # Exclude if target color is not present
        if tuple(target_color.tolist()) not in color_counts:
            return False

        return True

    def __len__(self):
        return len(self.imgs)

    #     return img, target
    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Convert mask to class IDs
        mask_np = np.array(mask)
        instance_mask = np.zeros(mask_np.shape[:2], dtype=np.int64)
        for color, class_id in COLOR_TO_ID.items():
            # Extract RGB
            if mask_np.shape[-1] == 4:
                mask_np = mask_np[:, :, :3]

            color = np.array(color[:3])  # just use rgb
            instance_mask[np.all(mask_np == color, axis=-1)] = class_id

        # Get unique class IDs and generate binary masks
        obj_ids = np.unique(instance_mask)
        obj_ids = obj_ids[obj_ids > 0]  # Exclude background (0)

        if len(obj_ids) == 0:
            # No objects: set empty bounding boxes and labels
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, instance_mask.shape[0], instance_mask.shape[1]), dtype=torch.uint8)
        else:
            # Normal processing
            masks = instance_mask == obj_ids[:, None, None]
            boxes = []
            for i in range(len(obj_ids)):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(obj_ids, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Filter invalid bounding boxes
        if len(boxes) > 0:
            # Ensure positive height and width
            valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_indices]
            labels = labels[valid_indices]
            masks = masks[valid_indices]

        image_id = torch.tensor([idx])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([0.0])  # Set area to 0 if no boxes exist
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


class ImageTitleDatasetResNet():
    def __init__(self, root_dir, transforms=None, device='cuda', filter=False, num_images_per_building=500):
        self.image_paths = []
        self.labels = []
        self.invalid_images = []
        self.filter = filter
        self.transforms = transforms
        self.num_images_per_building = num_images_per_building
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.building_counts = Counter()

        building_folders = [b for b in BUILDING_COLORS.keys() if os.path.isdir(os.path.join(root_dir, b))]

        for building in tqdm(building_folders, desc="Processing buildings"):
            color = torch.tensor(BUILDING_COLORS[building], device=self.device, dtype=torch.uint8)
            building_dir = os.path.join(root_dir, building)

            for filename in tqdm(os.listdir(building_dir), desc=f"Filtering images in {building}", leave=False):
                if filename.endswith("camera.png"):
                    base_name = filename.replace(".camera.png", "")
                    seg_path = os.path.join(building_dir, f"{base_name}.camera.semantic segmentation.png")
                    full_image_path = os.path.join(building_dir, filename)

                    if os.path.exists(seg_path) and (not self.filter or self.is_valid_image(seg_path, color)):
                        if self.building_counts[building] < self.num_images_per_building:
                            self.image_paths.append(full_image_path)
                            self.labels.append(building)
                            self.building_counts[building] += 1
                    elif self.filter:
                        self.invalid_images.append(full_image_path)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def is_valid_image(self, seg_path, target_color):
        seg_image = Image.open(seg_path).convert("RGBA")
        seg_array = torch.tensor(np.array(seg_image), device=self.device, dtype=torch.uint8)
        unique, counts = torch.unique(seg_array.view(-1, 4), dim=0, return_counts=True)
        color_counts = {tuple(color.tolist()): count.item() for color, count in zip(unique, counts) 
                        if tuple(color.tolist()) != BACKGROUND_COLOR}
        return tuple(target_color.tolist()) in color_counts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        
        # pic to tensor
        if self.transforms:
            image = self.transforms(image)
        else:
            image = ToTensor()(image)

        return image, label

if __name__ == "__main__":
    root_directory = r""  # Replace with your root directory
    dataset = ImageTitleDataset(root_dir=root_directory, filter_images=True)

    # Print dataset statistics
    print("Dataset statistics:")
    for building, count in dataset.building_counts.items():
        print(f"{building}: {count} images")

    print(f"Total images: {len(dataset)}")

    # Plot 20 valid and 20 invalid images
    def plot_images(image_paths, title, num_images=20):
        plt.figure(figsize=(15, 10))
        for i, img_path in enumerate(image_paths[:num_images]):
            img = Image.open(img_path)
            plt.subplot(4, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path))
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\nDisplaying valid images:")
    plot_images(dataset.valid_images, "Valid Images")

    print("\nDisplaying invalid images:")
    plot_images(dataset.invalid_images, "Invalid Images")
