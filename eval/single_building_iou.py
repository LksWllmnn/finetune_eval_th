import cv2
import numpy as np
from pathlib import Path

def calculate_iou(mask_gt, mask_pred):
    """
    Berechnet die Intersection over Union (IoU) zweier Masken.
    """
    mask_gt_bin = (mask_gt > 0).astype(np.uint8)
    mask_pred_bin = (mask_pred > 0).astype(np.uint8)

    intersection = np.sum(np.logical_and(mask_gt_bin, mask_pred_bin))
    union = np.sum(np.logical_or(mask_gt_bin, mask_pred_bin))

    return intersection / union if union > 0 else 0

def load_image_as_mask(image_path):
    """
    Lädt ein Bild und konvertiert es in eine binäre Maske.
    """
    if image_path is None or not image_path.exists():
        return None
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    return image

def compare_masks(gt_path, pred_path):
    """
    Vergleicht eine Ground-Truth-Maske mit einer vorhergesagten Maske und berechnet die IoU.
    Zusätzlich klassifiziert die Methode die Vorhersage in vier Kategorien.
    """
    mask_gt = load_image_as_mask(gt_path)
    mask_pred = load_image_as_mask(pred_path)

    # Überprüfung auf komplett schwarze vorhergesagte Bilder
    if mask_pred is not None and np.sum(mask_pred) == 0:
        mask_pred = None

    if mask_gt is None and mask_pred is not None:
        # Gebäude wurde fälschlicherweise vorhergesagt
        return 0.0, "Falsch vorhanden"
    elif mask_gt is not None and mask_pred is None:
        # Gebäude wurde fälschlicherweise nicht vorhergesagt
        return 0.0, "Falsch nicht vorhanden"
    elif mask_gt is None and mask_pred is None:
        # Gebäude ist korrekt nicht vorhanden
        return None, "Richtig nicht vorhanden"
    else:
        # Gebäude ist korrekt vorhanden
        iou = calculate_iou(mask_gt, mask_pred)
        return iou, "Richtig vorhanden"

def process_directory(gt_dir, pred_dir, output_log, mapping):
    """
    Verarbeitet die Ground-Truth- und Vorhersage-Ordner und berechnet IoU-Werte für alle Paare.
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    output_log = Path(output_log)

    # Erstellen des Logs
    with output_log.open("w") as log_file:
        log_file.write("Building, GT_Image, Pred_Image, IoU, Category\n")
        
        for building in BUILDING_COLORS.keys():
            gt_files = {str(img.stem).split("_")[0]: img for img in gt_dir.glob(f"*_{building}.png")}
            pred_files = {str(i).zfill(5): img for i, img in enumerate(pred_dir.joinpath(building).glob("*"))}
            
            for pred_index, gt_index in mapping.items():
                gt_image = gt_files.get(gt_index, None)
                pred_image = pred_files.get(pred_index, None)
                iou, cat = compare_masks(gt_image, pred_image)

                # Protokoll in die Log-Datei schreiben
                log_file.write(f"{building}, {gt_index or 'None'}, {pred_index or 'None'}, {iou if iou is not None else 'N/A'}, {cat}\n")

# Beispielkonfiguration
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

mapping = {
    "00000": "502",
    "00001": "291",
    "00002": "333",
    "00003": "422",
    "00004": "836",
    "00005": "568",
    "00006": "654",
}

gt_dir = Path(r"F:\Studium\Master\Thesis\data\perception\usefull_data\test_data\segmented_semantic_segmentation")

#lerf-lite stuff
# pred_dir = Path(r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\no-finetuning\just-mask")
# output_log = Path(r"F:\Studium\Master\Thesis\data\perception\usefull_data\lerf-lite-data\renders\no-finetuning\just-mask\output_log.txt")

# resnet stuff
pred_dir = Path(r"F:\Studium\Master\Thesis\data\final_final_results\resnet_scene\just-mask")
output_log = Path(r"F:\Studium\Master\Thesis\data\final_final_results\resnet_scene\just-mask\output_log.csv")
process_directory(gt_dir=gt_dir, pred_dir=pred_dir, output_log=output_log, mapping=mapping)
