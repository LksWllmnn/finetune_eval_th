import cv2
import numpy as np
from pathlib import Path

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

def convert_rgb_to_bgr(color):
    return (color[2], color[1], color[0], color[3])

def extract_building_segments(image_path, output_dir, image_number):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for building_name, color in BUILDING_COLORS.items():
        target_color = np.array(convert_rgb_to_bgr(color), dtype=np.uint8)

        mask = cv2.inRange(image, target_color, target_color)

        if np.any(mask > 0):
            output_path = output_dir / f"{image_number}_{building_name}.png"
            cv2.imwrite(str(output_path), mask)
            print(f"Segment für {building_name} gespeichert: {output_path}")
        else:
            print(f"{building_name} nicht im Bild gefunden.")

segmented_image_path = Path(r"")
image_number = ""
output_directory = Path(r"")

try:
    extract_building_segments(segmented_image_path, output_directory, image_number)
except FileNotFoundError as e:
    print(e)
