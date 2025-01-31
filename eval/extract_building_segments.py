import cv2
import numpy as np
from pathlib import Path

# Farben für die verschiedenen Gebäudetypen
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
    """
    Konvertiert einen Farbwert von RGB zu BGR.
    """
    return (color[2], color[1], color[0], color[3])

def extract_building_segments(image_path, output_dir, image_number):
    """
    Extrahiert die verschiedenen Gebäude-Segmentierungen aus einem semantisch segmentierten Bild
    und speichert sie als separate Bilder.

    Parameter:
        image_path: Pfad zum semantisch segmentierten Bild
        output_dir: Verzeichnis, in dem die extrahierten Bilder gespeichert werden
    """
    # Bild laden
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    # Ausgabeordner erstellen
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for building_name, color in BUILDING_COLORS.items():
        # Farbwerte in BGR umwandeln
        target_color = np.array(convert_rgb_to_bgr(color), dtype=np.uint8)

        # Maske erstellen
        mask = cv2.inRange(image, target_color, target_color)

        # Überprüfen, ob das Gebäude im Bild vorhanden ist
        if np.any(mask > 0):
            # Maske als separates Bild speichern
            output_path = output_dir / f"{image_number}_{building_name}.png"
            cv2.imwrite(str(output_path), mask)
            print(f"Segment für {building_name} gespeichert: {output_path}")
        else:
            print(f"{building_name} nicht im Bild gefunden.")

# Beispiel für die Verwendung
segmented_image_path = Path(r"F:\Studium\Master\Thesis\data\perception\usefull_data\test_data\step836.camera.semantic segmentation.png")
image_number = "836"
output_directory = Path(r"F:\Studium\Master\Thesis\data\perception\usefull_data\test_data\segmented_semantic_segmentation")

try:
    extract_building_segments(segmented_image_path, output_directory, image_number)
except FileNotFoundError as e:
    print(e)
