import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import precision_score, recall_score, f1_score

output_folder = r""
os.makedirs(output_folder, exist_ok=True)

def map_filename(number): 
    if number == 0:
        return "No Finetuning"
    if number == 1:
        return "Big-Surround"
    if number == 2:
        return "Scene"
    if number == 3:
        return "Surround"


file_paths = [
    r".. \output_log.csv",
    r".. \output_log.csv",      #big-surround
    r".. \output_log.csv",      #scene 
    r".. \output_log.csv"]      #surround
tech=""

dataframes = [pd.read_csv(file, sep=",") for file in file_paths]
fig, ax = plt.subplots(1, len(dataframes), figsize=(20, 6), sharey=True)


for i, df in enumerate(dataframes):
    print(f"Spaltennamen der Datei {i+1}: {df.columns}")
    df[" IoU"] = pd.to_numeric(df[" IoU"], errors='coerce')
    
    if " Category" in df.columns:
        df_filtered = df[df[" Category"].isin([" Richtig vorhanden", " Falsch nicht vorhanden"])]
    else:
        df_filtered = df

    if not df_filtered.empty:
        print(len(df_filtered))
        median = df_filtered[" IoU"].median(numeric_only=True)
        mean = df_filtered[" IoU"].mean()
        count = df_filtered[" IoU"].count()
        variance = df_filtered[" IoU"].var()
        
        sns.boxplot(y=df_filtered[" IoU"], ax=ax[i])
        ax[i].set_title(f"{map_filename(i)} (Mean: {mean: .3f}, Var: {variance: .3f}, Median: {median: .3f})")
        
        ax[i].set_ylabel("IoU" if i == 0 else "")
        ax[i].set_xlabel(f"Boxplot (n={count})")

plt.tight_layout()
boxplot_path = os.path.join(output_folder, f"{tech}_boxplots_iou.png")
plt.savefig(boxplot_path)
plt.close()


dataframes = [pd.read_csv(file, sep=",") for file in file_paths]
def map_to_labels(category, is_true_label=True):
    """
    Map categories to 'vorhanden' or 'nicht vorhanden'.
    - `is_true_label`: Determines if the mapping is for true labels or predicted labels.
    """
    if is_true_label:
        if category in [" Richtig vorhanden", " Falsch nicht vorhanden"]:
            return "present"
        elif category in [" Richtig nicht vorhanden", " Falsch vorhanden"]:
            return "not present"
    else:
        if category in [" Richtig vorhanden", " Falsch vorhanden"]:
            return "present"
        elif category in [" Richtig nicht vorhanden", " Falsch nicht vorhanden"]:
            return "not present"
    return None

all_true_labels = []
all_predicted_labels = []

for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path, sep=",")
    print(f"Verarbeite Datei {i+1}: {file_path}")

    if " Category" in df.columns:
        true_labels = df[" Category"].map(lambda x: map_to_labels(x, is_true_label=True))

        predicted_labels = df[" Category"].map(lambda x: map_to_labels(x, is_true_label=False))

        valid_indices = true_labels.notna() & predicted_labels.notna()
        true_labels = true_labels[valid_indices]
        predicted_labels = predicted_labels[valid_indices]

        categories_mapped = ["vorhanden", "nicht vorhanden"]
        cm = confusion_matrix(true_labels, predicted_labels, labels=categories_mapped)

        cm_df = pd.DataFrame(cm, index=categories_mapped, columns=categories_mapped)

        precision = precision_score(true_labels, predicted_labels, labels=categories_mapped, average='binary', pos_label="vorhanden")
        recall = recall_score(true_labels, predicted_labels, labels=categories_mapped, average='binary', pos_label="vorhanden")
        f1 = f1_score(true_labels, predicted_labels, labels=categories_mapped, average='binary', pos_label="vorhanden")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.title(f'{map_filename(i)} - p:{precision: .3}, r:{recall: .3}, f1:{f1: .3}')

        file_name = f"{tech}_{map_filename(i)}_confusion_matrix.png"
        conf_matrix_path = os.path.join(output_folder, file_name)
        plt.savefig(conf_matrix_path)
        plt.close()

        print(f"Konfusionsmatrix für Datei {map_filename(i)} wurde erstellt und gespeichert in: {conf_matrix_path}")