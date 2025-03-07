# finetune_eval_th

This repository contains the scripts for fine-tuning the CLIP, Resnet and Mask R-CNN models from the thesis “CERF”.

The scripts were tested with Microsoft 11, Python 3.12, CUDA 11.8, matplotlib 3.9.2, numpy 2.1.3, open_clip_torch 2.29.0, opencv-python 4.10.0.84, pandas 2.2.3, pillow 11.0.0, scikit-learn 1 .5.2, segment_anything @ git+https://github.com/facebookresearch/segment-anything.git, torch 2.5.1+cu118, transformers 4.46.3, tqdm 4.67.0, seaborn 0.13.2

the references-scripts:
coco_eval.py
coco_utils.py
engine.py
transforms.py
utils.py
were used from the pytorch repository (vision/references/detection at main · pytorch/vision)

## Finetuning
The fine-tuning can be obtained from [this link](). This can be used to train the respective Mask R-CNN, ResNet and CLIP models.

Starting from the Big-Surround fine-tuning dataset, the path to the dataset can be stored in the script [finetune_CLIP.py](./finetune_CLIP.py) in line 111 r“.\finetune_data\building_big_surround_pictures” depending on where the dataset was stored. The Name of the Model can be changed in line 22.

For the Mask R-CNN model, the path can be stored in the script [finetune_MRCNN.py](./finetune_MRCNN.py) in line 130. The Name of the modle which will be saved can be changed in line 152.

For ResNet in the script [finetune_Resnet.py](./finetune_Resnet.py) in line 134. The name of the saved model can be changed in line 175.

The scripts then save the models in the same path where the scripts are located. These can then be used to apply the respective techniques to the test images.
LeRF and feature splatting are explained in the respective repositories. Pre-trained models can be downloaded from [this share]().

## Using Mask R-CNN and ResNet + SAM - Approach
For Mask R-CNN, the script [M-RCNN_identifyer.py](./M-RCNN_identifier.py) can be used. In line 66, the path to the finetuned model must be entered. In line 124, the path to the images to be analyzed. (The test images from the thesis can be found at this link)
In line 125, the path to the output folder can be stored where the images with the masking should be saved.

The script [ResNet_identifier.py](./ResNet_identifier.py) can be used for the ResNet + SAM approach. It is assumed that a SAM model can be found under the path “./chkpts/sam_vit_h_4b8939.pth”. This can be changed if necessary on line 127. A SAM model can be downloaded from [this link](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). (In this Project the Model ViT-H SAM was used)

The fine-tuned ResNet model can be added in line 143 with the corresponding path.

In line 154, the path to the images to be analyzed can be stored and in line 155 the path where the images with the masks are to be stored.

## Evaluating
The following steps must be taken for the evaluation.
1. The ground truth masks of the individual buildings must exist in the corresponding images.
2. The ground truth masks and the “just-mask” images of the individual techniques must be stored.
3. the generated.csv files of a technique must be specified in the correct order in the plots.py file.

The script [extract_building_segments.py](./eval/extract_building_segments.py) can be used to generate the masks for the test images. For each image, the path to the semantic segmentation image must be specified on line 44, in line 45 the image number (925 Images where createt in the scene path to create the radiance field) and in line 46 the path where the results are to be saved. For simplification, you can simply use the segmented data set from [here]().

In with the [single_building_iou.py](eval/single_building_iou.py), the iou of a technique can be calculated with respect to a fine-tuning data set. The script assumes that the folder structure is organized as follows when calculating:

- root(usually Technique name and finetuning dataset name, but doesn't matter)
    - A-Building (continue with all 14 Building names)
    - B-Building
    - C-Building...

Example Folder Structure can be found in [this link](), where the results of this thesis are.
The path to the individual extracted ground-truth masks can be stored in line 106. The path to the images in which only the predicted masks of a technique of a fine-tuning data set are located can be stored in line 108. In line 109, a path must then be specified where the csv file in which the results are stored is to be saved.

In the [plots.py](./eval/plots.py) script, the .csv files must be specified in lines 23 - 26. the script is adapted so that the csv. files without fine-tuning are specified first, then those of the big-surround data set, then scene and then those of the surround data set.
The path where the plots are to be saved can be specified in line 8. The technique used (mrcnn, resnet, lerf or feature-splatting) can be specified in line 27 so that it is stored in the names of the box plots.