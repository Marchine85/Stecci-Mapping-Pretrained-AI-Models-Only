# Stecci Mapping - Pretrained AI Models Only

This repository contains a Python-based image analysis pipeline for mapping a stecci (a carved tombstone or stone monument) from a photograph. The goal is to find the central rock, isolate it from the background, detect cracks, and classify surface features such as biological growth.

The project uses pretrained models and classical image-processing methods together so each component is applied where it is strongest.

## What is done here

This workflow is built around four chained stages:

1. Grounding DINO finds the rock area in the image and returns a rough bounding box.
2. SAM (Segment Anything Model) turns that box into a precise pixel mask of the rock silhouette.
3. Classical computer vision detects cracks within the isolated rock using morphology and ridge-based filtering.
4. SAM auto-segmentation plus CLIP classifies rock sub-regions, such as biological colonization or similar surface features.

In short: the system answers, in sequence:

- Where is the rock?
- How can the rock be isolated precisely?
- Which dark linear structures are cracks?
- What parts of the rock surface appear to be biological growth or other materials?

## Summary of the pipeline

### 1. Grounding DINO: locate the rock
Grounding DINO is used as an open-vocabulary object detector. Instead of a fixed class list, it receives a text prompt such as "rock" and returns candidate boxes matching that description. The script picks the box closest to the image centre, assuming the stecci is the main subject.

### 2. SAM: isolate the rock mask
The selected box is passed into SAM as a prompt. SAM generates a precise rock mask that follows the actual silhouette of the stone. Everything outside the rock is removed so the following analysis stages focus only on the real rock region.

### 3. Classical CV: detect cracks
Once the rock is isolated, classical image-processing methods look for cracks. The project supports multiple approaches, including:

- tiled black-hat morphology with local thresholding
- Frangi multiscale ridge filtering for elongated crack-like structures

This stage filters crack candidates by geometry such as aspect ratio, length, and area and overlays the results back on the rock mask.

### 4. SAM + CLIP: segment sub-regions and classify them
SAM is used again in auto-mask mode to divide the rock into smaller regions. Each region is then classified by CLIP against text prompts such as "biological colonization", "moss", or related descriptors. Regions that are too uncertain fall back to a safe default label.

### 5. Review / QA viewer
The project includes an interactive Matplotlib viewer for overlaying the detected layers (rock, cracks, segment labels, background) so the results can be reviewed visually.

## Files in this project

The repository currently includes the following kinds of scripts:

- `SAM.py` - basic SAM mask generation example
- `option1_sam.py` - SAM automatic segmentation workflow
- `option2_classical_cv_for_cracks.py` - classical crack detection experiments
- `option3_sam_clip.py` - SAM + CLIP segmentation/classification
- `option4_grounding_dino_sam.py` - Grounding DINO + SAM object detection pipeline
- `option5_combined_gdino_cv_sam.py` - full rock isolation + crack + segmentation workflow
- `crack_detection.py` - crack detection algorithms and filtering
- `crack_param_tuner.py` - interactive tuning for classical crack detection
- `rock_contour_tuner.py` - tuning for edge/contour rendering
- `ClassicCV.py` - classical CV examples
- `show_image.py` - image viewing helper

## Setup

To create the environment:

```bash
conda env create -f environment.yml
conda activate stecci-mapping-pretrained
```

The project expects pretrained weights to be downloaded and stored locally outside of Git tracking. The repo intentionally ignores the virtual environment and model checkpoints so they are not committed to version control.

## Pretrained model downloads

This project depends on downloaded weights that are not bundled in this repository.

- SAM checkpoints from Meta's Segment Anything project
  - vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  - vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  - vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- Grounding DINO weights from Hugging Face (`IDEA-Research/grounding-dino-base`)
- CLIP weights via `open_clip` / Hugging Face when the classification stage is run

The checkpoints should be placed in a local folder such as `sam_model_ckpts/` or `model_ckpts/`, but that folder is excluded from git by the repository ignore rules.

## Repository intent

This repository is intentionally focused on pretrained-model-based mapping and analysis only. It does not include the trained model files or the local Python virtual environment. The project is designed to be reproducible when the required weights are downloaded into a local environment.

## Notes

- Model files and the environment are intentionally excluded from Git.
- Scripts may reference local checkpoint paths depending on the machine used for development.
- This is a research/prototyping workflow rather than a packaged deployment pipeline.
