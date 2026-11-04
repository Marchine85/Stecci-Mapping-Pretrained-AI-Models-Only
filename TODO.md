# Pretrained model download todo

This project depends on local pretrained weights. Download them before running the pipeline.

## Required downloads

- [ ] Download the SAM checkpoint(s) into a local model folder such as `sam_model_ckpts/` or `model_ckpts/`
  - [ ] `sam_vit_h_4b8939.pth`
  - [ ] `sam_vit_l_0b3195.pth` (optional smaller alternative)
  - [ ] `sam_vit_b_01ec64.pth` (optional smaller alternative)
  - URL: https://dl.fbaipublicfiles.com/segment_anything/

- [ ] Download the Grounding DINO base model from Hugging Face
  - Model ID: `IDEA-Research/grounding-dino-base`
  - The `transformers` pipeline will fetch it automatically when the project runs.

- [ ] Download the CLIP/OpenCLIP weights used for segmentation labels
  - The project uses `open_clip` with the `ViT-B-32` OpenAI checkpoint by default.
  - These weights are downloaded automatically by the library when used.

## Local path setup

- [ ] Confirm the checkpoint paths in the scripts point to the correct local directory.
- [ ] Keep the model directory outside the Git repo or ensure it is ignored.
- [ ] Confirm the environment is created from `environment.yml`.

## Validation

- [ ] Run the rock-detection step on a sample image.
- [ ] Verify SAM isolates the rock correctly.
- [ ] Run the crack-detection stage and inspect overlays.
- [ ] Run the SAM + CLIP segmentation stage and review the label outputs.

## Notes

- The repository intentionally does not store checkpoint files or the local virtual environment.
- Keep large pretrained model files in a local data directory rather than in Git tracking.
