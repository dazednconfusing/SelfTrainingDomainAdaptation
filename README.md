# Code for the reproduction and ablation study of "Understanding Self-Training for Gradual Domain Adaptation"
- This code is built off of [this repo](https://github.com/p-lambda/gradual_domain_adaptation)
### Dataset instructions
- The portraits dataset is an existing dataset, and can be downloaded from [here](https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0)
- After downloading, extract the tar file, and copy the "M" and "F" folders inside a folder called `dataset_32x32` inside the
code folder (current folder, where the README is). Then run `python create_dataset.py`
- Create a folder called `saved_files` in the current code folder.
- Experiments on other datasets should work without downloading additional datasets.

### Main files
- `vae.py` contains the autoencoding classifier
- `gradual_shift_better.py` contains reproduction and ablation scripts for experiments in Section 5.1 of the original paper
