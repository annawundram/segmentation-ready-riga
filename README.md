# RIGA Optic Disc / Cup Mask Preprocessing

This repository contains a preprocessing script that converts the raw **RIGA** (Retinal fundus Images for Glaucoma Analysis) dataset into clean, multi-rater segmentation masks of the **optic disc** and **optic cup**.

The raw RIGA data does not ship with ready-made masks. Instead, each expert drew their outline directly onto a copy of the fundus image. `preprocess_RIGA.py` recovers those drawn outlines, turns them into filled regions, separates disc from cup, and saves one mask per image and expert.

You can directly download the preprocessed dataset from Zenodo [here](https://doi.org/10.5281/zenodo.23158636).

---

## Contents

- [Requirements](#requirements)
- [Expected input layout](#expected-input-layout)
- [Usage](#usage)
- [How the data is processed](#how-the-data-is-processed)
- [Output format](#output-format)
- [Error log](#error-log)
- [Known limitations](#known-limitations)

---

## Requirements

- Python 3.8+
- `numpy`
- `Pillow`
- `opencv-python`
- `scikit-learn` (>= 1.2, since `KMeans(n_init="auto")` is used)

```bash
pip install numpy pillow opencv-python scikit-learn
```

---

## Expected input layout

The script expects the unprocessed RIGA files in the following structure. Each image `N` has one original image (`imageNprime`) and one annotated copy per expert (`imageN-J`, with `J` from 1 to 6).

```
<RIGA_directory>/
├── MESSIDOR/                      # 460 images, .tif
├── BinRushed/
│   ├── BinRushed1-Corrected/      # 50 images, .jpg or .tif
│   ├── BinRushed2/                # 47 images
│   ├── BinRushed3/                # 47 images
│   └── BinRushed4/                # 47 images
└── Magrabia/
    ├── MagrabiFemale/             # 47 images, .tif
    └── MagrabiaMale/              # 47 images, .tif (files named "ImageN...")
```

File naming inside each folder:

| File | Meaning |
|------|---------|
| `image{N}prime.<ext>` | Original fundus image, no annotation |
| `image{N}-{J}.<ext>` | Same image with the outline of expert `J` drawn on it |

The `MagrabiaMale` folder uses a capitalised prefix (`Image{N}prime`, `Image{N}-{J}`). The script falls back to this spelling automatically.

---

## Usage
Either directly download the data from Zenodo [here](https://doi.org/10.5281/zenodo.23158636) or compute them yourself as follows

```bash
python preprocess_riga.py \
    --RIGA_directory /path/to/raw/RIGA \
    --save_directory /path/to/processed/RIGA
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--RIGA_directory` | yes | Directory containing the unprocessed RIGA files |
| `--save_directory` | yes | Directory where the processed masks are written |
| `--error_log` | no | Path of the txt file listing failed masks (default: `<save_directory>/failed_masks.txt`) |

---

## How the data is processed

For every image and every expert, the script runs the following pipeline.

### 1. Load the image pair

The original image (`prime`) and the expert's annotated copy (`anno`) are opened. Several file extensions are tried (`.jpg`, `.tif`, depending on the dataset), and the alternate capitalisation is tried if the first name is not found.

### 2. Extract the drawn outline

The expert's annotation is recovered by **subtracting the original image from the annotated image**:

```
edges = anno - prime
```

Pixels that are identical in both images become 0. Only the pixels the expert drew remain. Because each sub-dataset was annotated with a different drawing tool and colour, the result is thresholded differently per dataset:

| Dataset | Processing of `edges` |
|---------|-----------------------|
| **MESSIDOR** | Every non-zero pixel is set to 1; the first colour channel is used |
| **BinRushed** | First colour channel only; values above 220 or below 25 are discarded (removes noise and compression artefacts); the rest is set to 255 |
| **Magrabia** | Channels are summed; every non-zero pixel is set to 255 |

The dataset is detected from the output path, so the output folder names must contain `MESSIDOR`, `BinRushed` or `Magrabia`.

### 3. Find contours and fill them

`cv2.findContours` detects the outlines in the difference image. For each contour:

1. The **convex hull** is computed, which closes gaps in hand-drawn lines.
2. Hulls with 15 points or fewer are discarded as stray pixels or noise.
3. The remaining hull is drawn **filled** onto a blank image, giving one filled shape per contour.

### 4. Separate optic disc and optic cup

An expert draws two outlines per image: a large one (disc) and a smaller one inside it (cup). To tell them apart without relying on drawing order or colour:

1. The number of pixels in each filled shape is counted.
2. The shape sizes are clustered into **two groups with k-means** (`k = 2`).
3. The cluster with the larger shapes is labelled **disc**, the other **cup**. This is only accepted if the ordering is consistent, meaning every shape in one cluster is larger than its counterpart in the other.
4. All shapes in each cluster are summed into a single disc image and a single cup image.

At least two shapes must be found, otherwise the image is reported as failed.

### 5. Combine and save the mask

Disc and cup are combined into one mask. Because the cup lies inside the disc, the overlapping region adds up to the highest value. The mask is saved as an RGB PNG (all three channels identical).

---

## Output format

Masks are written to the matching sub-folder of `--save_directory`, preserving the dataset structure:

```
<save_directory>/
├── MESSIDOR/image{N}-{J}.png
├── BinRushed/BinRushed1-Corrected/image{N}-{J}.png
├── BinRushed/BinRushed2/...
├── BinRushed/BinRushed3/...
├── BinRushed/BinRushed4/...
├── Magrabia/MagrabiaFemale/...
├── Magrabia/MagrabiaMale/...
└── failed_masks.txt
```

Pixel values in each mask (stored as `uint8`):

| Value | Region |
|-------|--------|
| `0` | Background |
| `127` | Optic disc (outside the cup) |
| `255` | Optic cup (inside the disc) |

`N` is the image index and `J` the expert index (1 to 6), so each image has up to six masks, one per rater. The Magrabia input folder `MagrabiFemale` is written to `MagrabiaFemale` in the output.

---

## Error log

Some annotations cannot be converted, for example because fewer than two shapes were found. These are written to a txt file so you can inspect or exclude them. Each line contains the path of the problematic file and the reason, separated by a tab:

```
/path/to/RIGA/BinRushed/BinRushed2/image17-3.tif	Fewer than two shapes found
/path/to/RIGA/MESSIDOR/image102-5.tif	Disk/cup clustering ambiguous
/path/to/RIGA/Magrabia/MagrabiaMale/Image31prime	Original image not found
```

Possible reasons:

| Reason | Meaning |
|--------|---------|
| `Original image not found` | The `...prime` file is missing, so all experts for that image are skipped |
| `Annotation image not found` | A single expert file is missing |
| `Fewer than two shapes found` | The difference image did not contain both a disc and a cup outline |
| `Disk/cup clustering ambiguous` | The size clusters overlap, so disc and cup could not be separated reliably |
| `Wrong input (unknown dataset in save_dir)` | The output path contains none of `MESSIDOR`, `BinRushed`, `Magrabia` |
| `<ExceptionType>: <message>` | Any other unexpected error during mask creation |

The script continues after a failure, so one bad file does not stop the run. The log is reset at the start of every run.

---

## Known limitations

- **Annotation-dependent.** Masks are only as good as the drawn outlines. Gaps, overlapping lines, or annotations in an unexpected colour can cause a failure or an inaccurate mask.
- **Convex hull.** Each outline is replaced by its convex hull, so concave disc or cup shapes are slightly over-segmented.
- **Dataset detection by path.** Thresholding is chosen from substrings in the output path, so renaming output folders will break it.

## Disclaimer
This README was generated with AI assistance. No AI was used for the code generation.

## Citation
If you use this dataset, please cite it as follows
```
Wundram, A. M., & Baumgartner, C. (2026). Segmentation-Ready-RIGA (Version 1.0.0)
[Data set]. Zenodo. https://doi.org/10.5281/zenodo.23158636
```

