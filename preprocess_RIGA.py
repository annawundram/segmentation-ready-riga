import numpy as np
from PIL import Image
import cv2
import os
from sklearn.cluster import KMeans
import argparse

ERROR_LOG_PATH = None


def log_failure(path: str, reason: str):
    """Append a failed file path (and the reason) to the error log txt file."""
    print(f"FAILED: {path} -> {reason}")
    if ERROR_LOG_PATH is not None:
        with open(ERROR_LOG_PATH, "a") as f:
            f.write(f"{path}\t{reason}\n")


def preprocess(prime, anno, idx, expert, save_dir, directory):
    """Create the mask. Returns None on success, or a string with the failure reason."""
    prime = np.asarray(prime)
    anno = np.asarray(anno)

    # get annotations from expert by subtracting original image from image with drawn annotation
    edges = anno - prime

    if "MESSIDOR" in save_dir:
        edges[edges != 0] = 1
        edges = edges[:, :, 0]
    elif "BinRushed" in save_dir:
        edges = edges[:, :, 0]
        edges[edges > 220] = 0
        edges[edges < 25] = 0
        edges[edges != 0] = 255
    elif "Magrabia" in save_dir:
        edges = np.sum(edges, axis=2)
        edges[edges != 0] = 255
    else:
        return "Wrong input (unknown dataset in save_dir)"

    # find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    shapes = []

    # for all contours
    for i in range(len(contours)):
        # create an image that contains the contour filled
        hull = cv2.convexHull(contours[i])

        if len(hull) > 15:  # catch tiny pixels and discard them
            cimg = np.zeros_like(prime)
            cv2.drawContours(cimg, [hull], -1, (255, 0, 0), -1)
            shapes.append(cimg)
    shapes = np.asarray(shapes)

    # find shapes that belong to disk and to cup
    # therefore cluster found shapes by size

    # get number of pixels belonging to each shape
    shape_sizes = np.asarray(list(map(np.count_nonzero, shapes)))

    # at least two shapes must be found
    if shape_sizes.reshape(-1, 1).shape[0] < 2:
        return "Fewer than two shapes found"

    clustering_shape_sizes = KMeans(n_clusters=2, random_state=0, n_init="auto").fit_predict(
        shape_sizes.reshape(-1, 1))

    # decide which cluster belongs to which area (disk/cup)
    cluster0 = shape_sizes[clustering_shape_sizes == 0]
    cluster1 = shape_sizes[clustering_shape_sizes == 1]

    if np.asarray(list(c1 > c0 for c1, c0 in zip(cluster1, cluster0))).all() == 1:  # cluster 1 contains bigger shapes
        # cluster 1 is disk and 0 is cup
        disk_ind = np.where(clustering_shape_sizes == 1)[0]
        cup_ind = np.where(clustering_shape_sizes == 0)[0]
        disk = shapes[disk_ind]
        cup = shapes[cup_ind]
    elif np.asarray(list(c1 < c0 for c1, c0 in zip(cluster1, cluster0))).all() == 1:  # cluster 0 contains bigger shapes
        # cluster 0 is disk and 1 is cup
        disk_ind = np.where(clustering_shape_sizes == 0)[0]
        cup_ind = np.where(clustering_shape_sizes == 1)[0]
        disk = shapes[disk_ind]
        cup = shapes[cup_ind]
    else:
        # both clusters contain both sizes or else
        return "Disk/cup clustering ambiguous"

    # add up all shapes found for disk and cup respectively
    disk = np.sum(disk, axis=0)
    cup = np.sum(cup, axis=0)

    # only colour channel where hull is drawn
    disk = disk[:, :, 0]
    cup = cup[:, :, 0]

    # assign right colour values
    cup[cup != 0] = 127.5
    disk[disk != 0] = 127.5
    segmentation = disk + cup  # now disk has value 127.5 and cup has value 255 as they overlap

    # saving
    final_segmentation = np.zeros_like(prime)
    final_segmentation[:, :, 0] = segmentation
    final_segmentation[:, :, 1] = segmentation
    final_segmentation[:, :, 2] = segmentation
    Image.fromarray(final_segmentation).save(save_dir + "/image" + str(idx) + "-" + str(expert) + ".png")
    return None


def ensure_dir(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def open_image_with_extensions(base_path: str, extensions=("jpg", "tif")):
    """Try opening an image with multiple extensions. Returns (image, actual_file_path)."""
    for ext in extensions:
        filename = f"{base_path}.{ext}"
        if os.path.isfile(filename):
            return Image.open(filename), filename
    raise FileNotFoundError(f"No image found for {base_path} with extensions {extensions}")


def preprocess_dataset(
    input_subdir: str,
    output_subdir: str,
    base_input_path: str,
    base_output_path: str,
    num_images: int,
    expert_range: range = range(1, 7),
    extensions=("jpg", "tif"),
    image_prefix="image",
    case_sensitive=False,
):
    """Preprocess a dataset where input and output folder names can differ."""
    input_path = os.path.join(base_input_path, input_subdir)
    save_path = os.path.join(base_output_path, output_subdir)
    ensure_dir(save_path)

    for i in range(1, num_images + 1):
        prime_name = f"{image_prefix}{i}prime"
        prime_name_alt = f"Image{i}prime" if case_sensitive else prime_name

        try:
            try:
                prime, prime_path = open_image_with_extensions(os.path.join(input_path, prime_name), extensions)
            except FileNotFoundError:
                prime, prime_path = open_image_with_extensions(os.path.join(input_path, prime_name_alt), extensions)
        except FileNotFoundError:
            # without the original image no mask can be made for any expert
            log_failure(os.path.join(input_path, prime_name), "Original image not found")
            continue

        for j in expert_range:
            anno_name = f"{image_prefix}{i}-{j}"
            anno_name_alt = f"Image{i}-{j}" if case_sensitive else anno_name

            try:
                anno, anno_path = open_image_with_extensions(os.path.join(input_path, anno_name), extensions)
            except FileNotFoundError:
                try:
                    anno, anno_path = open_image_with_extensions(os.path.join(input_path, anno_name_alt), extensions)
                except FileNotFoundError:
                    log_failure(os.path.join(input_path, anno_name), "Annotation image not found")
                    continue

            try:
                reason = preprocess(prime, anno, i, j, save_path, input_path)
            except Exception as e:  # any unexpected error while creating the mask
                reason = f"{type(e).__name__}: {e}"
            finally:
                anno.close()

            if reason is not None:
                log_failure(anno_path, reason)

        prime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing file for the RIGA dataset.")
    parser.add_argument(
        "--RIGA_directory",
        dest="input",
        type=str,
        required=True,
        help="The directory that contains the unprocessed RIGA files.",
    )
    parser.add_argument(
        "--save_directory",
        dest="output",
        type=str,
        required=True,
        help="The directory that processed RIGA files should be saved in.",
    )
    parser.add_argument(
        "--error_log",
        dest="error_log",
        type=str,
        default=None,
        help="Txt file listing paths where mask creation failed "
             "(default: <save_directory>/failed_masks.txt).",
    )
    args = parser.parse_args()

    ensure_dir(args.output)
    ERROR_LOG_PATH = args.error_log or os.path.join(args.output, "failed_masks.txt")
    open(ERROR_LOG_PATH, "w").close()  # start with an empty log

    preprocess_dataset("MESSIDOR", "MESSIDOR", args.input, args.output, num_images=460, extensions=("tif",))
    preprocess_dataset("BinRushed/BinRushed1-Corrected", "BinRushed/BinRushed1-Corrected", args.input, args.output, num_images=50)
    preprocess_dataset("BinRushed/BinRushed2", "BinRushed/BinRushed2", args.input, args.output, num_images=47)
    preprocess_dataset("BinRushed/BinRushed3", "BinRushed/BinRushed3", args.input, args.output, num_images=47)
    preprocess_dataset("BinRushed/BinRushed4", "BinRushed/BinRushed4", args.input, args.output, num_images=47)

    # Handle spelling fix: input has "Magrabi", output should be "Magrabia"
    preprocess_dataset("Magrabia/MagrabiFemale", "Magrabia/MagrabiaFemale", args.input, args.output, num_images=47, extensions=("tif",))
    preprocess_dataset("Magrabia/MagrabiaMale", "Magrabia/MagrabiaMale", args.input, args.output, num_images=47, extensions=("tif",), case_sensitive=True)

    print(f"Done. Failed paths are listed in: {ERROR_LOG_PATH}")