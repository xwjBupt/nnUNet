import multiprocessing
import shutil
from multiprocessing import Pool
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(
    input_image: str,
    input_seg: str,
    output_image: str,
    output_seg: str,
    min_component_size: int = 50,
):
    # seg = io.imread(input_seg)
    # img = io.imread(input_image)
    # seg[seg == 255] = 1
    # image = io.imread(input_image)
    # image = image.sum(2)
    # mask = image == (3 * 255)
    # # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # # Remove the road label in these areas
    # mask = generic_filter_components(
    #     mask,
    #     filter_fn=lambda ids, sizes: [
    #         i for j, i in enumerate(ids) if sizes[j] > min_component_size
    #     ],
    # )
    # mask = binary_fill_holes(mask)
    # seg[mask] = 0
    # io.imsave(output_seg, seg, check_contrast=False)
    seg = io.imread(input_seg)
    seg = np.where(seg == 255, 1, 0).astype(np.uint8)
    if list(np.unique(seg)) != [0, 1]:
        print(input_seg, np.unique(seg))
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
    source = "/ai/mnt/data/FPDSA-Split/REMOVE-venous"
    dataset_name = "Dataset516_FPDSA"

    imagestr = join(nnUNet_raw, dataset_name, "imagesTr")
    imagests = join(nnUNet_raw, dataset_name, "imagesTs")
    labelstr = join(nnUNet_raw, dataset_name, "labelsTr")
    labelsts = join(nnUNet_raw, dataset_name, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, "train")
    test_source = join(source, "val")

    with multiprocessing.get_context("spawn").Pool(8) as p:
        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source), join=False, suffix="jpg")
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    (
                        (
                            join(train_source, v),
                            join(train_source, v[:-4] + ".png"),
                            join(imagestr, v[:-4] + "_0000.png"),
                            join(labelstr, v[:-4] + ".png"),
                            50,
                        ),
                    ),
                )
            )

        # test set
        valid_ids = subfiles(join(test_source), join=False, suffix="jpg")
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    (
                        (
                            join(test_source, v),
                            join(test_source, v[:-4] + ".png"),
                            join(imagests, v[:-4] + "_0000.png"),
                            join(labelsts, v[:-4] + ".png"),
                            50,
                        ),
                    ),
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: "R", 1: "G", 2: "B"},
        {"background": 0, "vessel": 1},
        num_train,
        ".png",
        dataset_name=dataset_name,
    )
    print("DONE")
