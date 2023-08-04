from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import glob
import random
from tqdm import tqdm


def convert_ich2023(
    ich_base_dir: str, nnunet_dataset_id: int = 220, train_test_ratio: float = 0.7
):
    task_name = "ICH2023"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)

    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    all_cases = glob.glob(ich_base_dir + "/ICH-PA*_0000.nii.gz")
    random.shuffle(all_cases)
    random.shuffle(all_cases)
    num_samples = len(all_cases)
    train_cases = all_cases[: int(num_samples * train_test_ratio)]
    test_cases = all_cases[int(num_samples * train_test_ratio) :]
    print("DEAL WITH TRAINING SAMPLE")
    for train_case in tqdm(train_cases):
        scan_name = train_case.split("/")[-1]
        label_name = scan_name.replace("_0000.nii.gz", ".nii.gz")

        shutil.copy(train_case, join(imagestr, scan_name))
        shutil.copy(
            train_case.replace("_0000.nii.gz", ".nii.gz"),
            join(labelstr, label_name),
        )

    print("DEAL WITH TEST SAMPLE")
    for test_case in tqdm(test_cases):
        scan_name = test_case.split("/")[-1]
        label_name = scan_name.replace("_0000.nii.gz", ".nii.gz")
        shutil.copy(test_case, join(imagests, scan_name))
        shutil.copy(
            test_case.replace("_0000.nii.gz", ".nii.gz"),
            join(labelsts, label_name),
        )

    print("generate_dataset_json")
    generate_dataset_json(
        out_base,
        {0: "CT"},
        labels={
            "background": 0,
            "hemorrhage": 1,
        },
        num_training_cases=len(train_cases),
        num_test_cases=len(test_cases),
        file_ending=".nii.gz",
        dataset_name=task_name,
        reference="none",
        release="prerelease",
        overwrite_image_reader_writer="SimpleITKIO",
        description="ICH2023",
        train_cases=train_cases,
        test_cases=test_cases,
    )
    print("DONE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/ai/mnt/data/JNIS-SITK/ICH-ALL-0804-edit",
        help="The downloaded and extracted ICH2023 dataset (must have case_XXXXX subfolders)",
    )
    parser.add_argument(
        "--d",
        required=False,
        type=int,
        default=515,
        help="nnU-Net Dataset ID, default: 515",
    )
    args = parser.parse_args()
    amos_base = args.input_folder
    convert_ich2023(amos_base, args.d)
