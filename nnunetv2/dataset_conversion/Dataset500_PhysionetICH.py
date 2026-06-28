from pathlib import Path

import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_physionet_ich(
    source_dir: str = "/home/wjx/CodeData/data/ICH/Physionet",
    dataset_id: int = 500,
    dataset_name: str = "PhysionetICH",
) -> None:
    source = Path(source_dir)
    images_src = source / "ct_scans"
    labels_src = source / "masks"

    case_ids = sorted(p.stem for p in images_src.glob("*.nii"))
    label_ids = sorted(p.stem for p in labels_src.glob("*.nii"))
    if case_ids != label_ids:
        missing_labels = sorted(set(case_ids) - set(label_ids))
        missing_images = sorted(set(label_ids) - set(case_ids))
        raise RuntimeError(
            f"Image/mask mismatch. Missing labels: {missing_labels}; missing images: {missing_images}"
        )

    target_dataset_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    target_base = Path(nnUNet_raw) / target_dataset_name
    images_tr = target_base / "imagesTr"
    labels_tr = target_base / "labelsTr"
    maybe_mkdir_p(str(images_tr))
    maybe_mkdir_p(str(labels_tr))

    for case_id in case_ids:
        image = sitk.ReadImage(str(images_src / f"{case_id}.nii"))
        sitk.WriteImage(image, str(images_tr / f"PhysionetICH_{case_id}_0000.nii.gz"), True)

        label = sitk.ReadImage(str(labels_src / f"{case_id}.nii"))
        label_arr = sitk.GetArrayFromImage(label)
        label_bin = sitk.GetImageFromArray((label_arr > 0).astype("uint8"))
        label_bin.CopyInformation(label)
        sitk.WriteImage(label_bin, str(labels_tr / f"PhysionetICH_{case_id}.nii.gz"), True)

    generate_dataset_json(
        str(target_base),
        {0: "CT"},
        {"background": 0, "hemorrhage": 1},
        len(case_ids),
        ".nii.gz",
        dataset_name=dataset_name,
        reference="PhysioNet ct-ich intracranial hemorrhage CT dataset",
        release="2018",
        description="75 non-contrast head CT scans with binary intracranial hemorrhage segmentations.",
        overwrite_image_reader_writer="SimpleITKIO",
        license="See source dataset LICENSE.txt",
        converted_by="Codex",
    )


if __name__ == "__main__":
    convert_physionet_ich()
