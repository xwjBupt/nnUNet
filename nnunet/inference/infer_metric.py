import torch
import numpy as np
from tqdm import tqdm
import glob
from loguru import logger
import SimpleITK as sitk
import argparse
import torch
import os
import csv


def write_to_csv(filename, content):
    file_exist = os.path.exists(filename)
    with open(filename, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        if not file_exist:
            writer.writeheader()

        writer.writerow(content)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def infer_metric(pred_dir, gt_dir, mode="2d"):

    online_eval_foreground_dc = []
    online_eval_foreground_acc = []
    online_eval_foreground_sen = []
    online_eval_foreground_spe = []
    online_eval_tp = []
    online_eval_fp = []
    online_eval_tn = []
    online_eval_fn = []
    logger.add(pred_dir + "/LOG.log")
    preds = glob.glob(pred_dir + "/*.nii.gz")
    for pred in tqdm(preds):
        gt = gt_dir + pred.split("/")[-1]
        output_seg = torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(pred)).astype(np.int32)
        )
        target = torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(gt)).astype(np.int32)
        )
        num_classes = len(list(np.unique(target)))
        axes = tuple(range(1, len(target.shape)))
        tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(
            output_seg.device.index
        )
        fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(
            output_seg.device.index
        )
        tn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(
            output_seg.device.index
        )
        fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(
            output_seg.device.index
        )
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor(
                (output_seg == c).float() * (target == c).float(), axes=axes
            )
            fp_hard[:, c - 1] = sum_tensor(
                (output_seg == c).float() * (target != c).float(), axes=axes
            )
            tn_hard[:, c - 1] = sum_tensor(
                (output_seg != c).float() * (target != c).float(), axes=axes
            )
            fn_hard[:, c - 1] = sum_tensor(
                (output_seg != c).float() * (target == c).float(), axes=axes
            )

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            tn_hard = tn_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            online_eval_foreground_dc.append(
                list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
            )
            online_eval_foreground_acc.append(
                list(
                    (tp_hard + tn_hard) / (tp_hard + fp_hard + tn_hard + fn_hard + 1e-8)
                )
            )
            online_eval_foreground_sen.append(
                list((tp_hard) / (tp_hard + fn_hard + 1e-8))
            )
            online_eval_foreground_spe.append(
                list((tn_hard) / (fp_hard + tn_hard + 1e-8))
            )

            online_eval_tp.append(list(tp_hard))
            online_eval_fp.append(list(fp_hard))
            online_eval_tn.append(list(tn_hard))
            online_eval_fn.append(list(fn_hard))

    online_eval_tp = np.sum(online_eval_tp, 0)
    online_eval_fp = np.sum(online_eval_fp, 0)
    online_eval_tn = np.sum(online_eval_tn, 0)
    online_eval_fn = np.sum(online_eval_fn, 0)

    global_dc_per_class = [
        i
        for i in [
            2 * i / (2 * i + j + k)
            for i, j, k in zip(online_eval_tp, online_eval_fp, online_eval_fn)
        ]
        if not np.isnan(i)
    ]

    global_acc_per_class = [
        i
        for i in [
            (i + k) / (i + j + k + l)
            for i, j, k, l in zip(
                online_eval_tp,
                online_eval_fp,
                online_eval_tn,
                online_eval_fn,
            )
        ]
        if not np.isnan(i)
    ]

    global_sen_per_class = [
        i
        for i in [i / (i + j) for i, j in zip(online_eval_tp, online_eval_fn)]
        if not np.isnan(i)
    ]

    global_spe_per_class = [
        i
        for i in [i / (i + j) for i, j in zip(online_eval_tn, online_eval_fp)]
        if not np.isnan(i)
    ]

    global_jaccard_per_class = [
        i
        for i in [
            (i) / (i + j + l)
            for i, j, k, l in zip(
                online_eval_tp,
                online_eval_fp,
                online_eval_tn,
                online_eval_fn,
            )
        ]
        if not np.isnan(i)
    ]

    global_pre_per_class = [
        i
        for i in [
            (i) / (i + j)
            for i, j, k, l in zip(
                online_eval_tp,
                online_eval_fp,
                online_eval_tn,
                online_eval_fn,
            )
        ]
        if not np.isnan(i)
    ]

    global_fpr_per_class = [
        i
        for i in [
            (j) / (j + k)
            for i, j, k, l in zip(
                online_eval_tp,
                online_eval_fp,
                online_eval_tn,
                online_eval_fn,
            )
        ]
        if not np.isnan(i)
    ]

    global_fnr_per_class = [
        i
        for i in [
            (l) / (l + i)
            for i, j, k, l in zip(
                online_eval_tp,
                online_eval_fp,
                online_eval_tn,
                online_eval_fn,
            )
        ]
        if not np.isnan(i)
    ]

    logger.info(
        ">>>> ### Metric infer on: PRED_DIR {} ---- GT_DIR {} ### >>>>".format(
            pred_dir, gt_dir
        )
    )
    logger.info(
        "\n ACC: {} \n F1: {}\n JACCARD: {}\n PRECISION: {}\n SEN: {}\n SPE:{}\n FNR:{} \n FPR:{} \n ".format(
            global_acc_per_class,
            global_dc_per_class,
            global_jaccard_per_class,
            global_pre_per_class,
            global_sen_per_class,
            global_spe_per_class,
            global_fnr_per_class,
            global_fpr_per_class,
        )
    )
    content = dict(
        METHOD=pred_dir.split("nnUNet_trained_models/nnUNet")[1],
        ACC=global_acc_per_class,
        F1=global_dc_per_class,
        JACCARD=global_jaccard_per_class,
        PRECISION=global_pre_per_class,
        SEN=global_sen_per_class,
        SPE=global_spe_per_class,
        FNR=global_fnr_per_class,
        FPR=global_fpr_per_class,
    )
    logger.info("<<<< ### Metric infer ### <<<<".format(pred_dir, gt_dir))
    filename = (
        pred_dir.split("nnUNetTrainer")[0] + pred_dir.split("/")[9] + "_results.csv"
    )
    logger.info(">>>> ### Write Metric To {} ### <<<<".format(filename))
    write_to_csv(filename=filename, content=content)
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer your model")
    parser.add_argument(
        "--pred_dir",
        default="/home/user/skip/code/nnUNet/nnUNet_trained_models/nnUNet/2d/Task512_FPDSA/nnUNetTrainerV2__nnUNetPlansv2.1/12_20-10_53/ERNET-wDS-epoch500/predictions/",
        help="the path of pred seg results",
    )
    parser.add_argument(
        "--gt_dir",
        default="/home/user/skip/code/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task512_FPDSA/labelsTs/",
        help="the path of gt annotations",
    )
    parser.add_argument(
        "--mode",
        default="2d",
        choices=["2d", "3d"],
        help="the 2d segmentation or 3d segmentation",
    )
    args = parser.parse_args()
    infer_metric(pred_dir=args.pred_dir, gt_dir=args.gt_dir, mode=args.mode)
