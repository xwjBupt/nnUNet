import os
import socket
from typing import Union, Optional
import pdb
import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn
from nnunetv2.inference.predict_from_raw_data import predict_entry_point_function
from nnunetv2.evaluation.evaluate_predictions import (
    evaluate_folder_entry_point_function,
)
from git import Repo
import glob
import yaml
import time
import os
import setproctitle


def git_commit(
    work_dir,
    levels=7,
    postfixs=[".py", ".sh"],
    commit_info="",
    debug=False,
):
    cid = "not generate"
    branch = "master"
    if not debug:
        repo = Repo(work_dir)
        toadd = []
        branch = repo.active_branch.name
        for i in range(levels):
            for postfix in postfixs:
                filename = glob.glob(work_dir + (i + 1) * "/*" + postfix)
                for x in filename:
                    if (
                        not ("play" in x)
                        and not ("local" in x)
                        and not ("Untitled" in x)
                        and not ("wandb" in x)
                    ):
                        toadd.append(x)
        index = repo.index  # 获取暂存区对象
        index.add(toadd)
        index.commit(commit_info)
        cid = repo.head.commit.hexsha

    commit_tag = (
        "COMMIT INFO >>> "
        + commit_info
        + " <<< \n"
        + "COMMIT BRANCH >>> "
        + branch
        + " <<< \n"
        + "COMMIT ID >>> "
        + cid
        + " <<<"
    )
    return commit_tag


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(
    dataset_name_or_id: Union[int, str],
    configuration: str,
    fold: int,
    trainer_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    use_compressed: bool = False,
    device: torch.device = torch.device("cuda"),
    record_commit_info: str = "record_commit_info",
    previous_stage: str = "previous_stage",
    arc: str = "undeclared",
):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if nnunet_trainer is None:
        raise RuntimeError(
            f"Could not find requested nnunet trainer {trainer_name} in "
            f"nnunetv2.training.nnUNetTrainer ("
            f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
            f"else, please move it there."
        )
    assert issubclass(nnunet_trainer, nnUNetTrainer), (
        "The requested nnunet trainer class must inherit from " "nnUNetTrainer"
    )

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith("Dataset"):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(
                f"dataset_name_or_id must either be an integer or a valid dataset name with the pattern "
                f"DatasetXXX_YYY where XXX are the three(!) task ID digits. Your "
                f"input: {dataset_name_or_id}"
            )

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(
        nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id)
    )
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, "dataset.json"))
    nnunet_trainer = nnunet_trainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=not use_compressed,
        device=device,
        record_commit_info=record_commit_info,
        previous_stage=previous_stage,
        arc=arc,
    )
    return nnunet_trainer


def maybe_load_checkpoint(
    nnunet_trainer: nnUNetTrainer,
    continue_training: bool,
    validation_only: bool,
    pretrained_weights_file: str = None,
    arc: str = "undeclared",
):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError(
            "Cannot both continue a training AND load pretrained weights. Pretrained weights can only "
            "be used at the beginning of the training."
        )
    if continue_training:
        expected_checkpoint_file = join(
            nnunet_trainer.output_folder, "checkpoint_final.pth"
        )
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(
                nnunet_trainer.output_folder, "checkpoint_latest.pth"
            )
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(
                nnunet_trainer.output_folder, "checkpoint_best.pth"
            )
        if not isfile(expected_checkpoint_file):
            print(
                f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                f"continue from. Starting a new training..."
            )
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(
            nnunet_trainer.output_folder, "checkpoint_final.pth"
        )
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(
                f"Cannot run validation because the training is not finished yet!"
            )
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(
                nnunet_trainer.network, pretrained_weights_file, arc=arc, verbose=True
            )
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(
    rank,
    dataset_name_or_id,
    configuration,
    fold,
    tr,
    p,
    use_compressed,
    disable_checkpointing,
    c,
    val,
    pretrained_weights,
    npz,
    world_size,
):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device("cuda", dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(
        dataset_name_or_id, configuration, fold, tr, p, use_compressed
    )

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f"Cannot set --c and --val flag at the same time. Dummy."

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def run_training(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    use_compressed_data: bool = False,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    device: torch.device = torch.device("cuda"),
    record_commit_info: str = "IN DEBUG DO NOT COMMIT",
    previous_stage: str = "previous_stage",
    arc: str = "undeclared",
):
    if isinstance(fold, str):
        if fold != "all":
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!'
                )
                raise e

    if num_gpus > 1:
        assert (
            device.type == "cuda"
        ), f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ["MASTER_PORT"] = port  # str(port)

        mp.spawn(
            run_ddp,
            args=(
                dataset_name_or_id,
                configuration,
                fold,
                trainer_class_name,
                plans_identifier,
                use_compressed_data,
                disable_checkpointing,
                continue_training,
                only_run_validation,
                pretrained_weights,
                export_validation_probabilities,
                num_gpus,
            ),
            nprocs=num_gpus,
            join=True,
        )
    else:
        nnunet_trainer = get_trainer_from_args(
            dataset_name_or_id,
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            use_compressed_data,
            device=device,
            record_commit_info=record_commit_info,
            previous_stage=previous_stage,
            arc=arc,
        )

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
            continue_training and only_run_validation
        ), f"Cannot set --c and --val flag at the same time. Dummy."

        maybe_load_checkpoint(
            nnunet_trainer, continue_training, only_run_validation, pretrained_weights
        )

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        nnunet_trainer.perform_actual_validation(export_validation_probabilities)
        return nnunet_trainer


def run_training_entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_id",
        default="515",
        type=str,
        help="Dataset name or ID to train with",
    )
    parser.add_argument(
        "--configuration",
        default="3d_lowres",
        type=str,
        choices=["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"],
        help="Configuration that should be trained",
    )
    parser.add_argument(
        "--fold",
        default="all",
        type=str,
        help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4.",
    )
    parser.add_argument(
        "--commit_info",
        default="DEBUG",
        type=str,
        help="description of this experiment",
    )
    parser.add_argument(
        "--previous_stage",
        default="###NOT IMPLEMENTED###",
        type=str,
        help="need to be specified if training in cascade model,the first stage output",
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans",
    )
    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="weather in debug mode, given = no debug, not given  = in debug",
    )
    parser.add_argument(
        "--arc",
        default="undeclared",  # undeclared
        type=str,
        help="weather to use custom networks",
    )
    parser.add_argument(
        "-pretrained_weights",
        type=str,
        required=False,
        default=None,
        help="[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only "
        "be used when actually training. Beta. Use with caution.",
    )
    parser.add_argument(
        "-num_gpus",
        type=int,
        default=1,
        required=False,
        help="Specify the number of GPUs to use for training",
    )
    parser.add_argument(
        "--use_compressed",
        default=False,
        action="store_true",
        required=False,
        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
        "data is much more CPU and (potentially) RAM intensive and should only be used if you "
        "know what you are doing",
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        required=False,
        help="[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted "
        "segmentations). Needed for finding the best ensemble.",
    )
    parser.add_argument(
        "--c",
        action="store_true",
        required=False,
        help="[OPTIONAL] Continue training from latest checkpoint",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.",
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and "
        "you dont want to flood your hard drive with checkpoints.",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the training should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!",
    )
    args = parser.parse_args()

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    if args.device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")
    if args.configuration == "3d_cascade_fullres":
        assert (
            args.previous_stage != "###NOT IMPLEMENTED###"
        ), "previous_stage output need to be specified in cacade mode"
    commit_info = args.commit_info
    if args.no_debug:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        record_commit_info = [
            git_commit(project_root, commit_info=commit_info),
            commit_info,
            True,
        ]
    else:
        record_commit_info = [commit_info, commit_info, False]
    setproctitle.setproctitle(commit_info)
    run_training(
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        args.tr,
        args.p,
        args.pretrained_weights,
        args.num_gpus,
        args.use_compressed,
        args.npz,
        args.c,
        args.val,
        args.disable_checkpointing,
        device=device,
        record_commit_info=record_commit_info,
        previous_stage=args.previous_stage,
        arc=args.arc,
    )

    # if nnunet_trainer:
    #     nnunet_trainer.print_to_log_file("\n\n\n>>>> train done start to predict")
    # predict_img_dir = glob.glob(
    #     os.environ.get("nnUNet_raw")
    #     + "/Dataset*"
    #     + args.dataset_name_or_id
    #     + "*/imagesTs"
    # )[0]
    # predict_img_gt_dir = predict_img_dir.replace("imagesTs", "labelTs")

    # predict_flag = predict_entry_point_function(
    #     i=predict_img_dir,
    #     o=nnunet_trainer.output_folder,
    #     d=args.dataset_name_or_id,
    #     c=args.c,
    #     f=args.fold,
    #     nnunet_trainer=nnunet_trainer,
    # )
    # if predict_flag:
    #     if nnunet_trainer:
    #         nnunet_trainer.print_to_log_file(
    #             "\n\n\n >>>>predict done start to evaluate"
    #         )
    #     evaluate_folder_entry_point_function(
    #         pred_folder=nnunet_trainer.output_folder,
    #         gt_folder=predict_img_gt_dir,
    #         nnunet_trainer=nnunet_trainer,
    #     )


if __name__ == "__main__":
    run_training_entry()
