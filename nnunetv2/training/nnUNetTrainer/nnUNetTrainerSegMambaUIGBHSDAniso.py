from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMambaUIG import nnUNetTrainerSegMambaUIG


class nnUNetTrainerSegMambaUIGBHSDAniso(nnUNetTrainerSegMambaUIG):
    default_union_loss_weight = 0.1
    default_intersection_loss_weight = 0.1
