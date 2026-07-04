from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMambaUI import nnUNetTrainerSegMambaUI


class nnUNetTrainerSegMambaUIBHSDAniso(nnUNetTrainerSegMambaUI):
    default_union_loss_weight = 0.1
    default_intersection_loss_weight = 0.1
