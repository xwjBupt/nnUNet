import pickle
import os.path as osp
import glob
from tqdm import tqdm
from git import Repo
import glob
import yaml
import shutil
import json
from collections import defaultdict
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
import os
import csv


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def flush(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "  [ %s ] [ avg : %.4f ] [ meidan : %.4f ]\n"
                % (name, meter.global_avg, meter.median)
            )
        return self.delimiter.join(loss_str)

    def lineout(self):
        loss_str = []
        loss_str.append("[")
        for name, meter in self.meters.items():
            loss_str.append(" %s:%.4f " % (name, meter.global_avg))
        loss_str.append("]")
        return self.delimiter.join(loss_str)

    def return_dict(self, method=None):
        output = {}
        if method:
            output.update(method)
        for name, meter in self.meters.items():
            output[name] = meter.global_avg
        return output


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


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
        commit_info
        + "\n"
        + "COMMIT BRANCH >>> "
        + branch
        + " <<< \n"
        + "COMMIT ID >>> "
        + cid
        + " <<<"
    )
    record_commit_info = " COMMIT TAG [\n%s]\n" % commit_tag
    return record_commit_info


def iter_wandb_log(wandb, metric, phase="Train", index=1):
    towrite = {"index": index}
    for k, v in metric.meters.items():
        towrite[phase + "_" + k] = v.global_avg
    wandb.log(towrite)


def write_to_csv(filename, content):
    file_exist = os.path.exists(filename)
    with open(filename, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        if not file_exist:
            writer.writeheader()
        writer.writerow(content)
