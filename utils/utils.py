import os
import yaml
import numpy as np
import pandas as pd
import torch
import random
from sklearn.preprocessing import LabelEncoder


def create_files(path_list):
    """Creates non-existing files"""

    for path in path_list:
        if os.path.exists(path) is False:
            os.makedirs(path)


def box_print(name):
    print("#" * (len(name) + 4))
    print("# " + name + " #")
    print("#" * (len(name) + 4))


def save_logs(logs, path):
    """Saves the config dict in yaml format"""

    with open(path + "/config_logs.yaml", "w") as output_file:
        yaml.dump(logs, output_file)


def get_lr(optimizer):
    """Returns current learning rate of optimizer"""

    for param_group in optimizer.param_groups:
        return param_group["lr"]


def flatten_dict(d, sep="/"):
    """Returns unnested dictionary"""

    df = pd.json_normalize(d, sep=sep)
    return df.to_dict(orient="records")[0]


def merge_dict_list(dict_list):
    result_dict = {}
    for key in dict_list[0].keys():
        result_dict[key] = [stats[key] for stats in dict_list]

    return result_dict


def save_model(epoch, name, model, optimizer, path, stats):
    """Saves the state dict of the current model and optimizer"""

    state_dict = model.state_dict()

    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save(
        {
            "epoch": epoch,
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "stats": stats,
        },
        path + "/{}.ckpt".format(name),
    )  # , _use_new_zipfile_serialization=False)


def set_model_params(requires_grad, *model_list):
    for model in model_list:
        for p in model.parameters():
            p.requires_grad = requires_grad


def encode_labels(y):
    encoder = LabelEncoder()
    y_new = encoder.fit_transform(["".join(str(l)) for l in y])
    return y_new, encoder


def get_device(device_name="cuda"):
    if device_name == "cuda":
        assert torch.cuda.is_available()

    return torch.device(device_name)


def get_predictions(output):
    output = torch.sigmoid(output)
    return output


def set_seed(SEED):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_results_config(results_path):
    print(results_path)
    results_config = {
        "results_path": results_path,
        "checkpoints_path": results_path + "/checkpoints",
        "logs_path": results_path + "/logs",
        "tensorboard_path": results_path + "/tensorboard",
        "sample_path": results_path + "/samples",
    }
    return results_config
