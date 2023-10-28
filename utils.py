"""A collection of useful helper functions"""

import os
import json
import torch
import glob
import auraloss
import librosa
import numpy as np
import torch.nn.functional as F
from torchaudio.transforms import MFCC
from mel_processing import mel_spectrogram_torch


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    filelist = glob.glob(os.path.join(dir_path, regex))
    if len(filelist) == 0:
        return None
    filelist.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    filepath = filelist[-1]
    return filepath


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=True):
    assert os.path.isfile(
        checkpoint_path), f"Checkpoint '{checkpoint_path}' not found"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint_dict["model"])
    else:
        model.load_state_dict(checkpoint_dict["model"])
    epoch = checkpoint_dict["epoch"]
    step = checkpoint_dict["step"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        print(f"Loaded optimizer with learning rate {learning_rate}")
    print("Loaded checkpoint '{}' (epoch {}, step {})".format(
        checkpoint_path, epoch, step))
    return model, optimizer, learning_rate, epoch, step


def save_state(model, optimizer, learning_rate, epoch, step, checkpoint_path):
    print(
        "Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "epoch": epoch,
            "step": step,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def model_size(model):
    """
    Returns size of the `model` in millions of parameters.
    """
    num_train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    params_scaled = num_train_params / 1e6
    # round to 2 decimal places
    return round(params_scaled, 2)


def format_lr_info(optimizer):
    lr_info = ""
    for i, pg in enumerate(optimizer.param_groups):
        lr_info += " {group %d: params=%.5fM lr=%.1E}" % (
            i, sum([p.numel() for p in pg['params']]) / (1024 ** 2), pg['lr'])
    return lr_info


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def multires_loss(output, gt, sr, params):
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        scale="mel",
        sample_rate=sr,
        device=output.device,
        **params
    )
    return loss_fn(output, gt)


def aux_mel_loss(output, gt, config):
    sr = config['data']['sr']
    aux_mel_loss_type = config['aux_mel']['type']
    config_params = config['aux_mel']['params']
    if aux_mel_loss_type == "multires":
        param_dict = {}
        config_params = config['aux_mel']['params']
        param_dict['fft_sizes'] = config_params['n_fft']
        param_dict['hop_sizes'] = config_params['hop_size']
        param_dict['win_lengths'] = config_params['win_size']
        param_dict['n_bins'] = config_params['num_mels']
        return multires_loss(output, gt, sr, param_dict)
    elif aux_mel_loss_type == "rvc":
        param_dict = config_params
        for k in param_dict:
            if isinstance(param_dict[k], list):
                param_dict[k] = param_dict[k][0]
        gt_mel = mel_spectrogram_torch(
            gt.float().squeeze(1),
            sr,
            **param_dict
        )
        output_mel = mel_spectrogram_torch(
            output.float().squeeze(1),
            sr,
            **param_dict
        )
        loss_mel = F.l1_loss(
            output_mel, gt_mel)
        return loss_mel
    else:
        raise ValueError(f"Unknown aux mel loss type, {aux_mel_loss_type}")


def mcd(predicted_audio, gt_audio, sr):
    mfcc = MFCC(sample_rate=sr).to(predicted_audio.device)
    predicted_mfcc = mfcc(predicted_audio)
    gt_mfcc = mfcc(gt_audio.to(predicted_audio.device))
    return torch.mean(torch.abs(predicted_mfcc - gt_mfcc))


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=16000,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.avg = 0

    def update(self, val):
        self.avg = (self.avg * self.n + val) / (self.n + 1)
        self.n += 1

    def reset(self):
        self.n = 0
        self.avg = 0

    def __call__(self):
        return self.avg


def load_wav_to_torch(full_path, sr):
    data = librosa.load(full_path, sr=sr)[0]
    return torch.FloatTensor(data.astype(np.float32))


def fairseq_loss(output, gt, fairseq_model):
    """
    fairseq feature mse loss, based on https://arxiv.org/abs/2301.04388
    """
    gt = gt.squeeze(1)
    output = output.squeeze(1)
    gt_f = fairseq_model.feature_extractor(gt)
    output_f = fairseq_model.feature_extractor(output)
    mse_loss = F.mse_loss(gt_f, output_f)
    return mse_loss


def glob_audio_files(dir):
    ext_list = ["wav", "mp3", "flac"]
    audio_files = []
    for ext in ext_list:
        audio_files.extend(glob.glob(
            os.path.join(dir, f"**/*.{ext}"), recursive=True))
    return audio_files
