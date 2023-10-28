"""
Torch dataset object for synthetically rendered spatial data.
"""

import os
import torch
from scipy.io.wavfile import read
import os
import glob


def get_dataset(dir):
    original_files = glob.glob(os.path.join(dir, "*_original.wav"))
    converted_files = []
    for original_file in original_files:
        converted_file = original_file.replace(
            "_original.wav", "_converted.wav")
        converted_files.append(converted_file)
    return original_files, converted_files


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


class LLVCDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dir,
        sr,
        wav_len,
        dset
    ):
        assert dset in [
            "train",
            "val",
            "dev"
        ], "`dset` must be one of ['train', 'val', 'dev']"
        self.dset = dset
        file_dir = os.path.join(dir, dset)
        self.wav_len = wav_len
        self.sr = sr
        self.original_files, self.converted_files = get_dataset(
            file_dir
        )

    def __len__(self):
        return len(self.original_files)

    def __getitem__(self, idx):
        original_wav = self.original_files[idx]
        converted_wav = self.converted_files[idx]

        original_data, o_sr = load_wav(original_wav)
        converted_data, c_sr = load_wav(converted_wav)

        assert o_sr == self.sr, f"Expected {self.sr}Hz, got {o_sr}Hz for file {original_wav}"
        assert c_sr == self.sr, f"Expected {self.sr}Hz, got {c_sr}Hz for file {converted_wav}"

        converted = torch.from_numpy(original_data)
        gt = torch.from_numpy(converted_data)

        converted = converted.unsqueeze(0).to(torch.float) / 32768
        gt = gt.unsqueeze(0).to(torch.float) / 32768

        if gt.shape[-1] < self.wav_len:
            gt = torch.cat(
                (gt, torch.zeros(1, self.wav_len - gt.shape[-1])), dim=1)
        else:
            gt = gt[:, : self.wav_len]

        if converted.shape[-1] < self.wav_len:
            converted = torch.cat(
                (converted, torch.zeros(1, self.wav_len - converted.shape[-1])), dim=1
            )
        else:
            converted = converted[:, : self.wav_len]

        return converted, gt
