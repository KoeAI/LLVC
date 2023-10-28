from .model import VoiceConvertModel
import sys
import logging
from pydub import AudioSegment
from tqdm import tqdm
from typing import Optional
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import os
import shutil
import argparse
import concurrent.futures
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


def flatten_LibriSpeech(root_path: Path | str,
                        flattened_path: Path | str,
                        overwrite_flattened: bool = False,
                        copy_dataset: bool = True) -> None:
    """
    Given a root directory containing the LibriSpeech dataset, flatten the directory structure from 
    root/SPEAKER/BOOK/SPEAKER-CHAPTER-PART.flac to root/SPEAKER/SPEAKER-CHAPTER-PART.flac

    Args:
        root_path (Path|str): Path to the root directory of the LibriSpeech dataset
        flattened_path (Path|str): Path to the directory where the flattened dataset will be saved
        overwrite_flattened (bool): If True, overwrite the flattened dataset if it already exists
        copy_dataset (bool): If True, copy the dataset to the flattened directory. If False, move the dataset (better for space restrictions)

    Returns:
        None

    Raises:
        FileNotFoundError: If root_path does not exist
        FileExistsError: If flattened_path already exists and overwrite_flattened is False
    """

    root_path = Path(root_path)
    flattened_path = Path(flattened_path)

    # Safety checks
    if flattened_path.exists() and not overwrite_flattened:
        raise FileExistsError(
            f'{flattened_path} already exists. Set overwrite_flattened=True to overwrite')
    if not root_path.exists():
        raise FileNotFoundError(f'{root_path} does not exist')

    for speaker in tqdm(list(root_path.glob('*'))):
        for flac in speaker.glob('**/*.flac'):
            path, filename = os.path.split(flac)
            new_path = flattened_path / speaker.name
            new_path.mkdir(parents=True, exist_ok=True)
            if copy_dataset:
                shutil.copy(flac, new_path / filename)
            else:
                flac.rename(new_path / filename)


def resample_folder(root_path: Path | str,
                    target_sr: int,
                    overwrite_resampled_og: bool = False,
                    resampled_path: Optional[Path | str] = None) -> None:
    """
    Given a folder of wav or flac files, resample all files to a target sampling rate

    Args:
        root_path (str|Path): Path to the folder containing wav or flac files
        target_sr (int): Target sampling rate
        overwrite_resampled_og (bool): If True, overwrite original resampled files if they already exist
        resampled_path (str|Path): Path to the folder where the resampled files will be saved. If None, resampled files will be saved in the same directory as the wav files

    Returns:
        None

    Raises:
        FileNotFoundError: If root_path does not exist
        ValueError: If root_path is a file
    """

    root_path = Path(root_path)

    # Safety checks
    if not root_path.exists():
        raise FileNotFoundError(f'{root_path} does not exist')
    if root_path.is_file():
        raise ValueError(f'{root_path} is a file. Please provide a folder')

    # if resampled_path is None, save the resampled files in the same directory as the wav files
    if resampled_path is None:
        resampled_path = root_path
    else:
        resampled_path = Path(resampled_path)

    if not resampled_path.exists():
        resampled_path.mkdir(parents=True, exist_ok=True)

    files_to_resample = []
    for wav in list(root_path.glob('**/*.wav')) + list(root_path.glob('**/*.flac')):
        # get the extension and create the resampled path
        extension = wav.name.split('.')[-1]

        # do not resample if the file is already resampled unless overwrite_resampled is True
        if ('_original' in wav.name) and (not overwrite_resampled_og):
            continue

        if "_converted" in wav.name:
            resampled_wav = resampled_path / wav.name

        else:
            resampled_wav = resampled_path / \
                wav.name.replace(f'.{extension}', f'_{target_sr}_original.wav')

        # this is the case where the file is already resampled and is not contained in the root_path
        if resampled_wav.exists() and not overwrite_resampled_og and not "_converted" in wav.name:
            continue

        # add the file to the list of files to resample
        files_to_resample.append((wav, resampled_wav, target_sr))

    # resample the files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        x = list(tqdm(executor.map(resample_file, files_to_resample),
                 total=len(files_to_resample)))


def resample_file(args):
    wav_path, out_path, target_sr = args
    y, sr = librosa.load(wav_path, sr=target_sr)
    sf.write(out_path, y, sr)
    return True


def convert_folder(root_path: Path | str,
                   model_path: Path | str,
                   model_name: str,
                   f0_up_key: int,
                   f0_method: str = 'rmvpe',
                   overwrite_converted: bool = False,
                   converted_path: Optional[Path | str] = None) -> None:
    """
    Given a folder of wav or flac files, convert all files to a target sampling rate using a VoiceConvertModel

    Args:
        root_path (Path|str): Path to the folder containing wav or flac files
        converted_path (Path|str): Path to the folder where the converted files will be saved. If None, converted files will be saved in the same directory as the wav files
        model_path (Path|str): VoiceConvertModel to use for conversion
        model_name (str): Name to be added to output filenames
        f0_up_key (int): Pitch adjust for conversion
        f0_method (str): f0 method for pitch extraction. Default is 'rmvpe'
        overwrite_converted (bool): If True, overwrite converted files if they already exist

    Returns:
        None

    Raises:
        FileNotFoundError: If root_path does not exist
        ValueError: If root_path is a file
    """
    model = VoiceConvertModel(
        model_name, torch.load(model_path, map_location="cpu"))

    root_path = Path(root_path)

    # Safety checks
    if not root_path.exists():
        raise FileNotFoundError(f'{root_path} does not exist')
    if root_path.is_file():
        raise ValueError(f'{root_path} is a file. Please provide a folder')

    # if converted_path is None, save the converted files in the same directory as the wav files
    if converted_path is None:
        converted_path = root_path
    else:
        converted_path = Path(converted_path)

    # (you like that walrus operator? pretty cool huh)
    for wav in (progress := tqdm(list(root_path.glob('**/*.wav')) + list(root_path.glob('**/*.flac')), unit="file")):

        # do not convert if the file is already converted unless overwrite_converted is True
        if ('_converted' in wav.name) and (not overwrite_converted):
            progress.set_description(
                f'{wav.name} is already converted. Skipping...')
            continue

        # never overwrite resampled files
        if '_original' not in wav.name:
            progress.set_description(f'{wav.name} not original. Skipping...')
            continue

        # get the extension and create the converted path
        extension = wav.name.split('.')[-1]
        converted_wav = converted_path / \
            wav.name.replace(f'.{extension}', f'_converted.wav').replace(
                '_original', '')

        # this is the case where the file is already converted and is not contained in the root_path
        if converted_wav.exists() and not overwrite_converted:
            progress.set_description(
                f'{converted_wav} already exists. Skipping...')
            continue

        # convert and save (explicitly typed to avoid misconstruing the audio type for something else)
        out: AudioSegment = model.single(sid=1,
                                         input_audio=str(wav),
                                         embedder_model_name='hubert_base',
                                         embedding_output_layer='auto',
                                         f0_up_key=f0_up_key,
                                         f0_file=None,
                                         f0_method=f0_method,
                                         auto_load_index=False,
                                         faiss_index_file=None,
                                         index_rate=None,
                                         f0_relative=True,
                                         output_dir='out')

        out.export(converted_wav, format="wav")


def val_split(root_path: Path | str,
              val_path: Path | str,
              val_percent: float,
              random_seed: int) -> None:
    """
    Given a folder of wav or flac files, split the files into a training and validation set

    Args:
        root_path (Path|str): Path to the folder containing wav or flac files
        val_path (Path|str): Path to the folder where the validation files will be saved
        val_percent (float): Percent of the dataset to use for validation
        random_seed (int): Random seed for splitting the dataset

    Returns:
        None

    """
    root_path = Path(root_path)
    val_path = Path(val_path)

    # Safety checks
    if not root_path.exists():
        raise FileNotFoundError(f'{root_path} does not exist')
    if root_path.is_file():
        raise ValueError(f'{root_path} is a file. Please provide a folder')

    if val_path.exists():
        return

    val_path.mkdir(parents=True, exist_ok=True)

    # list of all files in the root_path
    files = list(root_path.glob('**/*.wav')) + \
        list(root_path.glob('**/*.flac'))
    files = np.array(sorted(files, key=lambda x: x.stem))

    np.random.seed(random_seed)
    np.random.shuffle(files)

    # split the files into train and val
    val_files = files[:int(len(files)*val_percent)]

    # move the files to the val_path
    for file in tqdm(val_files, unit='file'):
        shutil.move(file, val_path/file.name)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root_path", help="Path to dataset to be processed", type=str, required=True, default='.LibriSpeech/')
    parser.add_argument("--train_set_path", help="Path to dataset to be processed",
                        type=str, default='./LibriSpeech/train-clean-100')
    parser.add_argument("--dev_set_path", help="path to the dev set",
                        type=str, default='./LibriSpeech/dev-clean/')
    parser.add_argument(
        '--flatten', help='Flatten LibriSpeech datasets before converting. This will also loop through each speaker in the dataset instead of a single folder', action='store_true')
    parser.add_argument(
        '--clean_flattened', help='Remove the flattened LibriSpeech dataset after processing TBA', action='store_true')
    parser.add_argument("--out_path", help="Path to save the new dataset folders",
                        type=str, required=True, default='./LibriSpeech_processed/')
    parser.add_argument(
        "--model_path", help="Path to RVC model to create `_converted` data", type=str)
    parser.add_argument(
        "--f0_up_key", help="Pitch adjust for conversion", type=int, default=12)
    parser.add_argument(
        "--f0_method", help="f0 method for pitch extraction", type=str)
    parser.add_argument("--model_name", help="Name to be added to output filenames. If `None`, the filename of --model_path will be used",
                        type=str, required=False, default=None)
    parser.add_argument(
        "--target_sr", help="Sample rate to resample the dataset into", type=int, default=16000)
    parser.add_argument(
        "--val_percent", help="Percent of the dataset to use for validation", type=float, default=0.1)
    parser.add_argument(
        "--random_seed", help="Random seed for splitting the dataset", type=int, default=42)
    args = parser.parse_args()

    # unpack, typecheck, and perform safety checks on args
    out_path = Path(args.out_path)
    model_path = Path(args.model_path)
    dev_set_path = Path(args.dev_set_path)
    train_set_path = Path(args.train_set_path)

    assert model_path.exists(), f'{model_path} does not exist'
    assert dev_set_path.exists(), f'{dev_set_path} does not exist'
    assert train_set_path.exists(), f'{train_set_path} does not exist'

    model_name: str = args.model_name
    if model_name is None:
        model_name = model_path.name.split('.')[0]

    target_sr: int = args.target_sr
    f0_up_key: int = args.f0_up_key
    val_percent: float = args.val_percent

    assert isinstance(target_sr, int), f'target_sr must be an integer'
    assert target_sr > 0, f'target_sr must be greater than 0'
    assert val_percent >= 0 and val_percent <= 1, f'val_percent must be between 0 and 1'

    clean_flattened: bool = args.clean_flattened  # TODO: implement this

    # flatten the LibriSpeech datasets
    if args.flatten:
        print('[INFER_FOLDER] Flattening LibriSpeech datasets')
        flatten_LibriSpeech(root_path=train_set_path,
                            flattened_path=out_path/'train-flatten',
                            overwrite_flattened=False,
                            copy_dataset=True)

        flatten_LibriSpeech(root_path=dev_set_path,
                            flattened_path=out_path/'dev-flatten',
                            overwrite_flattened=False,
                            copy_dataset=True)

        train_set_path = out_path/'train-flatten'
        dev_set_path = out_path/'dev-flatten'

    # split the train set into train and val
    val_split(root_path=train_set_path,
              val_path=out_path/'val-flatten',
              val_percent=val_percent,
              random_seed=args.random_seed)

    # create the resampled datasets
    print('[INFER_FOLDER] Resampling datasets')
    resample_folder(root_path=train_set_path,
                    resampled_path=out_path/'train',
                    target_sr=target_sr,
                    overwrite_resampled_og=False)

    resample_folder(root_path=dev_set_path,
                    resampled_path=out_path/'dev',
                    target_sr=target_sr,
                    overwrite_resampled_og=False)

    resample_folder(root_path=out_path/'val-flatten',
                    resampled_path=out_path/'val',
                    target_sr=target_sr,
                    overwrite_resampled_og=False)

    # convert the resampled datasets
    for folder in ['train', 'dev', 'val']:
        print(f'[INFER_FOLDER] Converting {folder}')
        convert_folder(root_path=out_path/folder,
                       model_path=model_path,
                       model_name=model_name,
                       f0_up_key=f0_up_key,
                       f0_method=args.f0_method,
                       overwrite_converted=False,
                       converted_path=out_path/folder)
        # resample the converted files to the target_sr
        print("[INFER_FOLDER] Resampling converted files")
        resample_folder(root_path=out_path/folder,
                        resampled_path=out_path/folder,
                        target_sr=target_sr,
                        overwrite_resampled_og=False)

    # print metrics
    print(
        f"Train set _original: {len(list((out_path/'train').glob('**/*_original.wav')))} files")
    print(
        f"Train set _converted: {len(list((out_path/'train').glob('**/*_converted.wav')))} files")
    print(
        f"Dev set _original: {len(list((out_path/'dev').glob('**/*_original.wav')))} files")
    print(
        f"Dev set _converted: {len(list((out_path/'dev').glob('**/*_converted.wav')))} files")
    print(
        f"Val set _original: {len(list((out_path/'val').glob('**/*_original.wav')))} files")
    print(
        f"Val set _converted: {len(list((out_path/'val').glob('**/*_converted.wav')))} files")


if __name__ == '__main__':
    main()
