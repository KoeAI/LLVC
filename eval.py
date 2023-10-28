from resemblyzer import VoiceEncoder, preprocess_wav
from wvmos import get_wvmos
import argparse
from tqdm import tqdm
import numpy as np
from utils import glob_audio_files
import librosa
import os
import soundfile as sf

def resemblyze_score(gt_dir, converted_dir):
    encoder = VoiceEncoder()
    # cast gt_dir to path

    # glob recursively
    gt_paths = glob_audio_files(gt_dir)
    gt_wavs = [preprocess_wav(wav_fpath) for wav_fpath in
               tqdm(gt_paths, "Preprocessing gt files", len(gt_paths))]
    gt_embs = np.array([encoder.embed_utterance(wav) for wav in
                        tqdm(gt_wavs, "Embedding gt files", len(gt_wavs))])
    converted_paths = glob_audio_files(converted_dir)
    converted_wavs = [preprocess_wav(wav_fpath) for wav_fpath in
                      tqdm(converted_paths, "Preprocessing converted files", len(converted_paths))]
    converted_embs = np.array([encoder.embed_utterance(wav) for wav in
                               tqdm(converted_wavs, "Embedding converted files", len(converted_wavs))])
    scores = (gt_embs @ converted_embs.T).mean(axis=0)
    return scores

def wvmos_score(model, dir):
    wvmos_list = []
    print(f"Calculating wvmos for {dir}")
    for paths in tqdm(glob_audio_files(dir), "Calculating wvmos", len(glob_audio_files(dir))):
        wvmos_list.append(model.calculate_one(paths))
    return wvmos_list

def make_gt_subset(gt_dir):
    print("Creating gt subset")
    SECONDS = 10
    NUM_SUBSETS = 10
    gt_files = glob_audio_files(gt_dir)
    subset_dir = f"{SECONDS}s_{NUM_SUBSETS}"
    os.makedirs(subset_dir, exist_ok=True)
    for gt_file in gt_files:
        audio_data = librosa.load(gt_file, sr=16000)[0]
        if len(audio_data) > 16000 * SECONDS:
            audio_subsets = np.array_split(audio_data, len(audio_data) // (16000 * SECONDS))
            subset_count = min(NUM_SUBSETS, len(audio_subsets))
            for i in range(subset_count):
                out_fname = os.path.join(subset_dir, f"{os.path.basename(gt_file)}_{i}.wav")
                sf.write(out_fname, audio_subsets[i], 16000)
    print(f"Attempted to create {NUM_SUBSETS} subsets of {SECONDS} seconds each for each audio file in {subset_dir}")
    return subset_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--converted_dir", type=str, required=True)
    parser.add_argument("--eval_gt", action="store_true")
    args = parser.parse_args()
    subset_dir = make_gt_subset(args.gt_dir)
    wvmos_model = get_wvmos(cuda=True)
    resemblyzer_scores = resemblyze_score(subset_dir, args.converted_dir)
    converted_wvmos_scores = wvmos_score(wvmos_model, args.converted_dir)
    if args.eval_gt:
        gt_wvmos_scores = wvmos_score(wvmos_model, subset_dir)
        gt_resemblyzer_scores = resemblyze_score(subset_dir, subset_dir)
        print(f"GT WVMOS score: {np.array(gt_wvmos_scores).mean():.3f}")
        print(f"GT Resemblyzer score: {gt_resemblyzer_scores.mean():.3f}")
    print(f"WVMOS score: {np.array(converted_wvmos_scores).mean():.3f}")
    print(f"Resemblyzer score: {resemblyzer_scores.mean():.3f}")

if __name__ == "__main__":
    main()
        

