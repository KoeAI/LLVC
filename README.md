# LLVC: **L**ow-Latency **L**ow-Resource **V**oice **C**onversion
This repository contains the code necessary to train Koe AI's LLVC models and to reproduce the LLVC paper.

LLVC paper: https://koe.ai/papers/llvc.pdf

LLVC samples: https://koeai.github.io/llvc-demo/

Windows executable: https://koe.ai/recast/download/

Koe AI homepage: https://koe.ai/

## Setup
1. Create a Python environment with e.g. conda: `conda create -n llvc python=3.11`
2. Activate the new environment: `conda activate llvc`
3. Install torch and torchaudio from https://pytorch.org/get-started/locally/ 
4. Install requirements with `pip install -r requirements.txt`
5. Download models with `python download_models.py`
6. `eval.py` has requirements that conflict with `requirements.txt`, so before running this file, create a seperate new Python virtual environment with python 3.9 and run `pip install -r eval_requirements.txt`

You should now be able to run `python infer.py` and convert all of the files in `test_wavs` with the pretrained llvc checkpoint, with the resulting files saved to `converted_out`.

## Inference
`python infer.py -p my_checkpoint.pth -c my_config.pth -f input_file -o my_out_dir` will convert a single audio file or folder of audio files using the given LLVC checkpoint and save the output to the folder `my_out_dir`. The `-s` argument simulate a streaming environment for conversion. The `-n` argument allows the user to specify the size of input audio chunks in streaming mode, trading increased latency for better RTF.

`compare_infer.py` allows you to reproduce our streaming no-f0 RVC and QuickVC conversions on input audio of your choice. By default, `window_ms` and `extra_convert_size` are set to the values used for no-f0 RVC conversion. See the linked paper for the QuickVC conversion parameters.

## Training
1. Create a folder `experiments/my_run` containing a `config.json` (see `experiments/llvc/config.json` for an example)
2. Edit the `config.json` to reflect the location of your dataset and desired architectural modifications
3. `python train.py -d experiments/my_run`
4. The run will be logged to Tensorboard in the directory `experiments/my_run/logs`

## Dataset
Datasets are comprised of a folder containing three subfolders: `dev`, `train` and `val`. Each of these folders contains audio files of the form `PREFIX_original.wav`, which are audio clips recorded by a variety of input speakers, and `PREFIX_converted.wav`, which are the original audio clips converted to a single target speaker. `val` contains clips from the same speakers as `test`. `dev` contains clips from different speakers than `test`. 

To recreate the dataset that we use in our paper:
1. Download dev-clean.tar.gz and train-clean-360.tar.gz from https://www.openslr.org/12 and unzip to `llvc/LibriSpeech`
2. 
```
python -m minimal_rvc._infer_folder \
                                    --train_set_path "LibriSpeech/train-clean-360" \
                                    --dev_set_path "LibriSpeech/dev-clean" \
                                    --out_path "f_8312_ls360" \
                                    --flatten \
                                    --model_path "llvc_models/models/rvc/f_8312_32k-325.pth" \
                                    --model_name "f_8312" \
                                    --target_sr 16000 \
                                    --f0_method "rmvpe" \
                                    --val_percent 0.02 \
                                    --random_seed 42 \
                                    --f0_up_key 12
```
## Evaluate results
1. Download test-clean.tar.gz from https://www.openslr.org/12
2. Use `infer.py` to convert the test-clean folder using the checkpoint that you want to evaluate
3. Activate the eval environment and run `eval.py` on your converted audio and directory of ground-truth audio files.

## Credits
Many of the modules written in `minimal_rvc/` are based on the following repositories:
- https://github.com/ddPn08/rvc-webui
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/teftef6220/Voice_Separation_and_Selection


## Citation
If you find out work relevant to your research, please cite:
```
@misc{sadov2023lowlatency,
      title={Low-latency Real-time Voice Conversion on CPU}, 
      author={Konstantine Sadov and Matthew Hutter and Asara Near},
      year={2023},
      eprint={2311.00873},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
