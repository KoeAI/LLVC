This repo contains the code necessary to reproduce PAPER LINK HERE, as well as train new LLVC models on a custom dataset. Demo here: https://koeai.github.io/llvc-demo/

# Setup
1. Create a Python virtual environment (this repo is tested with Python 3.11.4)
2. Install torch and torchaudio following https://pytorch.org/get-started/locally/ 
3. `pip install -r requirements.txt`
4. `python download_models.py`
5. `eval.py` has requirements that conflict with `requirements.txt`, so before running this file, create a seperate new Python virtual environment with python 3.9 and run `pip install -r eval_requirements.txt`

You should now be able to run `python infer.py` and convert all of the files in `test_wavs` with the pretrained llvc checkpoint, with the resulting files saved to `converted_out`.

# Inference
`python infer.py -p my_checkpoint.pth -c my_config.pth -f input_file -o my_out_dir` will convert a single audio file or folder of audio files using the given LLVC checkpoint and save the output to the folder `my_out_dir`. The `-s` argument simulate a streaming environment for conversion. The `-n` argument allows the user to specify the size of input audio chunks in streaming mode, trading increased latency for better RTF.

`compare_infer.py` allows you to reproduce our streaming no-f0 RVC and QuickVC conversions on input audio of your choice. By default, `window_ms` and `extra_convert_size` are set to the values used for no-f0 RVC conversion. See the linked paper for the QuickVC conversion parameters.

# Training
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

## Start a training run
1. Create a folder `experiments/my_run` containing a `config.json` (see `experiments/llvc/config.json` for an example)
2. Edit the `config.json` to reflect the location of your dataset and desired architectural modifications
3. `python train.py -d experiments/my_run`
4. The run will be logged to Tensorboard in the directory `experiments/my_run/logs`

## Evaluate results
1. Download test-clean.tar.gz from https://www.openslr.org/12
2. Use `infer.py` to convert the test-clean folder using the checkpoint that you want to evaluate
3. Activate the eval environment and run `eval.py` on your converted audio and directory of ground-truth audio files.

# Credits
Many of the modules written in `minimal_rvc/` are based on the following repositories:
- https://github.com/ddPn08/rvc-webui
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/teftef6220/Voice_Separation_and_Selection
