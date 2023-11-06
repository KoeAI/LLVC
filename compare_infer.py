import torch
import numpy as np
import librosa
import time
from tqdm import tqdm
import argparse
from scipy.io.wavfile import write
from minimal_rvc.models import SynthesizerTrnMs256NSFSidNono
from minimal_rvc.pipeline import VocalConvertPipeline
from minimal_quickvc.models import SynthesizerTrn
from minimal_quickvc.utils import load_checkpoint
from fairseq import checkpoint_utils
import scipy.signal
import json
import os
from utils import glob_audio_files


def init_model(model_type):
    if model_type == 'rvc':
        model_path = "llvc_models/models/rvc_no_f0/f_8312_no_f0-300.pth"
        state_dict = torch.load(model_path, map_location="cpu")
        state_dict["params"]["spk_embed_dim"] = state_dict["weight"][
            "emb_g.weight"
        ].shape[0]
        if not "emb_channels" in state_dict["params"]:
            state_dict["params"]["emb_channels"] = 768  # for backward compat.
        model = SynthesizerTrnMs256NSFSidNono(
            **state_dict["params"], is_half=False
        ).eval().to('cpu')
        model.load_state_dict(state_dict["weight"], strict=False)
    elif model_type == 'quickvc':
        model = SynthesizerTrn().eval().to('cpu')
        model_path = 'llvc_models/models/quickvc/quickvc_100.pth'
        _ = load_checkpoint(model_path, model, None)
    return model


def load_hubert(model_type):
    if model_type == 'rvc':
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ['llvc_models/models/embeddings/checkpoint_best_legacy_500.pt'],
            suffix="",
        )
        embedder_model = models[0]
        embedder_model = embedder_model.to('cpu')
    if model_type == 'quickvc':
        embedder_model = torch.hub.load(
            "bshall/hubert:main", "hubert_soft").eval().to('cpu')
    return embedder_model


def calc_RTF(audio, sr, conversion_time):
    return librosa.get_duration(y=audio, sr=sr) / conversion_time


class ChunkedInferer:
    def __init__(
        self, window_ms, crossfade_overlap, extra_convert_size, model_type, sr_out
    ):
        if model_type == 'rvc':
            self.pipeline = VocalConvertPipeline(
                sr_out, 'cpu', False, no_pad=True)
        self.sr_out = sr_out
        self.audio_buffer = None
        self.conv_sr = 16000
        self.sola_search_frame = int(0.012 * self.conv_sr)
        self.crossfade_overlap = crossfade_overlap
        self.crossfade_offset_rate = 0.0
        self.crossfade_end_rate = 1.0
        self.generate_strength()
        self.chunk_len = int(window_ms * self.conv_sr / 1000)
        self.block_len = int(self.chunk_len / self.conv_sr * self.sr_out)
        self.extra_convert_size = extra_convert_size
        self.model = init_model(model_type)
        self.hubert = load_hubert(model_type)
        self.model_type = model_type

    def clear_buffers(self):
        self.audio_buffer = None
        del self.sola_buffer

    def generate_strength(self):
        cf_offset = int(self.crossfade_overlap * self.crossfade_offset_rate)
        cf_end = int(self.crossfade_overlap * self.crossfade_end_rate)
        cf_range = cf_end - cf_offset
        percent = np.arange(cf_range) / cf_range

        np_prev_strength = np.cos(percent * 0.5 * np.pi) ** 2
        np_cur_strength = np.cos((1 - percent) * 0.5 * np.pi) ** 2

        self.np_prev_strength = np.concatenate(
            [
                np.ones(cf_offset),
                np_prev_strength,
                np.zeros(self.crossfade_overlap -
                         cf_offset - len(np_prev_strength)),
            ]
        )
        self.np_cur_strength = np.concatenate(
            [
                np.zeros(cf_offset),
                np_cur_strength,
                np.ones(self.crossfade_overlap -
                        cf_offset - len(np_cur_strength)),
            ]
        )

    def generate_frame(self, chunk):
        if self.audio_buffer is not None:
            self.audio_buffer = np.concatenate([self.audio_buffer, chunk], 0)
        else:
            self.audio_buffer = chunk
        convert_size = (
            self.block_len
            + self.crossfade_overlap
            + self.sola_search_frame
            + self.extra_convert_size
        )
        convert_offset = -1 * convert_size
        self.audio_buffer = self.audio_buffer[convert_offset:]
        return self.audio_buffer

    def postprocess(self, audio):
        if hasattr(self, "sola_buffer") is True:
            np.set_printoptions(threshold=10000)
            audio_offset = -1 * (
                self.sola_search_frame + self.crossfade_overlap + self.block_len
            )
            audio = audio[audio_offset:]
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC, https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI
            cor_nom = np.convolve(
                audio[: self.crossfade_overlap + self.sola_search_frame],
                np.flip(self.sola_buffer),
                "valid",
            )
            cor_den = np.sqrt(
                np.convolve(
                    audio[: self.crossfade_overlap +
                          self.sola_search_frame] ** 2,
                    np.ones(self.crossfade_overlap),
                    "valid",
                )
                + 1e-3
            )
            sola_offset = int(np.argmax(cor_nom / cor_den))
            sola_end = sola_offset + self.block_len
            output_wav = audio[sola_offset:sola_end].astype(np.float64)
            output_wav[: self.crossfade_overlap] *= self.np_cur_strength
            output_wav[: self.crossfade_overlap] += self.sola_buffer[:]
            result = output_wav.astype(np.int16)

        else:
            # print("[Voice Changer] no sola buffer. (You can ignore this.)")
            result = np.zeros(4096).astype(np.int16)

        if (
            hasattr(self, "sola_buffer") is True
            and sola_offset < self.sola_search_frame
        ):
            offset = -1 * (
                self.sola_search_frame + self.crossfade_overlap - sola_offset
            )
            end = -1 * (self.sola_search_frame - sola_offset)
            sola_buf_org = audio[offset:end]
            self.sola_buffer = sola_buf_org * self.np_prev_strength
        else:
            self.sola_buffer = audio[-self.crossfade_overlap:] * \
                self.np_prev_strength
        return result

    def vc_fn(self, audio):
        with torch.no_grad():
            wav_src = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
            if self.model_type == 'rvc':
                audio = self.pipeline(
                    self.hubert,
                    12,
                    self.model,
                    0,
                    audio,
                    12,
                    None,
                    None,
                    0,
                    0,
                    True
                )
            else:
                mel_tgt = torch.zeros(1, 80, 64)
                c = self.hubert.units(wav_src)
                c = c.transpose(2, 1)
                audio = self.model.infer(c, mel=mel_tgt)
                audio = audio.squeeze(0).squeeze(0).cpu().numpy()
                audio = (audio * 32767).astype(np.int16)
            return audio

    def file_infer(self, fname):
        # load audio
        audio, sr = librosa.load(fname, sr=self.conv_sr)
        # split audio into chunks of len window_len
        audio = audio[: len(audio) // self.chunk_len * self.chunk_len]
        chunks = np.split(audio, len(audio) // self.chunk_len)
        converted_chunks = []
        times = []
        for chunk in chunks:
            start = time.time()
            in_data = self.generate_frame(chunk)
            audio_out = self.vc_fn(in_data)
            postprocessed = self.postprocess(audio_out)
            converted_chunks.append(postprocessed)
            times.append(time.time() - start)
        # concatenate chunks
        out = np.concatenate(converted_chunks)
        conversion_time = sum(times)
        rtf = calc_RTF(audio, sr, conversion_time)
        avg_time = np.mean(times)
        return out, rtf, avg_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", "-f", type=str,
                        default="test_wavs", help="audio file to convert")
    parser.add_argument("--model_type", "-m", type=str,
                        default="rvc", help="model type: rvc or quickvc")
    parser.add_argument("--out_dir", "-o", type=str,
                        default="converted_out", help="directory to save converted files")
    parser.add_argument(
        "--window_ms",
        type=int,
        default=100,
        help="audio frame latency in milliseconds",
    )
    parser.add_argument(
        "--crossfade_overlap",
        type=int,
        default=256,
        help="overlap between contiguous frames",
    )
    parser.add_argument(
        "--extra_convert_size",
        type=int,
        default=1024,
        help="extra audio frame size to convert",
    )
    args = parser.parse_args()
    sr_out = 32000 if args.model_type == 'rvc' else 16000
    inferer = ChunkedInferer(
        args.window_ms,
        args.crossfade_overlap,
        args.extra_convert_size,
        args.model_type,
        sr_out
    )
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    # check if fname is a directory
    if os.path.isdir(args.fname):
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        # recursively glob wav files
        fnames = glob_audio_files(args.fname)
        rtf_list = []
        avg_times_list = []
        for fname in tqdm(fnames):
            out, rtf_, avg_time_ = inferer.file_infer(
                fname)
            rtf_list.append(rtf_)
            avg_times_list.append(avg_time_)
            out_fname = os.path.join(args.out_dir, os.path.basename(fname))
            write(out_fname, sr_out, out)
            inferer.clear_buffers()
        rtf = np.mean(rtf_list)
        avg_time = np.mean(avg_times_list)

    else:
        out, rtf, avg_time = inferer.file_infer(
            args.fname)

        # split extension from fname
        out_fname = os.path.join(args.out_dir, os.path.basename(args.fname))
        write(out_fname, sr_out, out)
    print(f"RTF: {rtf:.3f}")
    e2e_latency = args.window_ms + avg_time * 1000
    print(f"End-to-end latency: {e2e_latency:.3f}ms")


if __name__ == "__main__":
    main()
