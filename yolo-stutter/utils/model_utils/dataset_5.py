import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from queue import Empty
import librosa
import numpy as np

import os

from tqdm import tqdm

import json

import re

import random

import utils.vits.commons as commons
import utils.vits.utils as utils
from utils.vits.text import text_to_sequence
from utils.vits.utils import load_wav_to_torch
from utils.vits.mel_processing import spectrogram_torch


global_sampling_rate = 22050

# custom dataset used to load data
class Dataset(Dataset):
    def __init__(self, data_dir, hps, label_downsample_factor, skip_raw_label_comparison=False):
        self.data_dir = data_dir
        self.hps = hps
        self.label_downsample_factor = label_downsample_factor

        self.labels = ["rep", "block", "missing", "replace", "prolong"]

        text_files = os.listdir(os.path.join(self.data_dir, "gt_text"))
        label_files = os.listdir(os.path.join(self.data_dir, "disfluent_labels"))  # change

        # load data based on cleaned sample ids
        print("Loading sample filenames...")
        self.samples = []
       
        for file in tqdm(label_files):
            split_file = file.split("_")
            self.samples += [file]

        # for better partitioning
        random.seed(0)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # for parsing gt text
        split_file = self.samples[idx].split("_")

        # loading gt text
        with open(
            os.path.join(
                self.data_dir, "gt_text", split_file[0] + "_" + split_file[1] + ".txt"), "r", encoding='utf-8') as f:
                # self.data_dir, "gt_text", split_file[0] + "_" + split_file[1] + "_" + split_file[2] + "_" + split_file[3] + ".txt"), "r", encoding='utf-8') as f: # libritts
            try:
                raw_text = f.read()
            except Exception as e:
                print(f"Error loading data by Unicodeerror: {e}")
                return None 

        text = self.get_text(raw_text, self.hps)

        text_length = torch.tensor(text.size(0))

        # spec - [feature_dim, temporal_dim] -> temporal_dim is raw_audio_time (16k sampling freq) / 256 (hop_length)
        spec, wav = self.get_audio_a(
            os.path.join(
                self.data_dir,
                "disfluent_audio",
                self.samples[idx].replace(".json", ".wav"),
            )
        )  # [spec is 1, d, t]

        spec_length = torch.tensor(spec.shape[-1])

        # batch_size, donwsampled_temporal_dim, 8 feature_dim - initialize zero matrix
        num_regions = int(spec_length // self.label_downsample_factor)
        if num_regions > 64:  # add checking
            return None
        
        label_matrix = torch.zeros((num_regions, 8))

        disf_region = 0

        # loading disfluent label data
        with open(
            os.path.join(self.data_dir, "disfluent_labels", self.samples[idx]), "r", encoding='utf-8'
        ) as file:
            label_data = json.load(file)

        word_data = label_data[0]["phonemes"]

        # one prediction per region - start, end, exists (0 or 1), type(5) #########
        for word in word_data:
            if word["type"] is not None:
                # sampling_rate / hop_size
                disf_region = int(
                    (((word["end"] + word["start"]) / 2) * global_sampling_rate / 256)
                    // self.label_downsample_factor
                )

                if disf_region >= label_matrix.shape[0]:
                    disf_region = label_matrix.shape[0] - 1

                class_labels = torch.zeros((len(self.labels)))
                class_labels[self.labels.index(word["type"])] = 1

                # for confidence score + classification loss computations (start, end inclusive)
                start_region = int(
                    (word["start"] * global_sampling_rate / 256) // self.label_downsample_factor
                )
                end_region = int(
                    (word["end"] * global_sampling_rate / 256) // self.label_downsample_factor
                )

                # capping so it doesn't exceed duration
                if end_region >= label_matrix.shape[0]:
                    end_region = label_matrix.shape[0] - 1

                if start_region >= label_matrix.shape[0]:
                    start_region = label_matrix.shape[0] - 1

                num_disf_regions = end_region - start_region + 1

                # start, end, exists, type - 1024 is max length
                normalized_start = (word["start"] * global_sampling_rate / 256) / 1024
                normalized_end = (word["end"] * global_sampling_rate / 256) / 1024

                # all regions where disfluency exists
                disfluent_label_data = torch.cat(
                    (
                        torch.tensor(
                            [0, 0, 1],
                            dtype=torch.float32,
                        ),
                        class_labels,
                    ),
                    dim=-1,
                ).repeat((num_disf_regions, 1))

                # center of disfluency for bound prediction
                disfluent_region_label_data = torch.cat(
                    (
                        torch.tensor(
                            [normalized_start, normalized_end, 1],
                            dtype=torch.float32,
                        ),
                        class_labels,
                    ),
                    dim=-1,
                )

                label_matrix[start_region : end_region + 1] = disfluent_label_data
                label_matrix[disf_region] = disfluent_region_label_data

        return (
            text,
            text_length,
            spec,
            spec_length,
            label_matrix,
            num_regions,
            disf_region,  # for bound loss computation
            start_region,
            end_region,
        )

    def get_audio_a(self, filename, _sampling_rate=global_sampling_rate):  ###to change
        _max_wav_value = 32768.0
        _filter_length = 1024
        _hop_length = 256
        _win_length = 1024

        audio, sampling_rate = load_wav_to_torch(filename)

        # simple normalize
        # audio, sr = librosa.load(filename, sr=None)
        # audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)
        # target_dBFS = -20
        # rms = np.sqrt(np.mean(audio**2))
        # target_rms = 10 ** (target_dBFS / 20)
        # normalized_audio = audio * (target_rms / rms)
        # audio = torch.from_numpy(normalized_audio)
        # sampling_rate = sr
        
        # queue = mp.Queue()
        # p = mp.Process(target=vocoder, args=(queue, filename))
        # p.start()
        # try:
        #     audio = queue.get(timeout=10) 
        # except Empty:
        #     print("No data received from the subprocess.")
        
        # p.join()
        # # audio = vocoder(filename=filename)
        # sampling_rate = 22050

        # print("audio: {}".format(audio.shape))

        # print(sampling_rate)
        if sampling_rate != _sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    sampling_rate, _sampling_rate
                )
            )

        # Normalize the waveform between -1.0 and 1.0
        normalized_waveform = audio / torch.max(torch.abs(audio))

        # Scale the normalized waveform to the range of a 16-bit audio
        audio = normalized_waveform * 32767
        # print(f"max audio norm before is {torch.max(audio)}")
        audio_norm = audio / _max_wav_value
        # print(f"max audio norm is {torch.max(audio_norm)}")
        audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".wav", ".spec.pt")

        # # check if spec already exist
        # if os.path.exists(spec_filename):
        #     spec = torch.load(spec_filename)
        # else:
        spec = spectrogram_torch(
            audio_norm,
            _filter_length,
            _sampling_rate,
            _hop_length,
            _win_length,
            center=False,
        )
        # print(f"spec shape is {spec.shape}")
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filename)
        
        return spec, audio_norm

    def get_text(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
