import torch
import json
import commons
import torchaudio
import random
import re
import numpy as np

from text import symbols
from text.new import use_phoneme
from phonemizer import phonemize
from pydub import AudioSegment
from pysle import isletool




def infer_audio(stn_tst, net_g, sid):
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        # sid = torch.LongTensor([239]) # set speaker-id
        outputs = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, sid=sid,noise_scale_w=0.8, length_scale=1.3)
        audio = outputs[0][0,0].data.cpu().float().numpy()
        durations = outputs[-1]
    
    return audio, durations

def load_mapping(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        mapping = json.load(file)
    return mapping

def index_to_phoneme(index):
    file_path = 'symbol_index_mapping.json'  
    mapping = load_mapping(file_path)
    for phoneme, idx in mapping.items():
        if idx == index:
            return phoneme
    return None

def get_time_transcription(durations, stn_tst):
    # Calculate the starting and ending timestamps for each phoneme
    start_times = durations.cumsum(0) - durations
    end_times = durations.cumsum(0)
    timestamps = torch.stack((start_times, end_times), dim=1)

    ret = []
    # Print timestamps for each phoneme
    for i, (start, end) in enumerate(timestamps):
        if stn_tst[i] !=0:  ## blank and padding
            phoneme = index_to_phoneme(stn_tst[i])
            # print(f"{phoneme}: {start:.4f}s to {end:.4f}s")
            ret.append({
                "phoneme": phoneme,
                "start": round(start.item(), 3),  #
                "end": round(end.item(), 3),   #
                "type": None
            })
    
    return ret

def get_first_indexes(words): 
    # get index of the first word
    first_phonemes = [w[0] for w in words]
    symbol_indexes = [symbols.index(p) for p in first_phonemes]

    text_indexes = []
    cur = 0
    for w in words:
        text_indexes.append(cur) 
        cur += len(w) + 1
    
    return text_indexes, symbol_indexes

def insert_noise(audio, insert_position_sec, silence_duration_sec, noise_std_dev):
    silence = AudioSegment.silent(duration=silence_duration_sec * 1000)
    noise = np.random.normal(0, noise_std_dev, len(silence.get_array_of_samples())).astype(np.int16)
    noise_segment = AudioSegment(noise.tobytes(), frame_rate=silence.frame_rate, sample_width=silence.sample_width, channels=1)

    noisy_silence = silence.overlay(noise_segment)

    part1 = audio[:int(insert_position_sec)]
    part2 = audio[int(insert_position_sec):]

    noisy_silence = np.array(noisy_silence.get_array_of_samples())

    output = np.concatenate([part1, noisy_silence, part2], axis=0)

    return output


def write_audio_from_np(audio, out_file):
    torchaudio.save(out_file, torch.Tensor(audio).unsqueeze(0), sample_rate=22050)


def get_phonemes(text):
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    return phonemes

def get_text(text, hps):
    text_norm = use_phoneme(text)
    # print(clean_text)  # plˈiːz kˈɔːl kˈɔːl stˈɛlə  str
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    # print(text_norm)
    return text_norm


