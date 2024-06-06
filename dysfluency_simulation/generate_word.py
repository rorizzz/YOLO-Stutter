import torch
import random
import re
import json
import csv

from text import symbols
from enum import Enum
from generate_utils import *
from models import SynthesizerTrn
import utils
from process_miss import generate_missing
from replacement import generate_phone_replacement
import pysle
import gc
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from text.symbols import symbols, get_vowel
from text import text_to_sequence
from text.new import use_phoneme


def get_text_word(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_time_transcription_word(durations, stn_tst):
    # Calculate the starting and ending timestamps for each phoneme
    start_times = durations.cumsum(0) - durations
    end_times = durations.cumsum(0)
    timestamps = torch.stack((start_times, end_times), dim=1)

    current_start = None
    current_end = 0
    phoneme_combined = ''
    ret = []

    for i, (start, end) in enumerate(timestamps):
        if stn_tst[i] == 16:  # Use stn_tst[i] = 16 as a delimiter
            if phoneme_combined:  
                ret.append({
                    "phoneme": phoneme_combined,
                    "start": round(current_start.item(), 3),
                    "end": round(current_end.item(), 3),
                    "type": None
                })
                phoneme_combined = ''  
            current_start = None  
        elif stn_tst[i] != 0:  
            phoneme = index_to_phoneme(stn_tst[i])
            if current_start is None:
                current_start = start  
            current_end = end  
            if phoneme_combined:
                phoneme_combined += '' + phoneme  
            else:
                phoneme_combined = phoneme

    if phoneme_combined:
        ret.append({
            "phoneme": phoneme_combined,
            "start": round(current_start.item(), 3),
            "end": round(current_end.item(), 3),
            "type": None
        })

    return ret




def generate_word_rep(text, net_g, hps, out_file, sid):
    words = text.split() 
    word_to_repeat = random.choice(words)  
    repeat_times = random.randint(2, 4)
    index = words.index(word_to_repeat)  

    for _ in range(repeat_times - 1):
        words.insert(index, word_to_repeat)

    rep_text =  ' '.join(words)

    stn_tst = get_text_word(rep_text, hps)

    audio, durations = infer_audio(stn_tst, net_g, sid)
    unit_duration = 256 / 22050

    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription_word(durations, stn_tst)


    for i in range(len(words)):
        timestamps[i]["phoneme"] = words[i]

    start_time = timestamps[index]["start"]
    end_time = timestamps[index + repeat_times - 1]["end"]

    timestamps[index].update({
        "start": start_time,
        "end": end_time,
        "type": "rep" 
    })

    del timestamps[index + 1 : index + repeat_times]

    label = [{
        "start": timestamps[0]["start"],
        "end": timestamps[-1]["end"],
        "phonemes": timestamps
    }]

    json_path = out_file.replace("disfluent_audio", "disfluent_labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)

    
    
def generate_word_miss(text, net_g, hps, out_file, sid):
    words = text.split() 
    word_to_miss = random.choice(words)  
    index = words.index(word_to_miss)  
    words.remove(word_to_miss)  # delete after

    miss_text =  ' '.join(words)

    stn_tst = get_text_word(miss_text, hps)

    audio, durations = infer_audio(stn_tst, net_g, sid)
    unit_duration = 256 / 22050

    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription_word(durations, stn_tst)

    for i in range(len(words)):
        timestamps[i]["phoneme"] = words[i]

    miss_time = timestamps[index - 1]["end"]

    miss_item ={
            "phoneme": word_to_miss,
            "start": miss_time,
            "end": miss_time,
            "type": "missing"
        }

    timestamps.insert(index, miss_item)


    label = [{
        "start": timestamps[0]["start"],
        "end": timestamps[-1]["end"],
        "phonemes": timestamps
    }]

    json_path = out_file.replace("disfluent_audio", "disfluent_labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)






def generate_phone_miss(text, net_g, hps, out_file, sid):
    text = text.replace(", ", " ")
    text = text.replace("?", "")
    text = text.replace("-", " ")
    original_phonems = get_phonemes(text)
    # print(original_phonems)
    phonemes_miss = generate_missing(text)
    # print(phonemes_miss)
    stn_tst = get_text(phonemes_miss, hps)
    audio, durations = infer_audio(stn_tst, net_g, sid)
    unit_duration = 256 / 22050

    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription(durations, stn_tst)

    # find missing phoenems
    missing_phoneme_index = []
    j = 0
    for i in range(len(original_phonems)):
        if j >= len(phonemes_miss) or original_phonems[i] != phonemes_miss[j]:
            missing_phoneme_index.append(i)
        else:
            j += 1
    missing_phoneme = missing_phoneme_index[0]
    
    # print(missing_phoneme)
    # print(original_phonems[missing_phoneme])

    for i, phoneme in enumerate(original_phonems):
        if i == missing_phoneme:
            # print("find it:{}, {}".format(i, phoneme))
            missing_start_end = timestamps[i - 1]['end']
            timestamps.insert(i, {
                "phoneme": phoneme,
                "start": missing_start_end,
                "end": missing_start_end,
                "type": "missing"
            })
            break

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]
    
    json_path = out_file.replace("disfluent_audio", "disfluent_labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)



def generate_phone_replace(text, net_g, hps, out_file, sid):
    original_phonems = get_phonemes(text)
    # print(original_phonems)
    phonemes_replace, index = generate_phone_replacement(text)
    # print(phonemes_miss)
    stn_tst = get_text(phonemes_replace, hps)
    audio, durations = infer_audio(stn_tst, net_g, sid)
    unit_duration = 256 / 22050

    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription(durations, stn_tst)

    timestamps[index]["type"] = "replace"

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]
    
    json_path = out_file.replace("disfluent_audio", "disfluent_labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)








def generate(text, net_g, hps, out_path, sid):
    generate_word_rep(text, net_g, hps, out_path.replace(".wav", "_wrep.wav"), sid)
    generate_word_miss(text, net_g, hps, out_path.replace(".wav", "_wmissing.wav"), sid)





if __name__ == "__main__": 
    hps = utils.get_hparams_from_file("./configs/vctk_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint("./path/to/pretrained_vctk.pth", net_g, None)

    # sid = torch.LongTensor([51]).cuda()  # set speaker id 16
    
    # sid_value = int(sys.argv[1])
    # sid = torch.LongTensor([sid_value]).cuda()
    csv_file = "VCTK.csv"
    for speaker_id in tqdm(range(1, 109), desc="processing speakers"):
        sid = torch.LongTensor([speaker_id]).cuda()  # Set speaker id
        bad_cnt = 0

        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            for row in tqdm(reader, desc=f"Generating for Speaker {speaker_id}", leave=False):
            # for row in reader:
                    filename, text = row
                    file_num = filename.split('_')[1].split('.')[0]
                    sid_value = sid.item()
                    filename = filename.replace("239", f"{sid_value:03}")
                    out_path = f"../simulate/tts-word/disfluent_audio/{filename.replace('.txt', '.wav')}"

                    text_file_path = f"../simulate/tts-word/gt_text/{filename.replace('.wav', '.txt')}"
                    with open(text_file_path, 'w') as text_file:
                        text_file.write(text) 
                    try:
                        generate(text, net_g, hps, out_path=out_path, sid=sid)
                    except (IndexError, pysle.utilities.errors.WordNotInIsleError):
                        print("error:{}".format(out_path))
                        bad_cnt += 1
                        continue

                    # print("finish:{}".format(out_path))
            print("skip: " + str(bad_cnt))
            torch.cuda.empty_cache()
            gc.collect()



