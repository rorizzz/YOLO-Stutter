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

class StutterType(Enum):
    PHONEBLOCK=0
    PHONEREP=1
    PHONEMISS=2


def generate_phone_rep(text, net_g, hps, out_file, sid):
    phonemes = get_phonemes(text)
    words =  phonemes.split(" ")

    text_indexes, symbol_indexes = get_first_indexes(words)
    # print(text_indexes)
    chosen = random.sample(range(len(text_indexes)), k=1)[0]
    phone = symbols[symbol_indexes[chosen]]
    text_index = text_indexes[chosen]
    
    pause = '.' * random.randint(2, 3) 
    repeated = f"{(phone + pause) * random.randint(2, 4)}"  #1-4 repetitions in the yolo stutter paper

    generated = phonemes[:text_index] + repeated + phonemes[text_index + 1:]    
    # print(generated)
    stn_tst = get_text(generated, hps)

    audio, durations = infer_audio(stn_tst, net_g, sid)
    unit_duration = 256 / 22050

    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription(durations, stn_tst)
    # print(timestamps)
    merged_timestamps = []
    i = 0
    while i < len(timestamps):
        phoneme = timestamps[i]["phoneme"]
        if phoneme == phone and i + 1 < len(timestamps) and timestamps[i + 1]["phoneme"] == ".":
            start_time = timestamps[i]["start"]
            while i + 1 < len(timestamps) and timestamps[i + 1]["phoneme"] == ".":
                i += 1
            end_time = timestamps[i]["end"]
            merged_timestamps.append({
                "phoneme": phone,
                "start": start_time,
                "end": end_time,
                "type": "rep"
            })
        else:
            merged_timestamps.append(timestamps[i])
        i += 1


    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": merged_timestamps
    }]
    # merge same
    for entry in label:
        new_phonemes = []
        i = 0
        while i < len(entry['phonemes']):
            phoneme = entry['phonemes'][i]
            if phoneme['type'] == 'rep':
                start = phoneme['start']
                while i + 1 < len(entry['phonemes']) and entry['phonemes'][i + 1]['type'] == 'rep':
                    i += 1
                end = entry['phonemes'][i]['end']
                new_phonemes.append({'phoneme': phoneme['phoneme'], 'start': start, 'end': end, 'type': 'rep'})
            else:
                new_phonemes.append(phoneme)
            i += 1
        entry['phonemes'] = new_phonemes
    
    json_path = out_file.replace("disfluent_audio", "disfluent_labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)




def generate_phone_block(text, net_g, hps, out_file, sid, sample_rate=22050):
    phonemes = get_phonemes(text)
    stn_tst = get_text(phonemes, hps)

    audio, durations = infer_audio(stn_tst, net_g, sid)

    unit_duration = 256 / sample_rate

    durations = durations * unit_duration
    durations = durations.flatten()

    timestamps = get_time_transcription(durations, stn_tst)
    # json file
    # Create the final structure
    label = [{
        "start": timestamps[0]["start"],
        "end": timestamps[-1]["end"],
        "phonemes": timestamps
    }]

    words = phonemes.split(" ")
    # print(words)
    text_indexes, _ = get_first_indexes(words)
    # print(text_indexes)

    chosen = random.sample(text_indexes, k=1)[0]  # index
    # print(chosen)
    chosen_timestamp = timestamps[chosen]  ##
    # print(chosen_timestamp)
    silence_duration = random.randint(1,4) ## ->
    
    audio = np.array(audio)
    audio = insert_noise(audio, chosen_timestamp["start"]*sample_rate, silence_duration_sec=silence_duration, noise_std_dev=0.01)
    
    block_phoneme = {
        "phoneme": None,
        "start": chosen_timestamp["start"],
        "end": chosen_timestamp["start"] + silence_duration/2,
        "type": "block"
    }
    timestamps.insert(chosen, block_phoneme)

    for i in range(chosen + 1, len(timestamps)):
        timestamps[i]["start"] += silence_duration/2
        timestamps[i]["end"] += silence_duration/2
        timestamps[i]["start"] = round(timestamps[i]["start"], 3)
        timestamps[i]["end"] = round(timestamps[i]["end"], 3)

    
    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
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

    # replace_phoneme = phonemes_replace[index]
    
    # print(missing_phoneme)
    # print(original_phonems[missing_phoneme])

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



def generate_phone_prolong(text, net_g, hps, out_file, sid):
    s = get_phonemes(text)
    stn_tst = get_text(s, hps)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        # sid = torch.LongTensor([10]) # set speaker-id
        outputs = net_g.infer_prolong(s, x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.3)
        if outputs == None:
            return
        
        audio = outputs[0][0,0].data.cpu().float().numpy()
        durations = outputs[-2]
        index_in_s = outputs[-1]
    

    unit_duration = 256 / 22050
    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription(durations, stn_tst)

    timestamps[index_in_s]["type"] = "prolong"

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]
    
    json_path = out_file.replace("disfluent_audio", "disfluent_labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)




def generate(text, net_g, hps, out_path, sid, generate_type=StutterType.PHONEREP):
    generate_phone_block(text, net_g, hps, out_path.replace(".wav", "_block.wav"), sid)
    generate_phone_rep(text, net_g, hps, out_path.replace(".wav", "_rep.wav"), sid)
    generate_phone_miss(text, net_g, hps, out_path.replace(".wav", "_missing1.wav"), sid)
    generate_phone_replace(text, net_g, hps, out_path.replace(".wav", "_replace1.wav"), sid)
    generate_phone_prolong(text, net_g, hps, out_path.replace(".wav", "_prolong1.wav"), sid=sid)


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

    csv_file = "VCTK.csv"
    for speaker_id in tqdm(range(1, 109), desc="processing speakers"):
        sid = torch.LongTensor([speaker_id]).cuda()  # Set speaker id

        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            for row in tqdm(reader, desc=f"Generating for Speaker {speaker_id}", leave=False):
            # for row in reader:
                    filename, text = row
                    file_num = filename.split('_')[1].split('.')[0]
                    sid_value = sid.item()
                    filename = filename.replace("239", f"{sid_value:03}")
                    out_path = f"/data/xxxx/VCTK-tts-rm/disfluent_audio/{filename.replace('.txt', '.wav')}"

                    text_file_path = f"/data/xxxx/VCTK-tts-rm/gt_text/{filename.replace('.wav', '.txt')}"
                    with open(text_file_path, 'w') as text_file:
                        text_file.write(text) 
                    try:
                        generate(text, net_g, hps, out_path=out_path, sid=sid)
                    except (IndexError, pysle.utilities.errors.WordNotInIsleError):
                        print("error:{}".format(out_path))
                        continue

                    # print("finish:{}".format(out_path))
            torch.cuda.empty_cache()
            gc.collect()


