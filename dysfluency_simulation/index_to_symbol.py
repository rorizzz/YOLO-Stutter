import json
import torch

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

def get_time_transcription(w_ceil, x):
    sample_rate = 22050  
    unit_duration = 256 / sample_rate  # Duration represented by '1' in the tensor

    durations = w_ceil * unit_duration

    durations_flat = durations.flatten()

    # Calculate the starting and ending timestamps for each phoneme
    start_times = durations_flat.cumsum(0) - durations_flat
    end_times = durations_flat.cumsum(0)

    timestamps = torch.stack((start_times, end_times), dim=1)

    # Print timestamps for each phoneme
    for i, (start, end) in enumerate(timestamps):
        if x[i] !=0 and x[i] != 16:  ## blank and padding
            phoneme = index_to_phoneme(x[i])
            print(f"{phoneme}: {start:.4f}s to {end:.4f}s")


def get_time_transcription_word(w_ceil, x):
    sample_rate = 22050  
    unit_duration = 256 / sample_rate  # Duration represented by '1' in the tensor

    durations = w_ceil * unit_duration

    durations_flat = durations.flatten()

    # Calculate the starting and ending timestamps for each phoneme
    start_times = durations_flat.cumsum(0) - durations_flat
    end_times = durations_flat.cumsum(0)

    timestamps = torch.stack((start_times, end_times), dim=1)

    current_start = None
    current_end = 0
    phoneme_combined = ''

    for i, (start, end) in enumerate(timestamps):
        if x[i] == 16:  # Use x[i] = 16 as a delimiter
            if phoneme_combined:  # If we have a combined phoneme, print it
                print(f"{phoneme_combined}: {current_start:.4f}s to {current_end:.4f}s")
                phoneme_combined = ''  # Reset the combined phoneme
            current_start = None  # Reset start for the next segment
        elif x[i] != 0:  # If not a blank
            phoneme = index_to_phoneme(x[i])
            if current_start is None:
                current_start = start  # Set start for a new segment
            current_end = end  # Update end for the current segment
            phoneme_combined += phoneme  # Combine phonemes

    # Print the last segment if exists
    if phoneme_combined:
        print(f"{phoneme_combined}: {current_start:.4f}s to {current_end:.4f}s")