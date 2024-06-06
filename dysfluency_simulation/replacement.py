# Replacement Mapping
replacement_map = {
    'k': 't',
    'g': 't',
    
    'ɹ': ['l', 'j', 'w'],
    'l': ['r', 'j', 'w'],
    
    'f': ['p', 'b', 't', 'd', 'k', 'g'],
    'v': ['p', 'b', 't', 'd', 'k', 'g'],
    'θ': ['p', 'b', 't', 'd', 'k', 'g'],
    's': ['p', 'b', 't', 'd', 'k', 'g', 'θ'],
    'z': ['p', 'b', 't', 'd', 'k', 'g', 'θ'],
    'ʃ': ['p', 'b', 't', 'd', 'k', 'g'],
    'tʃ': ['p', 'b', 't', 'd', 'k', 'g'],
    'j': ['p', 'b', 't', 'd', 'k', 'g'],
    
    'dʒ': ['p', 'b', 't', 'd', 'z', 's', 'ʃ']
}

import random 
import re
from phonemizer import phonemize

def generate_phone_replacement(text):
    # print("original text: ", text)
    replacement_keys = set(replacement_map.keys())
    compound_phon = set(['tʃ', 'dʒ'])

    def contains_key(word):
        common_keys = set(word) & replacement_keys
        for compound in compound_phon:
            if compound in word:
                common_keys.add(compound)
                if compound == 'tʃ':
                    common_keys.remove('ʃ')
        # print("commmon keys are", common_keys)
        
        if common_keys:
            return (True, random.choice(list(common_keys)))
        return (False, None)
        # return any(phon in replacement_keys for phon in word)

    phonemes = phonemize(text, backend="espeak")
    # print("       " + phonemes)
    words = phonemes.split(" ")
    shuffled_indices = random.sample(range(len(words)), k=len(words))

    for index in shuffled_indices:
        chosen_word = words[index]
        contains_key_res = contains_key(chosen_word)
        if not contains_key_res[0]:
            continue
        else:
            replace_phon = contains_key_res[1]
            new_phon = random.choice(replacement_map[replace_phon])
            replaced_word = chosen_word.replace(replace_phon, new_phon, 1)
            result = re.sub(re.escape(chosen_word), replaced_word, phonemes, count=1)
            for index in range(len(phonemes)):
                if phonemes[index] != result[index]:
                    break
            return result, index


def generate_multi_replacement(text, max_replacements=3):
    # Function to determine if a word contains a phoneme that can be replaced
    def contains_key(word):
        common_keys = set(word) & replacement_keys
        for compound in compound_phon:
            if compound in word:
                common_keys.add(compound)
                if compound == 'tʃ':
                    common_keys.remove('ʃ')
        if common_keys:
            return (True, random.choice(list(common_keys)))
        return (False, None)

    # Define phonemes that form compound sounds
    compound_phon = set(['tʃ', 'dʒ'])
    replacement_keys = set(replacement_map.keys())

    # Initialize variables
    phonemes = phonemize(text, backend="espeak")
    # print("origin:{}".format(phonemes))
    words = phonemes.split(" ")
    replacements_done = 0
    indices = []

    # Determine the number of replacements to make
    if max_replacements is None:
        max_replacements = random.randint(1, len(words))  # Randomly choose how many words to replace

    while replacements_done < max_replacements and words:
        # Randomly select a word to try replacing
        index = random.choice(range(len(words)))
        chosen_word = words.pop(index)  # Remove the word from the list to avoid re-selecting

        # Check if the word contains a phoneme that can be replaced
        contains_key_res = contains_key(chosen_word)
        if not contains_key_res[0]:
            continue
        else:
            # Replace the phoneme and update the phonemes string
            replace_phon = contains_key_res[1]
            new_phon = random.choice(replacement_map[replace_phon])
            replaced_word = chosen_word.replace(replace_phon, new_phon, 1)
            result = re.sub(re.escape(chosen_word), replaced_word, phonemes, count=1)

            # Find the index of the replacement
            for i in range(len(phonemes)):
                if phonemes[i] != result[i]:
                    indices.append(i)
                    break

            phonemes = result  # Update the phonemes for the next iteration
            replacements_done += 1

    return phonemes, indices

# result, index = generate_multi_replacement("Ask her to bring these things with her from the store.")
# print("result:" + result)
# print(index)