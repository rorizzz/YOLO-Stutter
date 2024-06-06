import random

""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# _vowels_ipa = "AEIOUaeiouɑɐɒæɔəɘɚɛɜɝɞɨɪɯɰɵøœɶʉʊʌɤʏ"
_vowels_ipa = "AEIOUaeiou"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")


def get_vowel(s):
    vowels_in_input = [(i, char) for i, char in enumerate(s) if char in _vowels_ipa]

    if vowels_in_input:
        select = random.choice(vowels_in_input)
        index_in_w = 2 * select[0] + 1
        # print(select[1])
        return select[0], select[1], index_in_w
    else:
        return None, None, None