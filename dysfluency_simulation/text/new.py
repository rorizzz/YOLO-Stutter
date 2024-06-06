from text import _symbol_to_id


def use_phoneme(text):
    sequence = []

    for symbol in text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence  # list
