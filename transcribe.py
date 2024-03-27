import whisperx
import torch
import re
from transformers import logging
logging.set_verbosity_error()


def process_text(res, trash_words):
    text = ''
    for seg in res["segments"]:
        text += ' ' + seg["text"]

    recognize = text.strip()
    for words in trash_words:
        recognize = re.sub(words, '', recognize)

    list_words = recognize.split()
    list_words_clean = []

    for q1 in range(len(list_words)):
        if q1 == 0:
            list_words_clean.append(list_words[q1])
        if len(list_words[q1]) > 40:
            pass
        else:
            if list_words[q1] == list_words_clean[-1]:
                pass
            else:
                list_words_clean.append(list_words[q1])

    recognize = ' '.join(list_words_clean)
    return recognize

def transcribe(audio_path, device: str = 'cpu'):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if device == "cpu" else "float16"
    batch_size = 8
    model_whisperx = whisperx.load_model("large-v3", device, compute_type=compute_type, threads=16)

    trash_words = []
    with open('trash.txt', 'r') as f:
        words = f.readlines()
    for i in range(len(words)):
        trash_words.append(words[i].strip('\n'))

    audio = whisperx.load_audio(audio_path)
    result = model_whisperx.transcribe(audio, language='ru', batch_size=batch_size, num_workers=0)
    model_a, metadata = whisperx.load_align_model(language_code="ru", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio_path, device, return_char_alignments=False)
    model_whisperx = None
    return process_text(result, trash_words)