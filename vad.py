import sys, os
from collections import deque
import torch.nn.functional as F
import torch, torchaudio
torch.set_num_threads(1)

torchaudio.set_audio_backend("soundfile")  # switch backend

def read_audio_and_resample(path: str, resample: int = 16000):
    assert torchaudio.get_audio_backend() == 'soundfile'
    wav, sample_rate = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sample_rate != resample:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample)
        wav = transform(wav)
        sample_rate = resample

    assert sample_rate == resample
    return wav.squeeze(0)


def save_audio(path: str, tensor: torch.Tensor, sample_rate: int = 16000):
    torchaudio.save(path, tensor, sample_rate)


def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def get_speech_ts(wav: torch.Tensor,
                  sample_rate: int,
                  model,
                  trig_sum: float = 0.25,
                  neg_trig_sum: float = 0.07,
                  num_steps: int = 8,
                  batch_size: int = 200,
                  num_samples_per_window: int = 4000,
                  run_function=validate):

    num_samples = num_samples_per_window
    assert num_samples % num_steps == 0
    step = int(num_samples / num_steps)  # stride / hop
    outs = []
    to_concat = []
    for i in range(0, len(wav), step):
        chunk = wav[i: i+num_samples]
        if len(chunk) < num_samples:
            chunk = F.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk.unsqueeze(0))
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0))
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []

    if to_concat:
        chunks = torch.Tensor(torch.cat(to_concat, dim=0))
        out = run_function(model, chunks)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    buffer = deque(maxlen=num_steps)  # maxlen reached => first element dropped
    triggered = False
    speeches = []
    current_speech = {}

    speech_probs = outs[:, 1]  # this is very misleading
    #print('speech_probs:', speech_probs, file=sys.stderr)
    for i, predict in enumerate(speech_probs):  # add name
        print(F'{i*step/sample_rate:.2f}, {predict:.2f}', file=sys.stderr)
        buffer.append(predict)
        if ((sum(buffer) / len(buffer))>= trig_sum) and not triggered:
            triggered = True
            current_speech['start'] = step * max(0, i-num_steps)
        if ((sum(buffer) / len(buffer)) < neg_trig_sum) and triggered:
            current_speech['end'] = step * i
            if (current_speech['end'] - current_speech['start']) > 10000:
                speeches.append(current_speech)
            current_speech = {}
            triggered = False
    if current_speech:
        current_speech['end'] = len(wav)
        speeches.append(current_speech)
    return speeches

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('audio_file', type=str)
args = parser.parse_args()

model = init_jit_model(os.path.join('third_party', 'silero_vad', 'files', 'model.jit'))
sample_rate = 16000
wav = read_audio_and_resample(args.audio_file, sample_rate)

segments = get_speech_ts(wav, sample_rate, model, num_steps=4)
for x in segments:
    print(F'{x["start"]/sample_rate} {x["end"]/sample_rate}')
