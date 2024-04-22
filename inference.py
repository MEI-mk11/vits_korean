# import matplotlib.pyplot as plt
# import IPython.display as ipd
import numpy as np
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.korean.symbols import symbols
from text.korean import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/aihub_base.json")

# SynthesizerTrn의 첫 번째 매개변수로 len(symbols)이었으나 checkpoint와 맞지 않는 관계로 임의로 178상수를 입력하여 차원 맞춤.
net_g = SynthesizerTrn(
    178,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()

_ = net_g.eval()


_ = utils.load_checkpoint("logs/aihub_base/G_1400.pth", net_g, None)

sr = hps['data']['sampling_rate']

texts = [
    ' 안녕하세요! 반가워요. 지금은 챗가 한국어로 여러분을 인사합니다. 함께 대화를 나누어보아요!',
    '안녕하세요! 오늘은 어떤 일을 하고 계세요?',
    '한국 음식을 좋아하시나요? 제가 추천해 드릴 음식이 있어요.',
    '오늘 날씨가 정말 좋네요. 나들이를 가보는 건 어떠세요?'
]

for i, text in enumerate(texts):    
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([55]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        write(f'output/test{i}.wav', sr, audio.astype(np.float32))
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))


# 나는 너를 소중한 친구라고 생각했는데 너는 그렇게 생각하지 않았구나.
# 오늘 점심에 맛있는 음식을 먹어서 기분이 너무 좋아.
# 나를 배신하면 그 때는 정말 가만두지 않을줄 알아!
