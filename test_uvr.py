import uvr
from uvr import models
from uvr.utils.get_models import download_all_models
import torch
import audiofile
import soundfile as sf
import json

models_json = json.load(open("/home/ubuntu/liwen/ultimatevocalremover_api/src/models_dir/models.json", "r"))
download_all_models(models_json)
name = "/home/ubuntu/liwen/ultimatevocalremover_api/assert/test_data/emma_original.wav"
# name1 = "/home/ubuntu/liwen/ultimatevocalremover_api/assert/result/chinese.wav"
device = "cuda"

# print(vars(models))
# demucs = models.Demucs(name="hdemucs_mmi", other_metadata={"segment":2, "split":True}, device=device, logger=None)

vr = models.VrNetwork(name="1_HP-UVR", other_metadata={"segment":2, "split":True}, device=device, logger=None)
# Separating an audio file
res = vr(name)
print(res.keys())
ins = res["instrumental"]
vocal = res["vocals"]
# dict_keys(['drums', 'bass', 'other', 'vocals'])
# seperted_audio = res["separated"]
# vocals = seperted_audio["vocals"]
# base = seperted_audio["bass"]
# drums = seperted_audio["drums"]
# other = seperted_audio["other"]
# seperted_instrum = res["drums"]
# seperted_instrum = seperted_instrum
# print(type(seperted_instrum))
# print(seperted_instrum.shape)
# audiofile.write(seperted_instrum.numpy(), "seperate.wav", sampling_rate=16000)

sf.write('/home/ubuntu/liwen/ultimatevocalremover_api/assert/result/emma_instrumental.wav', ins.astype('float32').T, samplerate=44000)
sf.write('/home/ubuntu/liwen/ultimatevocalremover_api/assert/result/emma_vocals.wav', vocal.astype('float32').T, samplerate=44000)