import uvr
from uvr import models
from uvr.utils.get_models import download_all_models
import torch
import audiofile
import soundfile as sf
import json
import os
from tqdm import tqdm
import time

# model init
# default use the 1-HP model
models_json = json.load(open("/home/ubuntu/liwen/ultimatevocalremover_api/src/models_dir/models.json", "r"))
download_all_models(models_json)
device = "cuda"
vr = models.VrNetwork(name="1_HP-UVR", other_metadata={"segment":2, "split":True}, device=device, logger=None)

# Separating an audio file
count = 100
save_dir = "/home/ubuntu/liwen/ultimatevocalremover_api/loop_result"
os.makedirs(save_dir, exist_ok=True)
time_cosume = 0
for i in tqdm(range(count)):
    path = "/home/ubuntu/liwen/ultimatevocalremover_api/assert/test_data/chinese.wav"
    base_name = os.path.basename(os.path.splitext(path)[0])
    start_time = time.time()
    res = vr(path)
    end_time = time.time()
    time_cosume += end_time - start_time
    ins = res["instrumental"]
    vocal = res["vocals"]
    save_ins_path  = os.path.join(save_dir, base_name +  f"_ins_{i}.wav")
    save_voca_path = os.path.join(save_dir, base_name + f"_voca_{i}.wav") 
    sf.write(save_ins_path, ins.astype('float32').T, samplerate=44000)
    sf.write(save_voca_path, vocal.astype('float32').T, samplerate=44000)

print(f"total inference time is {time_cosume} s ")
print(f"average consume time is {time_cosume/count} s ")