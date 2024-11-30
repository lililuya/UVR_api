import os
import uvr
from uvr import models
from uvr.utils.get_models import download_all_models
import torch
import audiofile
import soundfile as sf
import json

models_json = json.load(open("/home/ubuntu/liwen/ultimatevocalremover_api/src/models_dir/models.json", "r"))
download_all_models(models_json)
device = "cuda"

input_dir = "/home/ubuntu/liwen/ultimatevocalremover_api/assert/test_data/"
output_dir = "/home/ubuntu/liwen/ultimatevocalremover_api/assert/result/"

os.makedirs(output_dir, exist_ok=True)
vr = models.VrNetwork(name="1_HP-UVR", other_metadata={"segment": 2, "split": True}, device=device, logger=None)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):  
        file_path = os.path.join(input_dir, filename)
        print(f"Processing {file_path}...")

        res = vr(file_path)
        ins = res["instrumental"]
        vocal = res["vocals"]

        instrumental_output = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_instrumental.wav")
        vocal_output = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_vocals.wav")

        sf.write(instrumental_output, ins.astype('float32').T, samplerate=44000)
        sf.write(vocal_output, vocal.astype('float32').T, samplerate=44000)

        print(f"Saved instrumental to {instrumental_output} and vocals to {vocal_output}.")