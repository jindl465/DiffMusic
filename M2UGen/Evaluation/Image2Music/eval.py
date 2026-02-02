import subprocess
from hear21passt.base import get_basic_model, get_model_passt
import librosa
from sklearn.metrics import mutual_info_score
import torch
from pathlib import Path
import json
import sys
import re
import os

from tqdm import tqdm

sys.path.append('../Models/imagebind_LLM/ImageBind')

import data
from models import imagebind_model
from models.imagebind_model import ModalityType
from pathlib import Path
import json
from PIL import Image
import laion_clap

# models = ['diffMusic', 'mozarts', 'codi-it']
models = ['diffMusic']
# scores = {model: {"FAD": 0, "FAD_ACC": 0} for model in models}
scores = {model: {"FAD": 0, "CLAP": 0, "IM_RANK": 0} for model in models}
model_files = {"diffMusic": "/mnt/storage1/Jin/diffMusic/result/test28_muimage/llm", "mozarts": "/mnt/storage1/Jin/MUImage/outputs_wav", "codi-i" : "/home/cvmlserver10/Jin/diffMusic/results/codi_image", "codi-it": "/home/cvmlserver10/Jin/diffMusic/results/codi_image_text"}
model_order = {k: v for k, v in enumerate(models)}

kl_model = get_basic_model(mode="logits")
kl_model.eval()
kl_model = kl_model.cuda()

imbind = imagebind_model.imagebind_huge(pretrained=True)
imbind.eval()
imbind.cuda()

clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()
clap_model.eval()
clap_model.cuda()

def load_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    y = y[:16000 * 10]
    return torch.tensor(y).unsqueeze(0)


def compare_files(file1, file2):
    audio1 = load_audio(file1)
    audio2 = load_audio(file2)
    audio_wave1 = audio1.cuda()
    logits1 = kl_model(audio_wave1).cpu().detach()
    probs1 = torch.softmax(logits1, dim=-1)
    audio_wave2 = audio2.cuda()
    logits2 = kl_model(audio_wave2).cpu().detach()
    probs2 = torch.softmax(logits2, dim=-1)
    return probs1, probs2


def load_clap_audio(filename):
    y, _ = librosa.load(filename, sr=48000)
    y = y.reshape(1, -1)
    return y

def load_clap_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    return data.load_and_transform_vision_data([image], "cuda")

def clap_score(image_path, audio_path):
    
    image_embedding = clap_model.get_image_embedding(load_clap_image(image_path)).cpu()
    audio_embedding = clap_model.get_audio_embedding(load_clap_audio(audio_path)).cpu()
    
    
    return torch.nn.functional.cosine_similarity(image_embedding, audio_embedding).item()

def imbind_rank(image, m1, m2, m3, c):
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data([image] * 4, "cuda"),
        ModalityType.AUDIO: data.load_and_transform_audio_data([m1, m2, m3, c], "cuda")
    }
    with torch.no_grad():
        embeddings = imbind(inputs)

    # cosine_similarity = torch.nn.functional.cosine_similarity(embeddings[ModalityType.AUDIO],
    #                                                           embeddings[ModalityType.VISION],
    #                                                           dim=1, eps=1e-8)

    dot_product = torch.sum(embeddings[ModalityType.AUDIO] * embeddings[ModalityType.VISION], dim=1)
    # intersection = torch.min(embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION]).sum(dim=1)
    # union = torch.max(embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION]).sum(dim=1)
    # jaccard_similarity = intersection / union

    ranking = torch.argsort(dot_product)
    # print(dot_product)
    # print(ranking)
    return {model: 1 / (ranking[i] + 1) for i, model in enumerate(models)}

def kl_divergence(pred_probs: torch.Tensor, target_probs: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    kl_div = torch.nn.functional.kl_div((pred_probs + epsilon).log(), target_probs, reduction="none")
    return kl_div.sum(-1).mean()


def fad_score(original, generated):
    print("Calculating FAD Score...")
    command = f"fad_embed --verbose vggish {original} {generated}".split(" ")
    subprocess.run(command)
    command2 = f"fad_score {original}_emb_vggish {generated}_emb_vggish".split(" ")
    result2 = subprocess.run(command2, stdout=subprocess.PIPE)
    match = re.search("FAD score\s=\s+(\d*\.?\d*)", result2.stdout.decode())
    print("FAD command output:", result2.stdout.decode())
    return float(match.group(1))

def normalize_accuracy(value, vmin, vmax):
    return max(0.0, min(1.0, 1 - (value - vmin) / (vmax - vmin))) * 100

json_data = json.load(open("/mnt/storage1/Jin/MUImage/MUImageInstructionsEval.json"))



for model in scores.keys():
    print("fad score start")
    scores[model]["FAD"] = fad_score("/mnt/storage1/Jin/MUImage/audioset_eval_wav", model_files[model])

print("normalized fad score start")
for model in scores.keys():
    generated_files = [str(x).split("/")[-1] for x in Path(model_files[model]).glob("*.wav")]
    target_prob, pred_prob = [], []
    print("kl score start")
    for music in generated_files:
        p1, p2 = compare_files(f"/mnt/storage1/Jin/MUImage/audioset_eval_wav/{music}",
                               f"{model_files[model]}/{music}")
        target_prob.append(p1)
        pred_prob.append(p2)
    target_prob = torch.stack(target_prob, dim=0)
    pred_prob = torch.stack(pred_prob, dim=0)
    kl_item = kl_divergence(pred_prob, target_prob)
    scores[model]["KL"] = kl_item.item()

print("im_rank score start")
for row in tqdm(json_data):
    # music = f"/mnt/storage1/Jin/MUImage/audioset_eval_wav/{row['output_file'].replace('.mp3', '.wav')}"
    image = f"/mnt/storage1/Jin/MUVideo/audioset_video_eval/first/{row['input_file'].replace('.mp4', '.jpg')}"
    music = row['output_file'].replace('.mp3', '.wav')
    # image = row['input_file']
    model_rankings = imbind_rank(image, f"{model_files['diffMusic']}/{music}",
                                 f"{model_files['mozarts']}/{music}",
                                 f"{model_files['codi-i']}/{music}",
                                 f"{model_files['codi-it']}/{music}")
    for model, rank in model_rankings.items():
        scores[model]["IM_RANK"] += rank

for model in models:
    scores[model]["IM_RANK"] /= len(json_data)
    scores[model]["IM_RANK"] = scores[model]["IM_RANK"].item()

print(scores)
print(json.dumps(scores, indent=4))