import sys
import json
from tqdm import tqdm
import os
import torch
from PIL import Image
import argparse
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from transformers import AutoProcessor, GPT2Tokenizer, pipeline, set_seed, BertTokenizer, MusicgenForConditionalGeneration
# from models.music_captioning_ladic import MusicCaptioningLaDiC
from diffusers import AudioLDM2Pipeline
import numpy as np
import soundfile as sf
from diff_models.diffusion import *
from torch import nn
from diff_models.ladic_lora import Diffuser, Diffuser_with_LN
from my_utils.blip_util import load_checkpoint
from safetensors.torch import load_file
import time
device = torch.device('cuda:0')
from torchvision.datasets.utils import download_url
# from evaluate import load
torch.backends.cudnn.benchmark = False

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_dir', default="/mnt/storage1/Jin/diffMusic/result/test26_melfusion", type=str,
    help='Directory to save generated music'
)
parser.add_argument(
    "--clip_model", default="openai/clip-vit-base-patch32", type=str,
    help="Path or name of the CLIP model"
)
parser.add_argument(
    "--music_decoder", default="audioldm", type=str, choices=["musicgen", "audioldm"],
    help="Music generation model to use: musicgen or audioldm"
)

args = parser.parse_args()

# Load the LaDiC model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(x, tokenizer, model, time_difference = 0):
    x_t = torch.randn((x["image"].shape[0], MAX_LENGTH , IN_CHANNEL), device=device)  # Gaussian noise (bsz, seqlen, 768)
    # each prediction involves multiple generation steps
    x_pred = torch.zeros_like(x_t, device=device)
    STEP = 30
    X_SIGMA.to(device)
    X_MEAN.to(device)
    time_start = time.time()
    t = STEP_TOT - 1
    flag = False
    while t > 0:
        t_diff = min(STEP_TOT - 1, t + time_difference)
        if not SELF_COND:
            x_pred = torch.zeros_like(x_t, device=device)
        cond_pred = model(x['image'].to(device), torch.cat([x_t, x_pred], dim=-1).to(device),
                              torch.ones((x["image"].shape[0], MAX_LENGTH), device=device),
                           torch.tensor([t_diff], device=device))
        # out1 = model.space_decoder(cond_noise)
        # indexes1 = nn.functional.softmax(out1, dim=-1).argmax(dim=-1)
        # cond_noise = model.space_encoder(indexes1)[0]
        uncond_pred = model(torch.zeros_like(x["image"], device=device), torch.cat([x_t, x_pred], dim=-1).to(device),
                                torch.ones((x["image"].shape[0], MAX_LENGTH), device=device),
                                # torch.tensor([1, 0], device=device).repeat(x["attention_mask"].shape[0], 1),
                                torch.tensor([t_diff], device=device))
        x_pred = (1 + CLASSIFIER_FREE_WEIGHT) * cond_pred - CLASSIFIER_FREE_WEIGHT * uncond_pred
        # x_pred = cond_pred
        if t < 600 and t > 300 and flag:
            tmp_out = model.lm_head(model.space_decoder(inputs_embeds=x_pred * X_SIGMA + X_MEAN)[0])
            softmax_tmp = nn.functional.softmax(tmp_out, dim=-1)
            # most_confident_token =softmax_tmp.max(dim=-1).values.argmax(dim=-1)
            confidence = softmax_tmp.max(dim=-1).values
            _, idx = torch.sort(confidence, descending=False)
            to_be_updated_idx = idx[:,:MAX_LENGTH//3].to(device)
            gaussian_noise = torch.randn_like(x_pred).to(x_pred.device)
            # x_pred[to_be_updated_idx, :] = gaussian_noise[to_be_updated_idx, :].clone()
            x_t = diffuse_t(x_pred, torch.tensor([t], device=device) - STEP, is_test=True)
            x_t[torch.arange(x_pred.shape[0])[:, None], to_be_updated_idx] = gaussian_noise[torch.arange(x_t.shape[0])[:, None], to_be_updated_idx].clone()
            # indexes1 = nn.functional.softmax(out1, dim=-1).argmax(dim=-1)
            # pred_x0 = (model.space_encoder(indexes1)[0] - X_MEAN)/X_SIGMA
            t = STEP_TOT - 1
            flag = False
        elif t > STEP:
            # noise = pred_x0
            x_t = diffuse_t(x_pred, torch.tensor([t], device=device) - STEP, is_test=True)
            #x_t = p_sample(x_t[:, :MAX_LENGTH, :], x_pred, torch.tensor([t], device=device) , STEP)
        t -= STEP
    cond_pred = x_pred * X_SIGMA + X_MEAN
    out = model.lm_head(model.space_decoder(inputs_embeds=cond_pred)[0])
    indexes = nn.functional.softmax(out, dim=-1).argmax(dim=-1)
    indexes = indexes.unique_consecutive(dim=-1)
    import itertools

    ans_strs = [tokenizer.decode(index) for index in indexes]
    time_end = time.time()
    # print('time cost', time_end - time_start, 's')
    ans_strs = [' '.join(k for k, _ in itertools.groupby(original_str.split())) for original_str in ans_strs]
    ans_strs = [original_str.strip('.').strip() + '.' for original_str in ans_strs]
    ans_strs = [original_str.split('.')[0] + '.' for original_str in ans_strs]

    return ans_strs, x['image_id'], x['image_path']


# Load LaDiC Model
def load_model():
    model = Diffuser_with_LN(image_size=224)
    PRETRAINED_DIR = "/home/cvmlserver10/Jin/diffMusic/LaDiC/pretrained_ckpt"
    MODEL_NAME = "/mnt/storage1/Jin/diffMusic/checkpoints/maxlen50_epoch60_best/"
    
    model.visual_encoder, _ = load_checkpoint(model.visual_encoder, f"{PRETRAINED_DIR}/model_base_capfilt_large.pth")
    model.load_state_dict(load_file(f"{MODEL_NAME}/acc_epoch_59/model.safetensors"), strict=False)
    return model.to(device)

# Generate music caption using LaDiC and GPT-2
def generate_music_caption(image, caption, model, gpt2_model, gpt2_tokenizer, bert_tokenizer):
    # description = bert_tokenizer([caption], padding=True, truncation=True, return_tensors="pt").to(device)

    input_text = f"Convert this image caption into a descriptive music caption:\n\n{caption}\n\nOutput:"
    generated_text = gpt2_model(input_text, max_new_tokens=50)[0]["generated_text"]
    modified_text = generated_text.split(":")[-1].strip()

    return modified_text

# Generate music using the selected model
def generate_music(caption, decoder_model, decoder_type, length_in_sec, output_file):
    if decoder_type == "musicgen":
        audio_processor = AutoProcessor.from_pretrained(decoder_model)
        music_generator = MusicgenForConditionalGeneration.from_pretrained(decoder_model)
        
        audio_data = audio_processor(text=[caption], padding=True, return_tensors="pt")
        audio_values = music_generator.generate(**audio_data, max_new_tokens=256)
        
        audio_array = audio_values[0].numpy()
        wavfile.write(output_file, rate=16000, data=audio_array)
    
    else:
        pipe = AudioLDM2Pipeline.from_pretrained(decoder_model, torch_dtype=torch.float16).to(device)
        audio = pipe(caption, num_inference_steps=50, audio_length_in_s=length_in_sec).audios[0]
        wavfile.write(output_file, rate=16000, data=audio)

# Model evaluation (Image → Caption)
def model_evaluate(model, dataset, dataloader):
    tokenizer = dataset.tokenizer
    model.eval()
    results = []
    image_paths = []
    gt_texts = []

    with torch.no_grad():
        for j, x in tqdm(enumerate(dataloader)):
            captions, ids, image_path = inference(x, tokenizer, model, time_difference=5)
            
            image_paths += image_path
            results += captions
            gt_texts += x['text']

    return image_paths, results, gt_texts

def main():
    # Load trained model
    model = load_model()

    # Load dataset and DataLoader
    from dataload import create_dataset
    from torch.utils.data import DataLoader

    test_csv_file = "/mnt/storage1/Jin/melfusion/test_data.csv"
    image_dir = "/mnt/storage1/Jin/melfusion/images"
    audio_dir = "/mnt/storage1/Jin/melfusion/audios"

    config = {"image_size": 224, "test_ann_file": test_csv_file, "image_root": image_dir, "audio_root": audio_dir}
    test_set = create_dataset("melfusion_test", config)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=32, drop_last=False, num_workers=4)

    # Get generated captions
    image_paths, generated_captions, gt_texts = model_evaluate(model, test_set, test_loader)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir+'/gt'):
        os.makedirs(args.output_dir+'/gt', exist_ok=True)
        os.makedirs(args.output_dir+'/nollm', exist_ok=True)

    # Load GPT-2 for music-friendly caption expansion
    gpt2_model = pipeline("text-generation", model="gpt2", device=0)
    set_seed(42)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Select music generation model
    decoder_model = "facebook/musicgen-small" if args.music_decoder == "musicgen" else "cvssp/audioldm2"
    result_json = []
    
    # Generate music using generated captions
    for index, result in enumerate(tqdm(generated_captions)):
        image_path = image_paths[index]
        caption = result
        gt_text = gt_texts[index]

        # Convert image path to audio path
        audio_path = image_path.replace(".png", ".mp3").replace(image_dir, audio_dir)
        if not os.path.exists(audio_path):
            print(f"Audio file not found for {image_path}. Skipping...")
            continue

        # Load image for caption expansion
        image = Image.open(image_path)

        # Determine music duration from existing audio file
        audio_segment = AudioSegment.from_file(audio_path)
        music_length = audio_segment.duration_seconds
        result_audio = audio_path.split("/")[-1].replace("mp3", "wav")
        
        output_music_path1 = f"{args.output_dir}/nollm/{result_audio}"
        generate_music(caption, decoder_model, args.music_decoder, music_length, output_music_path1)

        # Generate expanded music caption
        # music_caption = generate_music_caption(image, caption, model, gpt2_model, gpt2_tokenizer, bert_tokenizer)

        # Generate music
        output_music_path = f"{args.output_dir}/gt/{result_audio}"
        # generate_music(gt_text, decoder_model, args.music_decoder, music_length, output_music_path)

        print(f"Generated Music for {image_path}: {caption}")
        result_json.append({'image_path':image_path, 'gt_text':gt_text, 'generated_caption':caption})
    
    # JSON 파일로 저장
    with open(f"{args.output_dir}/result.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()

