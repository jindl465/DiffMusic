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
from safetensors.torch import load_file # safetensors import 추가
import time
device = torch.device('cuda')
from torchvision.datasets.utils import download_url
# from evaluate import load
torch.backends.cudnn.benchmark = False
from io import BytesIO
from loguru import logger
import scipy.io.wavfile
import re # 장르 파싱을 위해 re import
from collections import Counter # 장르 카운팅을 위해 Counter import

# --- 1. 장르 온톨로지 및 키워드 정의 (신규 추가) ---
# 'Parent_Genre': ['keyword1', 'keyword2', ...]
# (중요) 'rock and roll'처럼 긴 키워드를 'rock'보다 먼저 배치해야 합니다.
GENRE_ONTOLOGY = {
    'Rock': [
        'rock and roll', 'blues-rock', 'hard rock', 'folk rock', 'pop rock', 
        'metal', 'heavy metal', 'rock song', 'rock', 'punk', 'grungy'
    ],
    'Blues': ['blues', 'bluesy', 'country blues'],
    'Jazz': ['jazz', 'jazz fusion', 'swing', 'big band', 'jazzy', 'bossa nova'],
    'Classical': [
        'classical', 'orchestral', 'symphony', 'baroque', 'classical piece',
        'strings', 'string section', 'violin', 'cello', 'viola', 'harp',
        'harpsichord', 'piano solo' # 피아노/현악기 관련 키워드 보강
    ],
    'Electronic': [
        'electronic', 'techno', 'trance', 'ambient', 'new age', 'synth',
        'synthesizer', 'edm', 'house', 'industrial', 'electro', 
        'electronic dance music'
    ],
    'Folk/Country': [
        'folk', 'country', 'bluegrass', 'folk song', 'acoustic', 'banjo', 
        'mandolin', 'ukulele', 'fiddle'
    ],
    'Pop': ['pop', 'pop song', 'synth-pop', 'k-pop'],
    'Funk/Soul': ['funk', 'funky', 'soul', 'soulful', 'groovy', 'r&b', 'disco'],
    'Hip Hop': ['hip hop', 'rap'],
    'Latin': ['latin', 'salsa', 'latin dance'],
    'World': [
        'indian classical', 'indian', 'sitar', 'tabla', 'didgeridoo', 
        'bagpipes', 'traditional folk', 'aboriginal', 'arabic'
    ],
    # 'Other' 카테고리는 평가에서 제외하므로, 키워드가 적어도 됨
    'Other': ['lullaby', 'jig', 'experimental', 'fusion', 'ambient', 'drone', 'instrumental'] 
}

def build_keyword_map():
    """ { 'keyword': 'Parent_Genre' } 맵과 정렬된 키워드 리스트 생성 """
    parent_map = {}
    all_keywords = []
    
    for parent, children in GENRE_ONTOLOGY.items():
        for child in children:
            parent_map[child] = parent
            all_keywords.append(child)
            
    # 긴 키워드가 먼저 매치되도록 정렬 (예: "folk rock" > "rock")
    sorted_keywords = sorted(all_keywords, key=len, reverse=True)
    return parent_map, sorted_keywords

def get_parent_genre(text: str, keywords_list: list, parent_map: dict) -> str:
    """ 텍스트에서 첫 번째로 발견되는 장르 키워드의 상위 장르를 반환 """
    if not text:
        return None
        
    text_low = text.lower()
    for keyword in keywords_list:
        # \b: 단어 경계를 확인하여 'rock'이 'rocket'에 매치되는 것을 방지
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_low):
            return parent_map[keyword]
    return None # 매치되는 장르 없음


# --- 2. 기존 코드 (인수 수정) ---

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_dir', default="/mnt/storage1/Jin/diffMusic/result/test31_muimage_best2", type=str,
    help='Directory to save the evaluation result JSON'
)
parser.add_argument(
    "--clip_model", default="openai/clip-vit-base-patch32", type=str,
    help="Path or name of the CLIP model"
)
# music_decoder 인수는 더 이상 필요 없으므로 삭제
# parser.add_argument(
#     "--music_decoder", default="audioldm", type=str, choices=["musicgen", "audioldm"],
#     help="Music generation model to use: musicgen or audioldm"
# )

args = parser.parse_args()

# Load the LaDiC model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 기존 코드 (수정 없음) ---
# (inference, load_model, generate_music_caption, generate_music, model_evaluate 함수)
# (generate_music_caption와 generate_music는 main에서 호출되지 않지만, 
#  혹시 모를 의존성을 위해 그대로 둡니다.)

def inference(x, tokenizer, model, time_difference = 0):
    x_t = torch.randn((x["image"].shape[0], MAX_LENGTH , IN_CHANNEL), device=device) # Gaussian noise (bsz, seqlen, 768)
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
    MODEL_NAME = "/mnt/storage2/Jin/diffMusic/checkpoints/maxlen100_epoch100_newmuimage"
 
    model.visual_encoder, _ = load_checkpoint(model.visual_encoder, f"{PRETRAINED_DIR}/model_base_capfilt_large.pth")
    
    # 모델 경로를 하드코딩 대신 argparse에서 받아오도록 수정 (선택 사항)
    # model_path = args.model_path 
    model_path = f"{MODEL_NAME}/acc_epoch_15/model.safetensors"
    
    try:
        model.load_state_dict(load_file(model_path), strict=False)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
        
    return model.to(device)

# Generate music caption using LaDiC and GPT-2 (이 함수는 새 로직에서 사용되지 않음)
def generate_music_caption(image, caption, model, gpt2_model, gpt2_tokenizer, bert_tokenizer, gt_text):
    # ... (기존 코드와 동일, 호출되지 않음)
    input_text = f"Generate a detailed music description similar the given reference. Include instruments, rhythm, tempo, and other musical characteristics while preserving the reference's style and meaning. \n\nReference: {caption} \nMusic:"
    generated_text = gpt2_model(input_text, max_new_tokens=50)[0]["generated_text"]
    modified_text = generated_text.split(":")[-1].strip()
    return modified_text

# Generate music using the selected model (이 함수는 새 로직에서 사용되지 않음)
def generate_music(caption, decoder_model, decoder_type, length_in_sec, output_file):
    # ... (기존 코드와 동일, 호출되지 않음)
    if decoder_type == "musicgen":
        audio_processor = AutoProcessor.from_pretrained(decoder_model)
        music_generator = MusicgenForConditionalGeneration.from_pretrained(decoder_model).to(device)
        audio_data = audio_processor(text=[caption], padding=True, return_tensors="pt").to(device)
        audio_values = music_generator.generate(**audio_data, max_new_tokens=int(256 * 10 // 5))
        wav_file_data = BytesIO()
        scipy.io.wavfile.write(wav_file_data, rate=16000, data=audio_values[0, 0].cpu().numpy())
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
        for j, x in tqdm(enumerate(dataloader), desc="Generating MDE text outputs"):
            captions, ids, image_path = inference(x, tokenizer, model, time_difference=5)
 
            image_paths += image_path
            print(x)
            results += x['value']
            gt_texts += x['text'] # Dataloader가 'text' 키에 GT 캡션을 로드한다고 가정

    return image_paths, results, gt_texts

# --- 4. Main 함수 (대폭 수정) ---

def main():
    # Load trained model
    model = load_model()

    # Load dataset and DataLoader
    from dataload import create_dataset
    from torch.utils.data import DataLoader

    # (경로는 하드코딩된 원본을 따름)
    test_csv_file = "/mnt/storage1/Jin/MUImage/MUImageInstructionsEval.json"
    image_dir = "/mnt/storage1/Jin/MUImage/audioset_images_eval"
    audio_dir = "/mnt/storage1/Jin/MUImage/audioset_eval_wav"

    config = {"image_size": 224, "test_ann_file": test_csv_file, "image_root": image_dir, "audio_root": audio_dir}
    
    try:
        test_set = create_dataset("muimage_test", config)
    except ImportError as e:
        print(f"Error: 'dataload' 모듈을 찾을 수 없습니다. {e}")
        print("스크립트를 프로젝트 루트 디렉토리에서 실행 중인지 확인하세요.")
        sys.exit(1)
        
    test_loader = DataLoader(test_set, shuffle=False, batch_size=64, drop_last=False, num_workers=4)

    # Get generated captions
    print("Running model evaluation to get MDE text outputs...")
    # image_paths: 평가된 이미지 파일 경로 리스트
    # generated_captions: MDE가 생성한 텍스트 (D_gen) 리스트
    # gt_texts: 데이터로더가 로드한 GT 텍스트 (D_gt) 리스트
    image_paths, generated_captions, gt_texts = model_evaluate(model, test_set, test_loader)
    
    print(f"Successfully generated {len(generated_captions)} text outputs from MDE.")

    # --- [신규 장르 평가 로직 시작] ---
    # (기존의 GPT-2 및 음악 생성 코드는 모두 삭제)

    print("\n--- 📊 Starting Genre Evaluation ---")
    
    # 1. Build Genre Parser
    parent_map, sorted_keywords = build_keyword_map()

    # 2. Loop and Compare
    results = []
    matches = 0
    total_evaluated = 0

    # `generated_captions`와 `gt_texts`의 길이가 같다고 가정
    if len(generated_captions) != len(gt_texts) or len(generated_captions) != len(image_paths):
        print("Warning: Mismatch in lengths of evaluated data. Zipping to shortest list.")
        
    for i in tqdm(range(len(generated_captions)), desc="Comparing Genres"):
        try:
            image_path = image_paths[i]
            gt_text = gt_texts[i]
            pred_text = generated_captions[i]
        except IndexError:
            continue

        # 3. Get GT Genre
        # (Dataloader가 'text'에 무엇을 로드하는지 불확실하므로, 
        #  json 파일의 'conversation[1].caption'을 사용하는 것이 더 정확할 수 있으나,
        #  일단은 dataload.py가 GT 캡션을 로드했다고 가정)
        gt_genre = get_parent_genre(gt_text, sorted_keywords, parent_map)

        # Skip samples where GT genre is not clear
        if not gt_genre or gt_genre == 'Other':
            continue

        # 4. Get Predicted Genre
        pred_genre = get_parent_genre(pred_text, sorted_keywords, parent_map)
        if not pred_genre:
            pred_genre = 'Other' # Treat non-genre text as 'Other'

        # 5. Compare
        is_match = (gt_genre == pred_genre)
        if is_match:
            matches += 1
        total_evaluated += 1

        results.append({
            'image_path': image_path,
            'gt_text': gt_text,
            'gt_genre': gt_genre,
            'predicted_text': pred_text,
            'predicted_genre': pred_genre,
            'is_match': is_match
        })

    # 6. Report Final Score
    if total_evaluated > 0:
        accuracy = (matches / total_evaluated) * 100
        print("\n--- 📊 Evaluation Finished ---")
        print(f"Total Samples Evaluated (with clear GT genre): {total_evaluated}")
        print(f"Correct Genre Matches: {matches}")
        print(f"MDE Genre Accuracy: {accuracy:.2f}%")
    else:
        print("Error: No samples were evaluated. Check GT genre parsing or dataloader.")

    # 7. Save Results
    # `args.output_dir`을 사용
    output_filename = os.path.join(args.output_dir, "genre_evaluation_results.json")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'metrics': {
                'total_evaluated': total_evaluated,
                'matches': matches,
                'accuracy': accuracy if total_evaluated > 0 else 0
            },
            'results_details': results
        }, f, indent=2)
        
    print(f"Evaluation results saved to {output_filename}")
    
if __name__ == "__main__":
    main()