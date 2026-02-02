import sys
import json
from tqdm import tqdm
import os
import torch
from PIL import Image
import argparse
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from transformers import AutoProcessor, CLIPTokenizer, GPT2Tokenizer, GPT2LMHeadModel, pipeline, GPT2Model, set_seed, MusicgenForConditionalGeneration, BertTokenizer
from models.music_captioning_ladic import MusicCaptioningLaDiC
from diffusers import AudioLDM2Pipeline
import numpy as np
import soundfile as sf

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="./checkpoints/model_diffMusic_9.pth", type=str,
    help="Path to LaDiC pretrained checkpoint",
)
parser.add_argument(
    '--output_dir', default="./results/diffMusic4", type=str,
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

def load_model(device):
    model = MusicCaptioningLaDiC()
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    clip_processor = AutoProcessor.from_pretrained(args.clip_model)
    preprocess_fn = clip_processor.image_processor
    image = preprocess_fn(image, return_tensors="pt")
    return image.to(device)

# Generate music caption using LaDiC and GPT-2
def generate_music_caption(model, image, caption, prompt, gpt2_model, gpt2_tokenizer, bert_tokenizer):
    image = preprocess_image(image)
    description = bert_tokenizer(list(caption), padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_caption, _ = model(image, description)

    # Set prompt for style transformation
    make_caption = "Rewrite the following sentence in the same formal style as the first sentence."
    input_text = f"{make_caption}: {prompt}"
    
    # Generate modified text using GPT-2 pipeline
    generated_text_outputs = gpt2_model(input_text, max_new_tokens=50)
    generated_text = generated_text_outputs[0]['generated_text']
    
    # Extract the modified part after the colon
    modified_text = generated_text.split(':')[-1].strip()

    return modified_text

# Generate music using the selected music generation model
def generate_music(caption, decoder_model, decoder_type, length_in_sec=10, output_file="generated_music.wav"):
    if decoder_type == "musicgen":
        # Use MusicGen to generate music
        audio_processor = AutoProcessor.from_pretrained(decoder_model)
        music_generator = MusicgenForConditionalGeneration.from_pretrained(decoder_model)
        
        print("Executing musicgen generation...")
        audio_data = audio_processor(text=[caption], padding=True, return_tensors="pt")
        audio_values = music_generator.generate(**audio_data, max_new_tokens=256)
        sampling_rate = 16000

        audio_array = audio_values[0].cpu().numpy()
        wavfile.write(output_file, sampling_rate, audio_array)
    else:
        # Use AudioLDM to generate music
        pipe = AudioLDM2Pipeline.from_pretrained(decoder_model, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        audio = pipe(caption, num_inference_steps=200, audio_length_in_s=10.0).audios[0]
        wavfile.write(output_file, rate=16000, data=audio)

def main():
    model = load_model(device)

    # Load input data
    data = json.load(open("/mnt/storage1/Jin/MUImage/MUImageInstructionsEval.json"))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load GPT-2 and Tokenizers
    gpt2_model = pipeline('text-generation', model='gpt2', device=device)
    set_seed(42)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Select decoder model
    if args.music_decoder == "musicgen":
        decoder_model = "facebook/musicgen-small"
    else:
        decoder_model = "cvssp/audioldm2"

    for row in tqdm(data):
        image_path = os.path.join("/mnt/storage1/Jin/MUImage/audioset_images", row['input_file'])
        music_file = row['output_file']
        caption = row['conversation'][0]['caption']
        prompt = row['conversation'][1]['value']

        audio_segment_path = os.path.join("/mnt/storage1/Jin/MUImage/audioset", music_file)
        audio_segment = AudioSegment.from_file(audio_segment_path)

        image = Image.open(image_path)
        
        # Process and generate the extended caption
        generated_caption = generate_music_caption(model, image, caption, prompt, gpt2_model, gpt2_tokenizer, bert_tokenizer)

        # Generate audio file
        output_music_path = os.path.join(args.output_dir, music_file)
        generate_music(generated_caption, decoder_model, args.music_decoder,
                       length_in_sec=audio_segment.duration_seconds, output_file=output_music_path)

        print(f"Generated Caption and Music for {music_file}: {generated_caption}")

if __name__ == "__main__":
    main()