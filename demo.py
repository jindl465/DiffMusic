import torch
from models.music_captioning_ladic import MusicCaptioningLaDiC
from utils.loss_functions import clip_loss, diversity_loss, clap_similarity_loss
from PIL import Image
from transformers import AutoProcessor, CLIPTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import os
from collections import OrderedDict

# Image preprocessing
def preprocess_image(image_path):
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    preprocess_fn = clip_processor.image_processor
    image = preprocess_fn(image_path, return_tensors="pt")
    return image    

# Load the trained model
def load_model(device):
    model = MusicCaptioningLaDiC()
    state_dict = torch.load('checkpoints/music_captioning_ladic.pth', map_location=device)
    
    # Handle checkpoints saved with DataParallel (DDP)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove "module." prefix
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

# Generate caption using LaDiC and GPT-2
def generate_caption(model, image, description, device):
    image = image.to(device)
    description = description.to(device)
    
    with torch.no_grad():
        generated_caption, image_embedding = model(image, description)
    
    # Load GPT-2 for style refinement
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Prompt to induce a specific style
    prompt = (
        "Here is a music that is a passionate and bluesy performance. "
        "Imagine a scene with a man wearing glasses and a red shirt in the background. "
        "Now describe this music in a similar style: "
    ) + generated_caption[0]
    
    # Generate extended text using GPT-2
    gpt2_input = gpt2_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gpt2_output = gpt2_model.generate(gpt2_input, max_length=80, num_return_sequences=1)
    extended_caption = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
    
    return extended_caption

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    image_path = "test/3h8EPSvaVJE.jpg"
    description_text = "the image shows a man wearing glasses and a red shirt"
    
    image = Image.open(image_path)
    image = preprocess_image(image)
    
    # Prepare description tokens
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    description = tokenizer(description_text, padding=True, truncation=True, return_tensors="pt").to(device)

    # Generate final output
    generated_caption = generate_caption(model, image, description, device)
    
    print("Generated Caption:", generated_caption)

if __name__ == "__main__":
    main()