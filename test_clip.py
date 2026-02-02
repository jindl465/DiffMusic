from transformers import CLIPProcessor, CLIPModel, ClapModel, AutoTokenizer, GPT2LMHeadModel
import torch
from PIL import Image

# Step 1: CLIP을 활용하여 이미지 특징 추출
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_text_features(image_path):
    image = Image.open(image_path).convert("RGB")  # 이미지 로드
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

# Step 2: CLAP을 활용하여 Text Embedding을 Music Caption으로 변환
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_music_embedding(image_path):
    text_embedding = get_clip_text_features(image_path)  # 이미지에서 텍스트 특징 추출
    with torch.no_grad():
        music_embedding = clap_model.get_text_features(text_embedding)
    return music_embedding

# Step 3: GPT 기반 Caption 생성
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_music_caption_with_embedding(image_path):
    # CLAP을 활용하여 Music Embedding 생성
    music_embedding = get_music_embedding(image_path)

    # Prompt + Music Embedding을 입력으로 사용
    input_prompt = "Describe the mood and style of this music: "
    inputs = tokenizer(input_prompt, return_tensors="pt")

    # GPT 모델의 Hidden State와 결합
    with torch.no_grad():
        hidden_states = gpt_model.transformer.wte(inputs["input_ids"])
        combined_input = torch.cat([hidden_states, music_embedding.unsqueeze(1)], dim=1)

        outputs = gpt_model.generate(
            inputs_embeds=combined_input, 
            max_length=50, 
            num_return_sequences=1, 
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption


# 테스트할 이미지 경로
image_path = "--wy8QmLlM8_frame_65.png"

# 음악적인 Caption 생성
music_caption = generate_music_caption_with_embedding(image_path)
print("Generated Music Caption:", music_caption)


# gt : The track belongs to the trip-hop and downtempo genres. 
# The song has a serene, melancholy atmosphere that evokes longing and introspection. 
# In order to create a complex and emotionally resonant song, it prominently incorporates a variety of musical components, including a smooth and sensuous vocal, soft acoustic guitar, subtle electronic beats, and atmospheric synth sounds. A sense of yearning and introspection that may be connected to relationships or personal experiences appears to be the song's main topic. 
# If a video exists, it may be shot in a coastal or seaside location with the calming sound of the ocean waves and possibly distant chattering noises in the backdrop to heighten the introspective mood. It was probably recorded in a studio, allowing for exact management of the complex details.