import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
import torchaudio.transforms as ta_transforms
from datasets.melfusion_dataset import MeLFusionDataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

# Multi-GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")

# Projection Head 정의 (Image → 공통 공간)
class ImageProjection(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)

# Projection Head 정의 (Audio → 공통 공간)
class AudioProjection(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)

def plot_embedding_space(image_embeds_list, audio_embeds_list, epoch, type, save_path):
    """
    각 Epoch의 Image & Audio Embedding Space를 t-SNE로 시각화
    """
        
    # 리스트를 하나의 NumPy 배열로 변환
    all_image_embeds = torch.cat(image_embeds_list, dim=0).cpu().detach().numpy()
    all_audio_embeds = torch.cat(audio_embeds_list, dim=0).cpu().detach().numpy()

    # 데이터 개수 확인
    n_samples = len(all_image_embeds) + len(all_audio_embeds)

    # t-SNE 적용 (perplexity 값을 샘플 개수에 맞게 조정)
    perplexity_value = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    
    embeddings_2d = tsne.fit_transform(np.vstack([all_image_embeds, all_audio_embeds]))

    # 이미지와 오디오 데이터 분리
    image_2d = embeddings_2d[: len(all_image_embeds)]
    audio_2d = embeddings_2d[len(all_image_embeds):]

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(image_2d[:, 0], image_2d[:, 1], color="blue", label="Images")
    plt.scatter(audio_2d[:, 0], audio_2d[:, 1], color="red", label="Audio")
    plt.legend()
    plt.title(f"Embedding Space Visualization (Epoch {epoch})")
    
    # 저장 & 표시
    plt.savefig(f"{save_path + f'/{type}'}/{epoch}.png")
    # plt.show()

    print(f"✅ Embedding Space 저장 완료: {save_path + f'/{type}'}/{epoch}.png")

# Projection Head 초기화
image_projector = ImageProjection().to(device)
audio_projector = AudioProjection().to(device)

# `DataParallel`을 사용하여 여러 GPU에서 병렬 학습
if num_gpus > 1:
    image_projector = nn.DataParallel(image_projector)
    audio_projector = nn.DataParallel(audio_projector)

# CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# CLAP 모델 로드 (Audio Processing)
clap_model = AutoModel.from_pretrained("laion/larger_clap_music").to(device)
clap_processor = AutoProcessor.from_pretrained("laion/larger_clap_music")

if num_gpus > 1:
    clip_model = nn.DataParallel(clip_model)
    clap_model = nn.DataParallel(clap_model)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP 모델 입력 크기에 맞춤
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mel_spectrogram = ta_transforms.MelSpectrogram(
    sample_rate=16000, 
    n_fft=1024, 
    hop_length=512, 
    n_mels=64
).to(device)

# MeLFusionDataset 로드
image_root = "/mnt/storage1/Jin/melfusion/images"
audio_root = "/mnt/storage1/Jin/melfusion/audios"
ann_file = "/mnt/storage1/Jin/melfusion/train_data.csv"
ann_valid = "/mnt/storage1/Jin/melfusion/validation_data.csv"

train_dataset = MeLFusionDataset(
    transform=image_transform,  # CLIP Processor 사용
    tokenizer=clap_processor.tokenizer,  # CLAP의 토크나이저 사용
    image_root=image_root,
    ann_file=ann_file,
    audio_root=audio_root
)
valid_dataset = MeLFusionDataset(
    transform=image_transform,  # CLIP Processor 사용
    tokenizer=clap_processor.tokenizer,  # CLAP의 토크나이저 사용
    image_root=image_root,
    ann_file=ann_valid,
    audio_root=audio_root
)
train_dataloader = DataLoader(train_dataset, batch_size=32 * num_gpus, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=32 * num_gpus, shuffle=False, num_workers=4)

# Optimizer 설정
optimizer = optim.Adam(list(image_projector.parameters()) + list(audio_projector.parameters()), lr=1e-4)

save_path = "embedding_epoch_300"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    os.mkdir(save_path + '/train')
    os.mkdir(save_path + '/valid')

# Contrastive Loss 정의
def contrastive_loss(emb1, emb2):
    return 1 - F.cosine_similarity(emb1, emb2).mean()  # Cosine Similarity Loss

def train_one_epoch(epoch, dataloader, optimizer, loss_fn, device):
    """
    한 Epoch 동안 Training 수행
    """
    image_projector.train()
    audio_projector.train()
    total_loss = 0
    batch_image_embeds = []
    batch_audio_embeds = []
    image_embeds_list = []
    audio_embeds_list = []

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()

        # Forward Pass
        image_input = batch["image"].to(device)  # (batch, 3, 224, 224)
        audio_input = batch["waveform"].to(device)  # (batch, 1, 160000)
        audio_input = audio_input.mean(dim=1, keepdim=True)  # Mono 변환

        # Feature Extraction
        image_features = clip_model.module.get_image_features(image_input) if num_gpus > 1 else clip_model.get_image_features(image_input)
        image_embeds = image_projector(image_features)

        mel_input = mel_spectrogram(audio_input).permute(0, 1, 3, 2)  # (batch, 1, 313, 64)
        audio_features = clap_model.module.get_audio_features(mel_input.to(device)) if num_gpus > 1 else clap_model.get_audio_features(mel_input.to(device))
        audio_embeds = audio_projector(audio_features)

        # Contrastive Loss 계산
        loss = loss_fn(image_embeds, audio_embeds)

        # Backward Pass & Optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        batch_image_embeds.append(image_embeds.cpu())
        batch_audio_embeds.append(audio_embeds.cpu())

    print(f"Training Loss (Epoch {epoch+1}): {total_loss:.4f}")
    image_embeds_list.append(torch.cat(batch_image_embeds, dim=0))
    audio_embeds_list.append(torch.cat(batch_audio_embeds, dim=0))
    plot_embedding_space(image_embeds_list, audio_embeds_list, epoch, type='train', save_path=save_path)
    return total_loss

def validate_one_epoch(epoch, dataloader, loss_fn, device):
    """
    한 Epoch 동안 Validation 수행
    """
    image_projector.eval()
    audio_projector.eval()
    total_loss = 0
    batch_image_embeds = []
    batch_audio_embeds = []
    image_embeds_list = []
    audio_embeds_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}"):
            image_input = batch["image"].to(device)
            audio_input = batch["waveform"].to(device)
            audio_input = audio_input.mean(dim=1, keepdim=True)  # Mono 변환

            # Feature Extraction
            image_features = clip_model.module.get_image_features(image_input) if num_gpus > 1 else clip_model.get_image_features(image_input)
            image_embeds = image_projector(image_features)

            mel_input = mel_spectrogram(audio_input).permute(0, 1, 3, 2)
            audio_features = clap_model.module.get_audio_features(mel_input.to(device)) if num_gpus > 1 else clap_model.get_audio_features(mel_input.to(device))
            audio_embeds = audio_projector(audio_features)

            # Contrastive Loss 계산
            loss = loss_fn(image_embeds, audio_embeds)
            total_loss += loss.item()
            
            batch_image_embeds.append(image_embeds.cpu())
            batch_audio_embeds.append(audio_embeds.cpu())

    print(f"Validation Loss (Epoch {epoch+1}): {total_loss:.4f}")
    image_embeds_list.append(torch.cat(batch_image_embeds, dim=0))
    audio_embeds_list.append(torch.cat(batch_audio_embeds, dim=0))
    plot_embedding_space(image_embeds_list, audio_embeds_list, epoch, type='valid', save_path=save_path)
    return total_loss

# 학습 루프
epochs = 300
best_val_loss = float("inf")  # 초기에는 무한대로 설정
best_model_path = "best.pth"

for epoch in tqdm(range(epochs)):
    train_loss = train_one_epoch(epoch, train_dataloader, optimizer, contrastive_loss, device)
    val_loss = validate_one_epoch(epoch, valid_dataloader, contrastive_loss, device)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch + 1,
            "image_projector_state_dict": image_projector.state_dict(),
            "audio_projector_state_dict": audio_projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, best_model_path)

        print(f"Best Model Updated (Epoch {epoch+1}) - Validation Loss: {best_val_loss:.4f}")

print("Projection Head 저장 완료!")
