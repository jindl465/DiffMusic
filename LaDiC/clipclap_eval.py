import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
import torchaudio.transforms as ta_transforms
from torchvision import transforms
from datasets.melfusion_dataset import MeLFusionDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# ✅ Projection Head 정의 (Image → 공통 공간)
class ImageProjection(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)

# ✅ Projection Head 정의 (Audio → 공통 공간)
class AudioProjection(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)

# ✅ Projection Head 초기화
image_projector = ImageProjection().cuda()
audio_projector = AudioProjection().cuda()

# CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# CLAP 모델 로드 (Audio Processing)
clap_model = AutoModel.from_pretrained("laion/larger_clap_music").cuda()
clap_processor = AutoProcessor.from_pretrained("laion/larger_clap_music")

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
).to("cuda")

# ✅ 학습된 Projection Head 불러오기
checkpoint = torch.load("projection_heads.pth")
image_projector.load_state_dict(checkpoint['image_projector'])
audio_projector.load_state_dict(checkpoint['audio_projector'])
print("✅ Projection Head 로드 완료!")

def compute_top_k_accuracy(query_emb, database_embs, ground_truth_idx, k=1):
    """
    Cosine Similarity를 기반으로 Top-k Accuracy 측정
    """
    similarities = F.cosine_similarity(query_emb.unsqueeze(0), database_embs)  # (1, N)
    print(ground_truth_idx)
    top_k_indices = torch.argsort(similarities, descending=True)[:k]  # 상위 k개 인덱스 추출
    print(top_k_indices)

    # 정답이 Top-k 내에 포함되어 있는지 확인
    return ground_truth_idx in top_k_indices

# ✅ 1. Image → Audio Retrieval Accuracy 측정
def evaluate_image_to_audio(test_dataloader, audio_db, k=1):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating Image → Audio Retrieval"):
            image_input = batch["image"].cuda()
            true_audio_indices = batch["image_id"]  # 정답 인덱스

            image_features = clip_model.get_image_features(image_input)
            image_embeds = image_projector(image_features)

            for i in range(len(image_embeds)):
                is_correct = compute_top_k_accuracy(image_embeds[i], audio_db, true_audio_indices[i], k)
                correct += int(is_correct)

            total += len(image_embeds)

    accuracy = correct / total
    print(f"✅ Image → Audio Retrieval (Top-{k} Accuracy): {accuracy:.4f}")
    return accuracy

# ✅ 2. Audio → Image Retrieval Accuracy 측정
def evaluate_audio_to_image(test_dataloader, image_db, k=1):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating Audio → Image Retrieval"):
            audio_input = batch["waveform"].cuda()
            true_image_indices = batch["image_id"]  # 정답 인덱스

            # Waveform → Mel-Spectrogram 변환
            audio_input = audio_input.mean(dim=1, keepdim=True)  # Mono 변환
            mel_input = mel_spectrogram(audio_input).permute(0, 1, 3, 2)  # (batch, 1, 313, 64)

            audio_features = clap_model.get_audio_features(mel_input.cuda())
            audio_embeds = audio_projector(audio_features)

            for i in range(len(audio_embeds)):
                is_correct = compute_top_k_accuracy(audio_embeds[i], image_db, true_image_indices[i], k)
                correct += int(is_correct)

            total += len(audio_embeds)

    accuracy = correct / total
    print(f"✅ Audio → Image Retrieval (Top-{k} Accuracy): {accuracy:.4f}")
    return accuracy

# ✅ 3. Retrieval DB 구축 (테스트 데이터)
def build_test_embedding_database(test_dataloader):
    """
    테스트 데이터셋의 이미지 & 오디오 임베딩을 저장하여 평가 가능하게 만듦
    """
    image_db = []
    audio_db = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Building Test Retrieval Database"):
            # 이미지 임베딩
            image_input = batch["image"].cuda()
            image_features = clip_model.get_image_features(image_input)
            image_embeds = image_projector(image_features)
            image_db.append(image_embeds)

            # 오디오 임베딩
            audio_input = batch["waveform"].cuda()
            audio_input = audio_input.mean(dim=1, keepdim=True)  # Mono 변환
            mel_input = mel_spectrogram(audio_input).permute(0, 1, 3, 2)  # (batch, 1, 313, 64)
            audio_features = clap_model.get_audio_features(mel_input.cuda())
            audio_embeds = audio_projector(audio_features)
            audio_db.append(audio_embeds)

    return torch.cat(image_db), torch.cat(audio_db)  # (N, 512), (N, 512)

# MeLFusionDataset 로드
print("start")
image_root = "/mnt/storage1/Jin/melfusion/images"
audio_root = "/mnt/storage1/Jin/melfusion/audios"
ann_file = "/mnt/storage1/Jin/melfusion/test_data.csv"

dataset = MeLFusionDataset(
    transform=image_transform,  # CLIP Processor 사용
    tokenizer=clap_processor.tokenizer,  # CLAP의 토크나이저 사용
    image_root=image_root,
    ann_file=ann_file,
    audio_root=audio_root
)
test_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# ✅ Test Dataset으로 Accuracy 평가
test_image_db, test_audio_db = build_test_embedding_database(test_dataloader)

# Image → Audio Retrieval Accuracy 평가
top1_acc_img2audio = evaluate_image_to_audio(test_dataloader, test_audio_db, k=1)
top5_acc_img2audio = evaluate_image_to_audio(test_dataloader, test_audio_db, k=5)

# Audio → Image Retrieval Accuracy 평가
top1_acc_audio2img = evaluate_audio_to_image(test_dataloader, test_image_db, k=1)
top5_acc_audio2img = evaluate_audio_to_image(test_dataloader, test_image_db, k=5)

print(f"🎯 Final Results:")
print(f"✅ Image → Audio (Top-1 Accuracy): {top1_acc_img2audio:.4f}")
print(f"✅ Image → Audio (Top-5 Accuracy): {top5_acc_img2audio:.4f}")
print(f"✅ Audio → Image (Top-1 Accuracy): {top1_acc_audio2img:.4f}")
print(f"✅ Audio → Image (Top-5 Accuracy): {top5_acc_audio2img:.4f}")
