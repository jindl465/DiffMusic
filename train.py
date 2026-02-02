import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from models.music_captioning_ladic import MusicCaptioningLaDiC
from transformers import AutoProcessor, CLIPModel, CLIPTokenizer, BertTokenizer, BertModel
from utils.loss_functions import clip_loss, clap_similarity_loss
from LaDiC.datasets.MUI_dataset import ImageAudioCaptionDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import laion_clap
import wandb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from models.LaDiC_model import LaDiCModel

# wandb 프로젝트 초기화
wandb.init(project="music_captioning", entity="jindl465")  # 사용자 이름이나 팀 이름을 입력


def train(model, data_loader, optimizer, scheduler, criterion, tokenizer, bert_model, bert_tokenizer, clap_model, device):
    model.train()
    total_loss = 0
    total_cross_entropy_loss = 0
    total_clip_loss = 0
    total_clap_loss = 0

    for images, waveforms, captions, gt_captions in tqdm(data_loader):
        images = images.to(device)
        # print(f"Cation : {captions}")
        # captions = tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt").to(device)
        # gt_captions = tokenizer(list(gt_captions), padding=True, truncation=True, return_tensors="pt").to(device)
        # CLIP 토크나이저로 캡션 전처리
        # captions = tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt").to(device)
        # gt_captions = tokenizer(list(gt_captions), padding=True, truncation=True, return_tensors="pt").to(device)
        captions_tokens = bert_tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt").to(device)
        
        generated_captions, image_embedding = model(images, captions_tokens)
        
        # BERT를 사용하여 캡션과 타겟의 텍스트 임베딩 계산
        generated_captions_tokens = bert_tokenizer(list(generated_captions), padding=True, truncation=True, return_tensors="pt").to(device)
        gt_captions_tokens = bert_tokenizer(list(gt_captions), padding=True, truncation=True, return_tensors="pt").to(device)
        
        # BERT 임베딩 생성
        generated_caption_embeddings = bert_model(**generated_captions_tokens).last_hidden_state[:, 0, :]  # CLS 토큰 사용
        target_caption_embeddings = bert_model(**gt_captions_tokens).last_hidden_state[:, 0, :]  # CLS 토큰 사용
        
        # Cross-Entropy Loss 계산
        cross_entropy_value = criterion(generated_caption_embeddings, target_caption_embeddings)
        total_cross_entropy_loss += cross_entropy_value.item()
        
        prediction = tokenizer(generated_captions, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # target = model.module.clip_model.get_text_features(**gt_captions)
        prediction = model.module.clip_model.get_text_features(**prediction)
        
        # cross_entropy_value = criterion(prediction, target)
        # total_cross_entropy_loss += cross_entropy_value.item()

        clip_loss_value = clip_loss(image_embedding, prediction)
        total_clip_loss += clip_loss_value.item()
        
        # generated_text_embedding = clap_model.get_text_embedding(generated_captions, use_tensor=True)
        # audio_embedding = clap_model.get_audio_embedding_from_filelist(x=waveforms, use_tensor=True)
        # clap_loss_value = clap_similarity_loss(generated_text_embedding, audio_embedding)
        # total_clap_loss += clap_loss_value.item()
        
        # print(f'L1 loss : {total_cross_entropy_loss}')
        # print(f'Clip loss : {total_clip_loss}')
        # print(f'Clap loss : {total_clap_loss}')
        # Ensure all loss tensors are on the same device
        cross_entropy_value = cross_entropy_value.to(device)
        clip_loss_value = clip_loss_value.to(device)
        # clap_loss_value = clap_loss_value.to(device)
        
        total_loss_value = cross_entropy_value + 0.1 * clip_loss_value # + 0.01 * clap_loss_value
        total_loss += total_loss_value.item()
        
        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(data_loader)
    print(generated_captions)
    
    wandb.log({
        "Train Loss": avg_loss,
        "L1 Loss": total_cross_entropy_loss / len(data_loader),
        "CLIP Loss": total_clip_loss / len(data_loader),
        # "CLAP Loss": total_clap_loss / len(data_loader),
    })

    return avg_loss

def validate(model, data_loader, criterion, tokenizer, bert_model, bert_tokenizer, clap_model, device):
    model.eval()
    total_loss = 0
    total_clip_loss = 0
    total_clap_loss = 0
    total_cross_entropy_loss = 0

    with torch.no_grad():
        for images, waveforms, captions, gt_captions in data_loader:
            images= images.to(device)
            captions_tokens = bert_tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt").to(device)
        
            generated_captions, image_embedding = model(images, captions_tokens)
            
            # BERT를 사용하여 캡션과 타겟의 텍스트 임베딩 계산
            generated_captions_tokens = bert_tokenizer(list(generated_captions), padding=True, truncation=True, return_tensors="pt").to(device)
            gt_captions_tokens = bert_tokenizer(list(gt_captions), padding=True, truncation=True, return_tensors="pt").to(device)
            
            # BERT 임베딩 생성
            generated_caption_embeddings = bert_model(**generated_captions_tokens).last_hidden_state[:, 0, :]  # CLS 토큰 사용
            target_caption_embeddings = bert_model(**gt_captions_tokens).last_hidden_state[:, 0, :]  # CLS 토큰 사용
            
            # Cross-Entropy Loss 계산
            cross_entropy_value = criterion(generated_caption_embeddings, target_caption_embeddings)
            total_cross_entropy_loss += cross_entropy_value.item()
            
            prediction = tokenizer(generated_captions, padding=True, truncation=True, return_tensors="pt").to(device)
            
            # target = model.module.clip_model.get_text_features(**gt_captions)
            prediction = model.module.clip_model.get_text_features(**prediction)

            clip_loss_value = clip_loss(image_embedding, prediction)
            total_clip_loss += clip_loss_value.item()

            # generated_text_embedding = clap_model.get_text_embedding(generated_captions, use_tensor=True)
            # audio_embedding = clap_model.get_audio_embedding_from_filelist(x=waveforms, use_tensor=True)
            # clap_loss_value = clap_similarity_loss(generated_text_embedding, audio_embedding)
            # total_clap_loss += clap_loss_value.item()
            
            # Ensure all loss tensors are on the same device
            cross_entropy_value = cross_entropy_value.to(device)
            clip_loss_value = clip_loss_value.to(device)
            # clap_loss_value = clap_loss_value.to(device)

            total_loss_value = cross_entropy_value + 0.1 * clip_loss_value # + 0.01 * clap_loss_value
            total_loss += total_loss_value.item()

    avg_loss = total_loss / len(data_loader)
    avg_cross_entropy_loss = total_cross_entropy_loss / len(data_loader)
    avg_clip_loss = total_clip_loss / len(data_loader)
    # avg_clap_loss = total_clap_loss / len(data_loader)

    wandb.log({
        "Validation Loss": avg_loss,
        "Validation L1 Loss": avg_cross_entropy_loss,
        "Validation CLIP Loss": avg_clip_loss,
        # "Validation CLAP Loss": avg_clap_loss,
    })

    return avg_loss, avg_cross_entropy_loss, avg_clip_loss

def save_model(model, epoch, save_path):
    # 모델 파일명 설정
    model_file = os.path.join(save_path, f"model_diffMusic_{epoch}.pth")
    
    # 모델 저장
    torch.save(model.module.state_dict(), model_file)
    print(f"Model saved to {model_file}")

def create_partial_dataset(dataset, use_ratio=0.1):
    # 사용하려는 비율에 따라 데이터셋을 분할
    partial_size = int(len(dataset) * use_ratio)
    remaining_size = len(dataset) - partial_size
    
    # 데이터셋을 분할하여 일부만 사용
    partial_dataset, _ = random_split(dataset, [partial_size, remaining_size])
    return partial_dataset

def main():
    if not dist.is_initialized():
        dist.init_process_group("nccl") 
    local_rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    preprocess_fn = clip_processor.image_processor
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt()
    
    json_file = '/mnt/storage1/Jin/MUImage/MUImageInstructionsTrain.json'
    image_dir = '/mnt/storage1/Jin/MUImage/audioset_images'
    audio_dir = '/mnt/storage1/Jin/MUImage/audioset'
    dataset = ImageAudioCaptionDataset(json_file, image_dir, audio_dir, preprocess_fn, tokenizer)

    # # 데이터셋의 10%만 훈련에 사용
    # partial_dataset = create_partial_dataset(dataset, use_ratio=0.1)

    # # 훈련 및 검증 데이터셋 생성
    # train_size = int(0.8 * len(partial_dataset))
    # val_size = len(partial_dataset) - train_size
    # train_dataset, val_dataset = random_split(partial_dataset, [train_size, val_size])
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 4

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=local_rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)

    model = MusicCaptioningLaDiC().to(device)
    # model = LaDiCModel().to(device)
    model = DDP(model, find_unused_parameters=True, static_graph=True, device_ids=[dist.get_rank()], output_device=dist.get_rank())

    wandb.watch(model)
    
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 현재 날짜와 시간을 가져와서 포맷팅
    current_time = datetime.now().strftime("%m-%d_%H-%M")
    base_dir="checkpoints"
    
    # 저장할 폴더 경로 생성
    save_path = os.path.join(base_dir, current_time)
    os.makedirs(save_path, exist_ok=True)
    
    epochs = 10
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        avg_train_loss = train(model, train_loader, optimizer, scheduler, criterion, tokenizer, bert_model, bert_tokenizer, clap_model, device)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        
        avg_val_loss, avg_cross_entropy_loss, avg_clip_loss = validate(
            model, val_loader, criterion, tokenizer, bert_model, bert_tokenizer, clap_model, device
        )
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, "
              f"Cross Entropy Loss: {avg_cross_entropy_loss:.4f}, "
              f"CLIP Loss: {avg_clip_loss:.4f}")
        
        save_model(model, epoch, save_path)

if __name__ == "__main__":
    main()
