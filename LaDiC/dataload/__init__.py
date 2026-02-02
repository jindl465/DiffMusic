import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BertTokenizer

from dataload.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval
from dataload.para_cap_dataset import para_train, para_eval
from datasets.melfusion_dataset import MeLFusionDataset
from datasets.MUI_dataset import MUImageDataset
from datasets.MUI_video_dataset import MUVideoDataset

# from transform.randaugment import RandomAugment


def create_dataset(dataset, config, min_scale=0.5):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
        #                                       'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        # transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  

    
    if dataset=='caption_coco':
        train_dataset = coco_karpathy_train(transform_train, tokenizer, image_root=config['image_root'], ann_root=config['ann_root'])
        val_dataset = coco_karpathy_caption_eval(transform_test, tokenizer, image_root=config['image_root'], ann_root=config['ann_root'], split='val')
        test_dataset = coco_karpathy_caption_eval(transform_test, tokenizer, image_root=config['image_root'], ann_root=config['ann_root'], split='test')
        return train_dataset, val_dataset , test_dataset


    elif dataset=='para_cap':
        train_dataset = para_train(transform_train, tokenizer, image_root='/diffusion/datasets/ParaCap/train_img', ann_root='/diffusion/datasets/ParaCap')
        val_dataset = para_eval(transform_test, tokenizer, image_root='/diffusion/datasets/ParaCap/val_img', ann_root='/diffusion/datasets/ParaCap', split='val')
        test_dataset = para_eval(transform_test, tokenizer, image_root='/diffusion/datasets/ParaCap/test_img', ann_root='/diffusion/datasets/ParaCap', split='test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'muimage':
        train_dataset = MUImageDataset(
            transform=transform_train,
            tokenizer=tokenizer,
            image_dir=config['image_root'],
            json_file=config['ann_file'],
            audio_dir=config['audio_root'],
            sample_rate=config.get('sample_rate', 16000),
            duration=config.get('duration', 10.0),
        )
        val_dataset = MUImageDataset(
            transform=transform_test,
            tokenizer=tokenizer,
            image_dir=config['image_root'],
            json_file=config['val_ann_file'],
            audio_dir=config['audio_root'],
            sample_rate=config.get('sample_rate', 16000),
            duration=config.get('duration', 10.0),
        )
        
        return train_dataset, val_dataset
    
    elif dataset == 'muimage_test':
        test_dataset = MUImageDataset(
            transform=transform_train,
            tokenizer=tokenizer,
            image_dir=config['image_root'],
            json_file=config['test_ann_file'],
            audio_dir=config['audio_root'],
            sample_rate=config.get('sample_rate', 16000),
            duration=config.get('duration', 10.0),
        )
        
        return test_dataset
    
    elif dataset == 'muvideo_test':
        test_dataset = MUVideoDataset(
            transform=transform_train,
            tokenizer=tokenizer,
            image_dir=config['image_root'],
            json_file=config['test_ann_file'],
            audio_dir=config['audio_root'],
            sample_rate=config.get('sample_rate', 16000),
            duration=config.get('duration', 10.0),
        )
        
        return test_dataset

    elif dataset == 'melfusion':  # MeLFusionDataset 추가
        train_dataset = MeLFusionDataset(
            transform=transform_train,
            tokenizer=tokenizer,
            image_root=config['image_root'],
            ann_file=config['ann_file'],
            audio_root=config['audio_root'],
            sample_rate=config.get('sample_rate', 16000),
            duration=config.get('duration', 10.0),
        )
        val_dataset = MeLFusionDataset(
            transform=transform_test,
            tokenizer=tokenizer,
            image_root=config['image_root'],
            ann_file=config['val_ann_file'],  # Validation set
            audio_root=config['audio_root'],
            sample_rate=config.get('sample_rate', 16000),
            duration=config.get('duration', 10.0),
        )
        
        return train_dataset, val_dataset
    
    elif dataset == 'melfusion_test':
        test_dataset = MeLFusionDataset(
            transform=transform_test,
            tokenizer=tokenizer,
            image_root=config['image_root'],
            ann_file=config['test_ann_file'],  # Test set
            audio_root=config['audio_root'],
            sample_rate=config.get('sample_rate', 16000),
            duration=config.get('duration', 10.0),
        )
        
        return test_dataset
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

