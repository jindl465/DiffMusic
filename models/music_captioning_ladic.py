import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor, BertModel
from models.LaDiC_model import LaDiCModel

class MusicCaptioningLaDiC(nn.Module):
    def __init__(self):
        super(MusicCaptioningLaDiC, self).__init__()
        
        # Initialize CLIP for visual features and LaDiC for diffusion-based captioning
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.ladic_model = LaDiCModel()
        
        # Enable gradients for LaDiCModel parameters to support fine-tuning
        for param in self.ladic_model.parameters():
            param.requires_grad = True

        # Custom text encoder to map BERT embeddings to the required latent dimension
        self.text_encoder = TrainableBertEncoder(embed_dim=1536)

    def forward(self, image, description):
        # Remove token_type_ids if present (typically not used by BERT in this context)
        if "token_type_ids" in description:
            description.pop("token_type_ids")

        # Extract text embeddings and project to match image feature dimensions
        description_embedding = self.text_encoder(**description)
        
        # Expand dimensions to match the sequence length expected by the LaDiC decoder
        description_embedding = description_embedding.unsqueeze(1).expand(-1, 24, -1)

        # Process image through CLIP visual encoder
        image_features = image['pixel_values'].squeeze(1).to(description_embedding.device)
        image_embedding = self.clip_model.get_image_features(image_features)

        # Combine image and text features via LaDiC to generate the musical caption
        caption = self.ladic_model.generate_caption(image_features, description_embedding)

        return caption, image_embedding
    
class TrainableBertEncoder(nn.Module):
    def __init__(self, embed_dim=1536):
        super(TrainableBertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask=None):
        # Extract the hidden state of the [CLS] token for sentence-level representation
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.linear(outputs.last_hidden_state[:, 0, :])
        return embeddings