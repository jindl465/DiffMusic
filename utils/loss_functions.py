# utils/loss_functions.py

import torch

def clip_loss(image_embedding, caption_embedding):
    """
    CLIP Loss: Calculates loss based on cosine similarity between image and caption embeddings.
    Ensures the generated text is semantically aligned with the visual content.
    """
    similarity = torch.cosine_similarity(image_embedding, caption_embedding, dim=-1)
    return 1 - similarity.mean()

def diversity_loss(generated_captions):
    """
    Diversity Loss: Penalizes high cosine similarity between different captions within a batch.
    Encourages the model to produce a wider variety of expressions and avoid repetitive outputs.
    """
    batch_size = len(generated_captions)
    if batch_size <= 1:
        return torch.tensor(0.0).to(generated_captions.device)
        
    loss = 0
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            similarity = torch.cosine_similarity(generated_captions[i], generated_captions[j], dim=-1)
            loss += similarity.mean()
            
    # Normalize by the number of unique pairs
    return -loss / (batch_size * (batch_size - 1) / 2)


def clap_similarity_loss(text_embedding, audio_embedding):
    """
    CLAP Similarity Loss: Measures the semantic alignment between generated text and Ground Truth (GT) audio.
    The loss decreases as the cosine similarity approaches 1.
    """
    similarity = torch.cosine_similarity(text_embedding, audio_embedding, dim=-1)
    loss = 1 - similarity.mean()
    return loss