import torch
from LaDiC.diff_models.diffusion import *
from torch import nn
from LaDiC.diff_models.diffcap_model import Diffuser, Diffuser_with_LN
import time

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = False

def inference(x, tokenizer, model, time_difference = 0):
    x_t = torch.randn((x["image"].shape[0], MAX_LENGTH , IN_CHANNEL), device=device)  # Gaussian noise (bsz, seqlen, 768)
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
        # out1 = model.space_decoder(cond_noise)
        # indexes1 = nn.functional.softmax(out1, dim=-1).argmax(dim=-1)
        # cond_noise = model.space_encoder(indexes1)[0]
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
    
    return ans_strs, x['img_id']