import os
os.environ["MASTER_PORT"] = "29555"


import re

from Levenshtein import ratio as levenshtein_ratio
import torch
from PIL import Image
import clip
from sympy.polys.polyroots import preprocess_roots

import torch
import alpha_clip
import numpy as np
from torchvision import transforms

import cv2



def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # print(completions) #debug
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matches = [re.match(pattern, content[0]['content'], re.DOTALL) for content in completions]
    for content in completions:
        print('prediction=='+content[0]['content']+'\n\n')
    return [0.6 if match else 0.0 for match in matches]

def levenshtein_reward_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    for completion, sol in zip(completions, solution):
        completion = completion[0]['content']
        if '</think>' in completion:
            t = completion.split('</think>')[-1]    # calculate result distance
            t = t.replace('<answer>', '').replace('</answer>', '').strip()
            res.append(levenshtein_ratio(t, sol))
        else:
            res.append(0.0)
    print(res)
    print('\n\n')
    return res


def reward_CoMa_v1_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    for completion, sol in zip(completions, solution):
        completion = completion[0]['content']
        if '</think>' in completion:
            t = completion.split('</think>')[-1]    # calculate result distance
            t = t.replace('<answer>', '').replace('</answer>', '').strip()
            sol = sol.split('#')[0]
            match_score = 0.0
            if sol.lower() in t.lower():
                match_score = 1.0
            # res.append(levenshtein_ratio(t, sol)+match_score)
            res.append(levenshtein_ratio(t, sol))
            # res.append(match_score)
        else:
            res.append(0.0)
    # if len(res) == 5:
    #     res[-1] = res[-1] * 0.95
    print(res)
    print('\n\n')
    return res


def reward_CoMa_v2_dro_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    cot_reward = kwargs['reward_cot']
    index = 0
    for completion, sol in zip(completions, solution):
        completion = completion[0]['content']
        if '</think>' in completion:
            t = completion.split('</think>')[-1]    # calculate result distance
            t = t.replace('<answer>', '').replace('</answer>', '').strip()
            sol = sol.split('#')[0]
            match_score = 0.0
            if sol.lower() in t.lower():
                match_score = 1.0
            # res.append(levenshtein_ratio(t, sol) + match_score + cot_reward[index])
            res.append(match_score + cot_reward[index])

            # res.append(match_score)
            index = index + 1
        else:
            res.append(0.0)
    # if len(res) == 5:
    #     res[-1] = res[-1] * 0.95
    print(res)
    print(cot_reward)
    print('\n\n')
    return res





#######################CLIP##############################
CLIP_reward = False
clipmodel = None
preprocess = None
if CLIP_reward == True:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipmodel, preprocess = clip.load("ViT-B/32", device=device)
    # clipmodel, preprocess = clip.load("RN50", device=device)
    clipmodel.eval()
def clip_cal_sim(img,text):
    global clipmodel, preprocess, device
    img.resize((224, 224), resample=Image.BICUBIC)
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    # logit_scale = clipmodel.logit_scale.exp().clamp(max=50)
    text = 'A cliped photo of ' + text
    text = text.lower()
    if len(text)>200:
        text = text[:200]
    try:
        text_tokens = clip.tokenize([text]).to(device)
    except Exception as e:
        print(f"Tokenizer failed: {e}")
        return 0.0
    with torch.no_grad():
        image_features = clipmodel.encode_image(image_tensor)
        text_features = clipmodel.encode_text(text_tokens)
        # 特征归一化（单位向量化）
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # 计算余弦相似度（原始范围[-1,1]）
        # raw_similarity = (logit_scale * image_features * text_features).sum(dim=-1).item()
        similarity = torch.cosine_similarity(image_features, text_features).item()
    # 线性变换到[0,1]范围
    # normalized_similarity = (similarity + 1) / 2.0
    # return max(0.0, min(1.0, normalized_similarity))
    return similarity
def reward_CoMa_clip_v2_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    clipres = []
    example = kwargs
    for completion, sol in zip(completions, solution):
        completion = completion[0]['content']
        if '</think>' in completion:
            t = completion.split('</think>')[-1]    # calculate result distance
            t = t.replace('<answer>', '').replace('</answer>', '').strip()
            # sol = sol.split('#')[0]
            match_score = 0.0
            if sol.lower() in t.lower():
                match_score = 1.0
            clip_score = clip_cal_sim(example["clip_image"][0], t)
            res.append(levenshtein_ratio(t, sol) + match_score + clip_score)
            clipres.append(clip_score)
            # res.append(match_score)
        else:
            res.append(0.0)
            clipres.append(0.0)

    print(res)
    print(clipres)
    print('\n\n')
    return res







#######################Alpha-CLIP##############################
Alpha_CLIP_reward = True
clipmodel_alpha = None
preprocess_alpha = None
if Alpha_CLIP_reward == True:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipmodel_alpha, preprocess_alpha = alpha_clip.load("ViT-B/16", device=device)  # change to your own ckpt path
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # change to (336,336) when using ViT-L/14@336px
        transforms.Normalize(0.5, 0.26)
    ])
    clipmodel_alpha.eval()
def alpha_clip_cal_sim(img,alpha,texts):
    global clipmodel_alpha, preprocess_alpha, device
    # logit_scale = clipmodel.logit_scale.exp().clamp(max=50)

    image = img
    mask = np.array(alpha)
    # get `binary_mask` array (2-dimensional bool matrix)
    if len(mask.shape) == 2: binary_mask = (mask == 255)
    if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)

    alpha = mask_transform((binary_mask * 255).astype(np.uint8))
    alpha = alpha.half().cuda().unsqueeze(dim=0)

    # print(alpha.shape)
    # calculate image and text features
    image = preprocess_alpha(image).unsqueeze(0).half().to(device)
    text = alpha_clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = clipmodel_alpha.visual(image, alpha)
        text_features = clipmodel_alpha.encode_text(text)

    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    ## print the result
    temperature = 0.02  # 可调参数，值越大越平缓
    similarity = (image_features @ text_features.T) / temperature
    similarity = similarity.softmax(dim=-1)
    return similarity[0]
def reward_CoMa_clip_v3_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    cliptexts = []
    example = kwargs
    for completion, sol in zip(completions, solution):
        completion = completion[0]['content']
        if '</think>' in completion:
            t = completion.split('</think>')[-1]    # calculate result distance
            t = t.replace('<answer>', '').replace('</answer>', '').strip()
            sol = sol.split('#')[0]
            match_score = 0.0
            if sol.lower() in t.lower():
                match_score = 1.0
            # res.append(levenshtein_ratio(t, sol) + match_score)
            res.append(levenshtein_ratio(t, sol))

            if len(t)>100:
                cliptexts.append('')
            else:
                cliptexts.append('A photo of ' + t)
            # res.append(match_score)
        else:
            res.append(0.0)
            cliptexts.append(' ')

    clip_score = alpha_clip_cal_sim(example["orgin_image"][0], example["alpha"][0], cliptexts)
    for i in range(len(res)):
        res[i] = res[i] + clip_score[i].item()

    print(example['filename'][0])
    print(res)
    print(clip_score)
    print('\n\n')
    return res




