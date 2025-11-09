import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from eval_utils import *
import os
from tqdm import tqdm
import random


'''
Qwen2.5-VL-7B-Instruct
Qwen2.5-VL-3B-Instruct
'''



sft_path = '/sshfs/pretrains/Qwen/Qwen2.5-VL-7B-Instruct/'
base_model_path = '/sshfs/pretrains/Qwen/Qwen2.5-VL-7B-Instruct/'

device = "cuda" if torch.cuda.is_available() else "cpu"

compute_dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

basemodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # base_model_path,
    sft_path,
    device_map=device,
    torch_dtype=compute_dtype,
    # quantization_config=bnb_config,
    attn_implementation="flash_attention_2" 
)


model = basemodel

processor = AutoProcessor.from_pretrained(
    base_model_path,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28
)


def generate_response(image_path, prompt,mask_task=False):
    if mask_task:
        messages = [
            {'role': 'system', 'content': [{"type": "text", "text": R1V_SYS}]},
            {"role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]
    else:
        messages = [
            {"role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    try:
        inputs = processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt"
        ).to(device)

        generation_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "use_cache": True
        }

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )

        # 解码输出
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response.strip()
    except:
        print("wrong")
        return ""










###########################################################################
def postcess_llm_answer(output,reason_type=False,choose=False):
    # print(output)
    if output == "":
        return ""
    if reason_type:
        reason_answer =  output.split('assistant')[2]
        # print(reason_answer)
        if '<answer>' in reason_answer:
            answer = reason_answer.split('<answer>')[1].split('</answer>')[0]
        else:
            answer = ''
        if choose:
            if 'a' in answer.lower():
                answer = 'A'
            elif 'b' in answer.lower():
                answer = 'B'
            elif 'c' in answer.lower():
                answer = 'C'
            elif 'd' in answer.lower():
                answer = 'D'
        else:
            if 'yes' in answer.lower():
                answer = 'yes'
            elif 'no' in answer.lower():
                answer = 'no'
    else:
        answer = output.split('assistant')[2]
        if choose:
            if 'a' in answer.lower():
                answer = 'A'
            elif 'b' in answer.lower():
                answer = 'B'
            elif 'c' in answer.lower():
                answer = 'C'
            elif 'd' in answer.lower():
                answer = 'D'
        else:
            if 'yes' in answer.lower():
                answer = 'yes'
            elif 'no' in answer.lower():
                answer = 'no'
    return answer

###########################################################################
#################################CoMa_eval#################################


def CoMa_judgement_eval(image_path,list):

    count = 0
    for ann in tqdm(list):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        name = ann['category_name']

        to_ask = []
        to_ask.append(name)
        to_ask.extend(ann['judgement_negtive_confusion'])
        to_ask.extend(random.sample(ann['irrelevant_negtive'], k=2))

        for object in to_ask:

            question = (
                f'I have covered A part of the image with black. Please guess if this is {object}.'
                f'Provide your thinking process in <think></think>, and directly answer yes or no in <answer></answer>.')
            llm_answer = generate_response(mask_img, question, mask_task=True)
            res = postcess_llm_answer(llm_answer,reason_type=True)
            gt = 'no'
            if object == name:
                gt = 'yes'
            if res == gt:
                count += 1
    return count





def CoMa_judgement_detail_eval(image_path,list):

    count = 0
    count_conf = 0
    count_irre = 0
    for ann in tqdm(list):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        name = ann['category_name']

        conf = []
        irre = []
        conf.extend(ann['judgement_negtive_confusion'])
        conf.extend(ann['choose_negtive_confusion'])
        irre.extend(random.sample(ann['irrelevant_negtive'], k=2))

        question = (
            f'I have covered A part of the image with black. Please guess if this is {name}.'
            f'Provide your thinking process in <think></think>, and directly answer yes or no in <answer></answer>.')
        llm_answer = generate_response(mask_img, question, mask_task=True)
        res = postcess_llm_answer(llm_answer,reason_type=True)
        if res == 'yes':
            count += 1

        for object in conf:
            question = (
                f'I have covered A part of the image with black. Please guess if this is {object}.'
                f'Provide your thinking process in <think></think>, and directly answer yes or no in <answer></answer>.')
            llm_answer = generate_response(mask_img, question, mask_task=True)
            res = postcess_llm_answer(llm_answer,reason_type=True)
            if res == 'no':
                count_conf += 1

        for object in irre:
            question = (
                f'I have covered A part of the image with black. Please guess if this is {object}.'
                f'Provide your thinking process in <think></think>, and directly answer yes or no in <answer></answer>.')
            llm_answer = generate_response(mask_img, question, mask_task=True)
            res = postcess_llm_answer(llm_answer,reason_type=True)
            if res == 'no':
                count_irre += 1

    return count,count_conf,count_irre








def CoMa_choose_HARD_eval(image_path,list):

    count = 0
    for ann in tqdm(list):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        name = ann['category_name']

        to_ask = []
        to_ask.append(name)
        to_ask.extend(ann['choose_negtive_confusion'])
        to_ask.extend(ann['judgement_negtive_confusion'])
        to_ask.extend(random.sample(ann['irrelevant_negtive'], k=2))
        choose_index = ["A","B","C","D","E","F","G"]
        random.shuffle(to_ask)
        gt = choose_index[to_ask.index(name)]
        question = ( f'Based on the contextual clues surrounding the black rectangular mask in the image, infer the most likely object hidden beneath it. '
                    f'You have the following seven choices: A.{to_ask[0]} B.{to_ask[1]} C.{to_ask[2]} D.{to_ask[3]} E.{to_ask[4]} F.{to_ask[5]} G.{to_ask[6]}. Please choose the most suitable option and answer. '
                    f'Provide your thinking process in <think></think>, and directly answer A, B, C, D, E, F or G in <answer></answer>.'
                     )
        llm_answer = generate_response(mask_img, question, mask_task=True)
        res = postcess_llm_answer(llm_answer,reason_type=True,choose = True)
        if res == gt:
            count = count + 1

    return count


def CoMa_choose_EASY_eval(image_path,list):

    count = 0
    for ann in tqdm(list):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        name = ann['category_name']

        to_ask = []
        to_ask.append(name)
        to_ask.extend(ann['choose_negtive_confusion'])
        to_ask.extend(random.sample(ann['irrelevant_negtive'], k=1))
        choose_index = ["A","B","C","D"]
        random.shuffle(to_ask)
        gt = choose_index[to_ask.index(name)]
        # print(gt)
        question = (
            f'Based on the contextual clues surrounding the black rectangular mask in the image, infer the most likely object hidden beneath it. '
            f'You have the following four choices: A.{to_ask[0]} B.{to_ask[1]} C.{to_ask[2]} D.{to_ask[3]}.'
            f'Please use clues such as the environment and mask size in the image to make inferences, then consider the possibilities of the options and select the best one.'
            f'Provide your thinking process in <think></think>, and choose the most suitable option with A, B, C or D in <answer></answer>.')
        llm_answer = generate_response(mask_img, question, mask_task=True)
        res = postcess_llm_answer(llm_answer,reason_type=True,choose = True)
        if res == gt:
            count = count + 1

    return count



question = (
    f'Based on the contextual clues surrounding the black rectangular mask in the image, infer the most likely object hidden beneath it. '
    # f'You have the following four choices: A.{to_ask[0]} B.{to_ask[1]} C.{to_ask[2]} D.{to_ask[3]}.'
    f'Provide your thinking process in <think></think>, and choose the most suitable option and answer, directly answer A, B, C or D in <answer></answer>.')



def CoMa_eval(image_path,json_path):
    with open(json_path) as f:
        annotations = json.load(f)
    hard = annotations['hard']
    middle = annotations['middle']
    easy = annotations['easy']

    #JUDGEMENT ###################################################
    # print("Let's go HARD_judgement!")
    # right_hard = CoMa_judgement_eval(image_path,hard)
    # print(f"HARD_JUD_ACC:{right_hard/(len(hard)*5)}")
    #
    # print("Let's go MIDDLE_judgement!")
    # right_middle = CoMa_judgement_eval(image_path, middle)
    # print(f"MIDDLE_JUD_ACC:{right_middle / (len(middle) * 5)}")
    #
    # print("Let's go EASY_judgement!")
    # right_easy = CoMa_judgement_eval(image_path, easy)
    # print(f"EASY_JUD_ACC:{right_easy / (len(easy) * 5)}")
    ##############################################################


    # CHOOSE_EASY ################################################
    print("Let's go HARD_choose_easy!")
    right_hard_cho = CoMa_choose_EASY_eval(image_path,hard)
    print(f"HARD_CHOeasy_ACC:{right_hard_cho/(len(hard))}")

    print("Let's go MIDDLE_choose_easy!")
    right_middle_cho = CoMa_choose_EASY_eval(image_path, middle)
    print(f"MIDDLE_CHOeasy_ACC:{right_middle_cho / (len(middle))}")

    print("Let's go EASY_judgement!")
    right_easy_cho = CoMa_choose_EASY_eval(image_path, easy)
    print(f"EASY_CHOeasy_ACC:{right_easy_cho / (len(easy))}")
    #############################################################


    # CHOOSE_HARD ################################################
    # print("Let's go HARD_choose_hard!")
    # right_hard_cho = CoMa_choose_HARD_eval(image_path, hard)
    # print(f"HARD_CHOhard_ACC:{right_hard_cho / (len(hard))}")
    #
    # print("Let's go MIDDLE_choose_hard!")
    # right_middle_cho = CoMa_choose_HARD_eval(image_path, middle)
    # print(f"MIDDLE_CHOhard_ACC:{right_middle_cho / (len(middle))}")
    #
    # print("Let's go EASY_choose_hard!")
    # right_easy_cho = CoMa_choose_HARD_eval(image_path, easy)
    # print(f"EASY_CHOhard_ACC:{right_easy_cho / (len(easy))}")
    ##############################################################


    # JUDGEMENT_DETAIL ###########################################
    # print("Let's go HARD_judgement_detail!")
    # count_gt,count_conf,count_irre = CoMa_judgement_detail_eval(image_path, hard)
    # print(f"HARD_judgement_detail_ACC(gt,conf,irre):\n {count_gt / (len(hard))} \n {count_conf/(len(hard)*4)} \n {count_irre/(len(hard)*2)}")
    #
    # print("Let's go MIDDLE_judgement_detail!")
    # count_gt, count_conf, count_irre = CoMa_judgement_detail_eval(image_path, middle)
    # print(f"MIDDLE_judgement_detail_ACC(gt,conf,irre):\n {count_gt / (len(middle))} \n {count_conf / (len(middle) * 4)} \n {count_irre / (len(middle) * 2)}")
    #
    # print("Let's go EASY_judgement_detail!")
    # count_gt, count_conf, count_irre = CoMa_judgement_detail_eval(image_path, easy)
    # print(f"EASY_judgement_detail_ACC(gt,conf,irre):\n {count_gt / (len(easy))} \n {count_conf / (len(easy) * 4)} \n {count_irre / (len(easy) * 2)}")
    ##############################################################

# 测试用例
if __name__ == "__main__":
    image_path = 'CoCo/CoMa_Eval/val2017/'
    json_path = 'CoCo/CoMa_Eval/Eval_Bench_all.json'
    CoMa_eval(image_path,json_path)
