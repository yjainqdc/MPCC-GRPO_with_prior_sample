import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from image_utils import *
import os
from tqdm import tqdm



# 参数配置
base_model_path = '/sshfs/pretrains/Qwen/Qwen2.5-VL-3B-Instruct/'
# '/sshfs/pretrains/Qwen/Qwen2.5-VL-7B-Instruct/'
# '/sshfs/pretrains/Qwen/Qwen2.5-VL-3B-Instruct/'
# base_model_path = '/sshfs/jiaao/workdir/Personal/qwen2.5vl/outputs/checkpoint-100/'
# adapter_path = "/sshfs/jiaao/workdir/Personal/qwen2.5vl/outputs/checkpoint-100/"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 量化配置（保持与训练一致）
compute_dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# 加载基础模型
basemodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    device_map=device,
    torch_dtype=compute_dtype,
    # quantization_config=bnb_config,
    attn_implementation="flash_attention_2"  # 启用加速[6,7](@ref)
)

# 合并LoRA适配器
# loramodel = PeftModel.from_pretrained(basemodel, adapter_path)
# model = loramodel.merge_and_unload()  # 合并适配器到基础模型
model = basemodel


# 加载处理器（保持与训练相同的视觉token配置）
processor = AutoProcessor.from_pretrained(
    base_model_path,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28
)


def generate_response(image_path, prompt,mask_task=False):
    # 准备多模态输入
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

    # 预处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt"
    ).to(device)

    # 生成配置（根据GRPO训练参数优化）
    generation_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "use_cache": True
    }

    # 执行推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )

    # 解码输出
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return response.strip()








###########################################################################
def postcess_llm_answer(output,reason_type=False):
    # print(output)
    if reason_type:
        reason_answer =  output.split('assistant')[2]
        if '<answer>' in reason_answer:
            answer = reason_answer.split('<answer>')[1].split('</answer>')[0]
        else:
            answer = reason_answer
        if 'yes' in answer.lower():
            answer = 'yes'
        elif 'no' in answer.lower():
            answer = 'no'
    else:
        answer = output.split('assistant')[2]
        if 'yes' in answer.lower():
            answer = 'yes'
        elif 'no' in answer.lower():
            answer = 'no'
    return answer

###########################################################################
#################################CoMa_eval#################################


def CoMa_eval():
    image_path = '/sshfs/datasets/YJA_dataset/CoCo/CoMa_Eval/val2017/'
    json_path = '/sshfs/datasets/YJA_dataset/CoCo/CoMa_Eval/Eval_bench_Bbox.json'
    with open(json_path) as f:
        annotations = json.load(f)
    hard = annotations['hard']
    middle = annotations['middle']
    easy = annotations['easy']
    #Hard
    count_hard = 0
    for ann in tqdm(hard):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        name = ann['category_name']
        question = f'I have covered A part of the image with black. Please guess if this is {name}. Please answer me with yes or no.'
        llm_answer = generate_response(mask_img, question, mask_task=False)
        # print(llm_answer)
        res = postcess_llm_answer(llm_answer)
        if res == 'yes':
            count_hard += 1
    print('Hard的准确率：')
    print(count_hard / len(hard))
    #Middle
    count_middle = 0
    for ann in tqdm(middle):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        name = ann['category_name']
        question = f'I have covered A part of the image with black. Please guess if this is {name}. Please answer me with yes or no.'
        llm_answer = generate_response(mask_img, question, mask_task=False)
        # print(llm_answer)
        res = postcess_llm_answer(llm_answer)
        if res == 'yes':
            count_middle += 1
    print('Middle的准确率：')
    print(count_middle / len(middle))
    #Easy
    count_easy = 0
    for ann in tqdm(easy):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        name = ann['category_name']
        question = f'I have covered A part of the image with black. Please guess if this is {name}. Please answer me with yes or no.'
        llm_answer = generate_response(mask_img, question, mask_task=False)
        # print(llm_answer)
        res = postcess_llm_answer(llm_answer)
        if res == 'yes':
            count_easy += 1
    print('Easy的准确率：')
    print(count_easy / len(easy))

    print('总准确率：')
    print((count_hard + count_middle + count_easy) / 1114)


def CoMa_QA():
    image_path = '/sshfs/datasets/YJA_dataset/CoCo/CoMa_Eval/val2017/'
    json_path = '/sshfs/datasets/YJA_dataset/CoCo/CoMa_Eval/Eval_Bench_new.json'
    with open(json_path) as f:
        annotations = json.load(f)
    hard = annotations['hard']
    middle = annotations['middle']
    easy = annotations['easy']
    #Hard
    new_hard = []
    for ann in tqdm(hard):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img,ann['bbox'])
        question = f'I have covered A part of the image with black. Please guess what it is. Please answer ten possible answers in a [], separate with commas. Answer directly.'
        llm_answer = generate_response(mask_img, question, mask_task=False)
        # print(llm_answer)
        res = llm_answer.split('assistant')[2]
        print(res)
        #搞个json
        ann['judgement_negtive_confusion'] = []
        ann['choose_negtive_confusion'] = []
        ann['irrelevant_negtive'] = []
        ann['confusion'] = res
        new_hard.append(ann)

    # Middle
    new_middle = []
    for ann in tqdm(middle):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img, ann['bbox'])
        question = f'I have covered A part of the image with black. Please guess what it is. Please answer ten possible answers in a [], separate with commas. Answer directly.'
        llm_answer = generate_response(mask_img, question, mask_task=False)
        # print(llm_answer)
        res = llm_answer.split('assistant')[2]
        print(res)
        # 搞个json
        ann['judgement_negtive_confusion'] = []
        ann['choose_negtive_confusion'] = []
        ann['irrelevant_negtive'] = []
        ann['confusion'] = res
        new_middle.append(ann)

    # Easy
    new_easy = []
    for ann in tqdm(easy):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img, ann['bbox'])
        question = f'I have covered A part of the image with black. Please guess what it is. Please answer ten possible answers in a [], separate with commas. Answer directly.'
        llm_answer = generate_response(mask_img, question, mask_task=False)
        # print(llm_answer)
        res = llm_answer.split('assistant')[2]
        print(res)
        # 搞个json
        ann['judgement_negtive_confusion'] = []
        ann['choose_negtive_confusion'] = []
        ann['irrelevant_negtive'] = []
        ann['confusion'] = res
        new_easy.append(ann)


    #保存一下
    data_choosen = {
        'hard': new_hard,
        'middle': new_middle,
        'easy': new_easy
    }
    with open(f'./Eval_Bench_confusion.json', 'w') as f:
        json.dump(data_choosen, f, indent=4)





# 测试用例
if __name__ == "__main__":

    # CoMa_eval()
    CoMa_QA()