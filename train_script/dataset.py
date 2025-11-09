from datasets import load_dataset,Dataset,load_from_disk

from LLama_factory.patent_data_script.check_patent_ipc_script import file_name
from image_utils import *
from src.open_r1.constants import question_template_registry
import json


#MED_DATASET
def get_med_dataset():
    ds = load_dataset("BUAADreamer/llava-med-zh-instruct-60k", split="train[0:2000]", cache_dir='./data').select(
        range(3200))
    def get_prompt_rft(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        dialogue_num = len(example['messages'])
        i = 0
        results = []
        while i < dialogue_num:
            assert example['messages'][i]['role'] == 'user' and example['messages'][i + 1]['role'] == 'assistant'
            question_sample = example['messages'][i]['content']
            answer_sample = example['messages'][i + 1]['content']
            img_pil = example['images'][0].resize((112, 112))  # reduce vRAM burden
            out_results = []
            SYSTEM_PROMPT = r'''
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Before answering, think carefully about the question and create a step-by-step chain of 
            thoughts to ensure a logical and accurate response.

            ### Instruction:
            You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
            Please answer the following medical question based on the input image. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output format should be as follows:
            <think> ... </think> <answer>...</answer>
            除了特殊符号，请用中文回答
            '''  # for a different language, please change the last few words.
            results.append({
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },
                        {"type": "text", "text": question_sample},
                    ]}
                ],
                'image': img_pil,
                'solution': answer_sample,
            })
            i += 2
        return results

    def dataset_gen():
        for items in ds:
            multiple_out = get_prompt_rft(items)
            for single_out in multiple_out:
                yield single_out

    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train









#MASK_DATASET
def get_mask_dataset():
    # ds = load_dataset("BUAADreamer/llava-med-zh-instruct-60k", split="train[0:2000]", cache_dir='./data').select(range(100))
    ds = load_from_disk('/sshfs/datasets/YJA_dataset/PR1_grounding/datasets_Kangheng_PR1-Datasets-Grounding/')['train'].select(range(300))
    def get_prompt_rft_mask(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        problem_dataset = example['problem']
        solution_dataset = example['solution']
        image_pil = example['image']
        image_masked = soft_mask(image_pil,solution_dataset)
        question_template = question_template_simple
        result = {
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": R1V_SYS}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },
                        {"type": "text", "text": question_template},
                    ]}
                ],
                'image': image_masked,
                'solution': problem_dataset,
            }
        return result
    def dataset_gen():
        for items in ds:
            single_out = get_prompt_rft_mask(items)
            yield single_out
    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train




#COCO_DATASET
def get_coco_object_dection_dataset():
    # with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_rl_plus.json', 'r',encoding='utf-8') as file:
    with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_rl_plus_1600.json', 'r',encoding='utf-8') as file:
        ds = json.load(file)#.select(range(300))
    # ds = ds[:500]
    def get_prompt_rft_mask(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        question_template = question_template_simple
        image_root_path = '/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/train2017/'
        file_name = image_root_path + example['file_name']
        image_pil = Image.open(file_name)
        image_masked = soft_mask_coco(image_pil, example['bbox'])
        category_name = example['category_name']

        result = {
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": R1V_SYS}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },
                        {"type": "text", "text": question_template},
                    ]}
                ],
                'image': image_masked,
                'solution': category_name,
            }
        return result
    def dataset_gen():
        for items in ds:
            single_out = get_prompt_rft_mask(items)
            yield single_out
    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train







#COCO_DATASET
def get_coco_dataset():
    # with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_rl_plus.json', 'r',encoding='utf-8') as file:
    with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_rl_plus_1600.json', 'r',encoding='utf-8') as file:
        ds = json.load(file)#.select(range(300))
    # ds = ds[:500]
    def get_prompt_rft_mask(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        question_template = question_template_simple
        image_root_path = '/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/train2017/'
        file_name = image_root_path + example['file_name']
        image_pil = Image.open(file_name)
        image_masked = soft_mask_coco(image_pil, example['bbox'])
        category_name = example['category_name']

        result = {
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": R1V_SYS}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },
                        {"type": "text", "text": question_template},
                    ]}
                ],
                'image': image_masked,
                'solution': category_name,
            }
        return result
    def dataset_gen():
        for items in ds:
            single_out = get_prompt_rft_mask(items)
            yield single_out
    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train



#COCO_DATASET
def get_coco_sft_dataset():
    # with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_filter_step2.json', 'r', encoding='utf-8') as file:
    with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_Cot_plus_withCOT.json', 'r',encoding='utf-8') as file:
        ds = json.load(file)#.select(range(300))
    # ds = ds[:30]
    def get_prompt_rft_mask(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        question_template = question_template_simple

        image_masked = "/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/Cot_mask_image/" + example['file_name']

        result = {
                'messages': [
                    # {'role': 'system', 'content': R1V_SYS},
                    {'role': 'user', 'content': R1V_SYS + question_template},
                    {'role': 'assistant', 'content': '<think> ' + example['cot_data'] + '</think><answer>' + example['category_name'] + '</answer>'},

                ],
                'image': image_masked
            }
        return result
    def dataset_gen():
        for items in ds:
            single_out = get_prompt_rft_mask(items)
            yield single_out
    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train





#COCO_DATASET
def get_coco_sfpo_dataset():
    with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_Cot_plus_withCOT.json', 'r',encoding='utf-8') as file:
        ds_1 = json.load(file)#.select(range(300))
    # with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_rl_plus.json', 'r',encoding='utf-8') as file:
    #     ds_2 = json.load(file)#.select(range(300))
    # ds = ds_1 + ds_2
    ds = ds_1
    # ds = ds[:500]
    def get_prompt_rft_mask(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        question_template = question_template_simple
        image_root_path = '/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/train2017/'
        file_name = image_root_path + example['file_name']
        image_pil = Image.open(file_name)
        image_masked = soft_mask_coco(image_pil, example['bbox'])
        category_name = example['category_name']

        if 'cot_data' in example:
            sft_answer = '<think> ' + example['cot_data'] + '</think><answer>' + example['category_name'] + '</answer>'
        else:
            sft_answer = 'NOCOT'

        result = {
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": R1V_SYS}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },
                        {"type": "text", "text": question_template},
                    ]}
                ],
                'image': image_masked,
                'solution': category_name,
                'sft_answer': sft_answer
            }
        return result
    def dataset_gen():
        for items in ds:
            single_out = get_prompt_rft_mask(items)
            yield single_out
    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train





#COCO_DATASET
def get_coco_clip_dataset():
    # with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_Cot_plus_withCOT.json', 'r',encoding='utf-8') as file:
    #     ds_1 = json.load(file)#.select(range(300))
    with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_rl_plus_1600.json', 'r',encoding='utf-8') as file:
        ds_2 = json.load(file)#.select(range(300))
    ds = ds_2
    def get_prompt_rft_mask(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        question_template = question_template_simple
        image_root_path = '/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/train2017/'
        file_name = image_root_path + example['file_name']
        image_pil = Image.open(file_name)
        image_masked = soft_mask_coco(image_pil, example['bbox'])
        category_name = example['category_name'] + '#' + example['supercategory']

        image_cliped = soft_clip_coco(image_pil, example['bbox'])

        if 'cot_data' in example:
            sft_answer = '<think> ' + example['cot_data'] + '</think><answer>' + example['category_name'] + '</answer>'
        else:
            sft_answer = 'NOCOT'

        result = {
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": R1V_SYS}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },
                        {"type": "text", "text": question_template},
                    ]}
                ],
                'image': image_masked,
                'solution': category_name,
                'sft_answer': sft_answer,
                'clip_image' : image_cliped
            }
        return result
    def dataset_gen():
        for items in ds:
            single_out = get_prompt_rft_mask(items)
            yield single_out
    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train





#COCO_DATASET
def get_coco_clip_v3_dataset():
    # with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_Cot_plus_withCOT.json', 'r',encoding='utf-8') as file:
    #     ds_1 = json.load(file)#.select(range(300))
    with open('/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/CoMa_train_rl_plus_1600.json', 'r',encoding='utf-8') as file:
        ds_2 = json.load(file)#.select(range(300))
    ds = ds_2
    def get_prompt_rft_mask(example):
        '''
        input: dict example, including PIL image object
        output: multiple samples, within a dict format
        '''
        question_template = question_template_simple
        image_root_path = '/sshfs/datasets/YJA_dataset/CoCo/CoCo2017/train2017/'
        file_name = image_root_path + example['file_name']
        image_pil = Image.open(file_name)
        image_masked = soft_mask_coco(image_pil, example['bbox'])
        category_name = example['category_name'] + '#' + example['supercategory']

        image_alpha = alpha_clip_mask(image_pil, example['bbox'], example['file_name'])

        if 'cot_data' in example:
            sft_answer = '<think> ' + example['cot_data'] + '</think><answer>' + example['category_name'] + '</answer>'
        else:
            sft_answer = 'NOCOT'

        result = {
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": R1V_SYS}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },
                        {"type": "text", "text": question_template},
                    ]}
                ],
                'image': image_masked,
                'solution': category_name,
                'sft_answer': sft_answer,
                'orgin_image' : image_pil,
                'alpha': image_alpha,
                'filename': example['file_name']
            }
        return result
    def dataset_gen():
        for items in ds:
            single_out = get_prompt_rft_mask(items)
            yield single_out
    my_gen = dataset_gen()
    dataset_train = Dataset.from_generator(dataset_gen)
    return dataset_train