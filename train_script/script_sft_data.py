import json
from image_utils import *

with open('/CoCo/CoCo2017/CoMa_train_Cot_plus_withCOT.json', 'r', encoding='utf-8') as file:
    ds = json.load(file)


data_instruct = []

for example in ds:
    question_template = question_template_simple
    image_root_path = '/CoCo2017/train2017/'
    file_name = image_root_path + example['file_name']
    image_pil = Image.open(file_name)
    image_masked = soft_mask_coco(image_pil, example['bbox'])

    mask_image_save = "/CoCo2017/Cot_mask_image/"
    image_masked_path = mask_image_save + example['file_name']
    image_masked.save(image_masked_path)
    result = {
        'messages': [
            {'role': 'system', 'content': R1V_SYS},
            {'role': 'user', 'content': '<image> ' + question_template},
            {'role': 'assistant',
             'content': '<think> ' + example['cot_data'] + '</think><answer>' + example['category_name'] + '</answer>'},

        ],
        'images': [image_masked_path]
    }

    data_instruct.append(result)

with open(f'./eval/cot_data_json/CoMa_train_Cot_instruct.json', 'w') as f:
    json.dump(data_instruct, f, indent=4)