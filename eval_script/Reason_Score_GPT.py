




import json
import os
import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import random
from openai import OpenAI
import base64
from io import BytesIO
from eval_utils import *



'''
gpt4o
'''


client = OpenAI(

    api_key="",
    base_url=""

)


def pil_to_base64(pil_image: Image.Image, format: str = "JPEG") -> str:
    buffered = BytesIO()
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format=format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


def generate_response(image, prompt,mask_task=False):
    img_base64 = pil_to_base64(image)
    response = client.chat.completions.create(
        model="gpt-4o-ca",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content


























###########################################################################
def postcess_llm_answer(output):
    answer = output.split('assistant')[-1]
    return answer










def Score_eval(image_path ,list):

    count = 0
    lenlist = 48

    for ann in tqdm(list):
        file_name = ann['file_name']
        img_path = os.path.join(image_path, file_name)
        img = Image.open(img_path)
        mask_img = soft_mask_coco(img ,ann['bbox'])

        gt = ann['gt']
        res = ann['res'].split('</think>')
        think = res[0].replace('<think>' ,'').replace('\n' ,' ')

        # answer = res[-1].split('nswer')[-1]
        answer = res[1].replace('<answer>' ,'').replace('</answer>' ,'').replace('\n' ,' ')

        question = (
            "Please act as an expert in multimodal reasoning quality assessment and conduct a 0-10 quantitative scoring (10 being the best) "
            "on the model's output reasoning process and conclusion based on the following dimensions.rovide an overall score and the rationale for the scoring:"
            "Visual context relevance: Does the reasoning strictly rely on visual cues in the image (e.g., object positions, scene features, information around the masked area)? "
            "Is there any subjective speculation divorced from the image content?"
            "Effectiveness of commonsense reasoning: Does it accurately apply commonsense related to the scene? "
            "Is the combination of commonsense and visual cues reasonable?"
            "Logical coherence: Are the reasoning steps coherently connected?"
            "Is there any logical gap or contradiction (e.g., first reasoning 'small items' and then identifying 'large furniture')?"
            "Conclusion consistency: Is the final answer reasonably derived from the reasoning process?"
            "Does it accurately respond to the task requirements (e.g., 'predict masked objects' 'judge spatial relationships')?"

            "This task is to predict the masked object by reasoning through the image context and common sense given an image with a masked part."
            f"Below is an example of reasoning and answering based on a given image, which ground truth is {gt}. Please score and evaluate it:"
            "Reasoning process:"
            f"{think}\n"
            "Answer result:"
            f"{answer}\n"


            "Please directly give me your score with a number of 0-10 , do not other words."
        )
        llm_answer = generate_response(mask_img, question, mask_task=False)
        res = postcess_llm_answer(llm_answer)

        print(res)

        try:
            # score = int(res.split("\n")[0])
            score = int(res)
        except:
            score = 0
            lenlist = lenlist -1

        count = count + score

    print(f"All_score: {count},Ave_score: {count / lenlist}")


def CoMa_eval(image_path, json_path):
    with open(json_path) as f:
        annotations = json.load(f)
    choosed = []
    file_list = ['000000229358', '000000309391', '000000512476', '000000053529', '000000193162', '000000112798',
                 '000000315257',
                 '000000271728', '000000139872', '000000368335', '000000017029', '000000338624', '000000473219',
                 '000000575357',
                 '000000153797', '000000236721', '000000222094', '000000113235', '000000226903', '000000402720',
                 '000000166287',
                 '000000411665', '000000133567', '000000129062', '000000273198', '000000569565', '000000017379',
                 '000000462614',
                 '000000082807', '000000295478', '000000233033', '000000113403', '000000091495', '000000157138',
                 '000000162732',
                 '000000214539', '000000438862', '000000173383', '000000229753', '000000264535', '000000017178',
                 '000000017899',
                 '000000054628', '000000130579', '000000404249', '000000440336', '000000006954', '000000017115']
    for i in annotations:
        if i['file_name'].split('.')[0] in file_list:
            choosed.append(i)

    choosed = choosed[5:10]

    Score_eval(image_path, choosed)


# 测试用例
if __name__ == "__main__":
    image_path = '/sshfs/datasets/YJA_dataset/CoCo/CoMa_Eval/val2017/'
    # json_path = '/sshfs/jiaao/workdir/Personal/qwen2.5vl/eval/Reasoning_save_prompt.json'
    json_path = '/sshfs/jiaao/workdir/Personal/qwen2.5vl/eval/Reasoning_save_sft.json'
    # json_path = '/sshfs/jiaao/workdir/Personal/qwen2.5vl/eval/Reasoning_save_rl.json'
    # json_path = '/sshfs/jiaao/workdir/Personal/qwen2.5vl/eval/Reasoning_save_sft+rl.json'
    # json_path = '/sshfs/jiaao/workdir/Personal/qwen2.5vl/eval/Reasoning_save_sfpo.json'

    CoMa_eval(image_path, json_path)
