from PIL import Image, ImageDraw
from PIL import ImageFilter
import numpy as np

question_template_simple = ('Based on the contextual clues surrounding the black rectangular mask in the image, infer the most likely object hidden beneath it. '
                            '\Provide your thinking process in <think></think>, and your answer as a concise phrase in <answer></answer>.')

question_template_simple_new = (f'Based on the contextual clues surrounding the black rectangular mask in the image, infer the most likely object hidden beneath it. '
                    f'Please use clues such as the environment and mask size in the image to make inferences.'
                    f'Provide your thinking process in <think></think>, and your answer as a concise phrase in <answer></answer>.')

question_template_reason = ('Analyze the visual environment around the black rectangular mask in the image and infer the most likely object hidden beneath it. Please provide:'
                            '1. A description of the adjacent objects and environmental features near the mask area'
                            '2. An analysis of the mask dimensions and spatial relationships'
                            '3. Several common object candidates in this type of scenario and an analysis of their likelihood'
                            '4. Make a final inference based on logical reasoning and common sense regarding the surrounding environment in the image')

R1V_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)



def soft_mask(original_img, bbox):
    cleaned_str = bbox.strip("[]")
    split_parts = [part.strip() for part in cleaned_str.split(",")]
    width, height = original_img.size
    bbox = list(map(int, split_parts))
    bbox = (int(bbox[0] / 1000 * width), int(bbox[1] / 1000 * height), int(bbox[2] / 1000 * width),int(bbox[3] / 1000 * height))
    mask = Image.new('L', original_img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(bbox, fill=255)
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    original_img.paste("black", mask=blurred_mask)
    return original_img




def soft_mask_coco(original_img, bbox):
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1]))
    mask = Image.new('L', original_img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(bbox, fill=255)
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    original_img.paste("black", mask=blurred_mask)
    return original_img



# def soft_clip_coco(original_img, bbox):
#     bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1]))
#     cropped_img = original_img.crop(bbox)
#     return cropped_img


def soft_clip_coco(original_img, bbox, expand_mode="percent", expand_value=2, max_pixels=50):

    x, y, w, h = bbox
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h

    # 计算扩展量
    if expand_mode == "percent":
        expand_x = int(w * expand_value / 2)
        expand_y = int(h * expand_value / 2)
    elif expand_mode == "fixed":
        expand_x = expand_y = expand_value
    elif expand_mode == "hybrid":
        expand_x = min(int(w * expand_value / 2), max_pixels)
        expand_y = min(int(h * expand_value / 2), max_pixels)
    else:
        raise ValueError("Invalid expand_mode. Use 'percent', 'fixed' or 'hybrid'")

    img_width, img_height = original_img.size
    new_x_min = max(0, x_min - expand_x)
    new_y_min = max(0, y_min - expand_y)
    new_x_max = min(img_width, x_max + expand_x)
    new_y_max = min(img_height, y_max + expand_y)

    expanded_bbox = (new_x_min, new_y_min, new_x_max, new_y_max)
    return original_img.crop(expanded_bbox)




def alpha_clip_mask(original_img, bbox, filename, expand_mode="percent", expand_value=1.0, max_pixels=50):
    x, y, w, h = bbox
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h

    if expand_mode == "percent":
        expand_x = int(w * expand_value / 2)
        expand_y = int(h * expand_value / 2)
    elif expand_mode == "fixed":
        expand_x = expand_y = expand_value
    elif expand_mode == "hybrid":
        expand_x = min(int(w * expand_value / 2), max_pixels)
        expand_y = min(int(h * expand_value / 2), max_pixels)
    else:
        raise ValueError("Invalid expand_mode. Use 'percent', 'fixed' or 'hybrid'")

    img_width, img_height = original_img.size
    new_x_min = max(0, x_min - expand_x)
    new_y_min = max(0, y_min - expand_y)
    new_x_max = min(img_width, x_max + expand_x)
    new_y_max = min(img_height, y_max + expand_y)

    expanded_bbox = [
        (new_x_min,new_y_min),
        (new_x_max, new_y_min),
        (new_x_max, new_y_max),
        (new_x_min,new_x_max)
    ]

    mask = Image.new('L', original_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(expanded_bbox, fill=255)

    if filename == '000000366406.jpg':
        mask.save('mask.png')
    return mask