import os
import pathlib
import random
import numpy as np
import wget
import zipfile
import glob

from functions.training.config import *
from PIL import Image 

cache = None

def format_input(input_image, _type = 'float64'):
    assert(input_image.size == default_in_shape[:2] or input_image.shape == default_in_shape)
    inp = np.array(input_image)
    if inp.shape[-1] == 4:
        input_image = input_image.convert('RGB')
    return np.array(np.expand_dims(np.array(input_image)/255., 0)).astype(_type)

def get_image_mask_pair(img_name, in_resize=None, out_resize=None, augment=True):
    in_img = image_dir.joinpath(img_name)
    out_img = mask_dir.joinpath(img_name.replace('jpg', 'png'))

    if not in_img.exists() or not out_img.exists():
        return None

    img  = Image.open(image_dir.joinpath(img_name))
    mask = Image.open(out_img)

    if in_resize:
        img = img.resize(in_resize[:2], Image.BICUBIC)
    
    if out_resize:
        mask = mask.resize(out_resize[:2], Image.BICUBIC)

    # the paper specifies the only augmentation done is horizontal flipping.
    if augment and bool(random.getrandbits(1)):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return (np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1))

def load_training_batch(batch_size=12, in_shape=default_in_shape, out_shape=default_out_shape):
    global cache
    if cache is None:
        cache = os.listdir(image_dir)
    
    imgs = random.choices(cache, k=batch_size)
    image_list = [get_image_mask_pair(img, in_resize=default_in_shape, out_resize=default_out_shape) for img in imgs]
    
    tensor_in  = np.stack([i[0]/255. for i in image_list])
    tensor_out = np.stack([i[1]/255. for i in image_list])
    
    return (tensor_in, tensor_out)