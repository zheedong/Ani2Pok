import torch
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from torchvision import transforms as tfms

import numpy as np
import gradio as gr

from models.magicmix_colab.modeling_magicmix import *

demo = gr.Interface(
    fn=gr_magic_mix,
    inputs=[gr.Image(shape=(538, 836)), gr.Textbox(lines=1, placeholder="Pokemon Type")],
    outputs=["image"]
)
demo.launch(share=True, debug=True)