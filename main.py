import torch
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms as tfms
import gradio as gr
from rembg import remove
import cv2

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def gr_magic_mix(input_image, type_prompt, nu=0.95, total_steps=50, guidance_scale=7.5):
  with torch.no_grad():

    #######################################
    # stage 1: remove the background
    input = remove(input_image, alpha_matting=True, alpha_matting_foreground_threshold=200,
                   alpha_matting_background_threshold=50, alpha_matting_erode_size=1)
    trans_mask = input[:,:,3] == 0
    input_image = input.copy()[:,:,:3]
    input_image[trans_mask] = [255, 255, 255]


    #######################################
    # stage 2: semangic mixing
    scheduler.set_timesteps(total_steps)
    # Define the details of the two phases. The first phase generates the rough layout, the second phase fine-tunes towards the prompt.
    t_min = round(0.3 * total_steps)
    t_max = round(0.6 * total_steps)
    layout_steps = list(range(total_steps - t_max, total_steps - t_min))
    fine_tune_steps = list(range(total_steps - t_min, total_steps))

    # Get embeddings for the text prompt
    prompt = "A cute illustration of" + type_prompt + "type pokemon"
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    encoded = pil_to_latent(input_image)
    noise = torch.randn_like(encoded)
    fine_tuned = None

    # First phase: generate the rough layout by interpolating the original image with denoising from the prompt
    for i in layout_steps:
      t = scheduler.timesteps[i]
      noisy_latents = scheduler.add_noise(encoded, noise, timesteps=torch.tensor([t]))
      if fine_tuned is not None:
        noisy_latents = nu * fine_tuned + (1-nu) * noisy_latents
      model_input = torch.cat([noisy_latents] * 2) 
      model_input = scheduler.scale_model_input(model_input, t)

      noise_pred = unet(model_input, t, encoder_hidden_states=text_embeddings).sample

      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      fine_tuned = scheduler.step(noise_pred, t, noisy_latents).prev_sample

    after_layout = fine_tuned

    # Second phase: fine-tune towards the prompt
    for i in fine_tune_steps:
      t = scheduler.timesteps[i]
      model_input = torch.cat([fine_tuned] * 2)
      model_input = scheduler.scale_model_input(model_input, t)

      noise_pred = unet(model_input, t, encoder_hidden_states=text_embeddings).sample

      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      fine_tuned = scheduler.step(noise_pred, t, fine_tuned).prev_sample

  return latents_to_pil(fine_tuned)[0]



if __name__ == "__main__":

  # Supress some unnecessary warnings when loading the CLIPTextModel
  logging.set_verbosity_error()
  # Set device
  torch_device = "cuda" if torch.cuda.is_available() else "cpu"
  '''
  # Load the autoencoder model which will be used to decode the latents into image space. 
  vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
  # Load the tokenizer and text encoder to tokenize and encode the text. 
  tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
  text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
  # The UNet model for generating the latents.
  unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

  vae = vae.to(torch_device)
  text_encoder = text_encoder.to(torch_device)
  unet = unet.to(torch_device)
  '''

  # The noise scheduler
  scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

  demo = gr.Interface(
    fn=gr_magic_mix,
    inputs=[gr.Image(shape=(512, 512)), gr.Textbox(lines=1, placeholder="Pokemon Type")],
    outputs=["image"]
  )
  demo.launch(share=True, debug=True)