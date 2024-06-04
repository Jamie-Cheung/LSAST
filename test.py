import argparse, os, sys, glob
import PIL
import cv2
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
import einops
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from transformers import CLIPProcessor, CLIPModel
from cldm.model import create_model, load_state_dict
class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)
apply_canny = CannyDetector()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cpu=torch.device("cpu")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512,512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/sd1.5/v1-5-pruned.ckpt"
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")
sampler = DDIMSampler(model)
canny_model_path = "./models/controlnet/control_v11p_sd15_canny.pth"


def main(prompt='', content_dir=None, ddim_steps=50, strength=0.5, ddim_eta=0.0, n_iter=1, C=4, f=8, n_rows=0,
         scale=10.0, \
         model=None, seed=42, prospect_words=None, n_samples=1, height=512, width=512):
    precision = "autocast"
    outdir = "outputs/comparison-ukiyoe-0.7"
    seed_everything(seed)
    controlnet_canny = create_model(
        './configs/controlnet/control_canny.yaml')
    model_resume_state = torch.load(canny_model_path, map_location='cpu')
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) + 10

    if content_dir is not None:
        content_name = content_dir.split('/')[-1].split('.')[0]
        content_image = load_img(content_dir).to(device)
        content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
        content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

        init_latent = content_latent

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")



    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c = model.get_learned_conditioning(prompts, prospect_words=prospect_words)
                        img = cv2.imread(content_dir)
                        img = cv2.resize(img, (512, 512))
                        # H, W, C = img.shape
                        detected_map = apply_canny(img, 100, 200)  # 100  200
                        # cv2.imwrite('out.png', detected_map)
                        image = detected_map[:, :, None]
                        image = np.concatenate([image, image, image], axis=2)
                        # img1 = rearrange(detected_map, 'h w c ->c h w')

                        control = torch.from_numpy(image.copy()).float().cuda() / 255.0

                        control = torch.stack([control for _ in range(n_samples)], dim=0)
                        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                        new_state_dict = {}
                        for key, value in model_resume_state.items():
                            if key.startswith('control_model.'):
                                new_key = key[len('control_model.'):]  # 去掉前缀
                                new_state_dict[new_key] = value
                        controlnet_canny.load_state_dict(new_state_dict)
                        controlnet_canny = controlnet_canny.to(device)
                        # print("c",len(c))
                        # print("c",c[0].size())
                        # dsa
                        # img2img
                        # t_enc = int(strength * 1000)
                        # x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc] * batch_size).to(device))
                        # model_output = model.apply_model(x_noisy, torch.tensor([t_enc] * batch_size).to(device), c)
                        # z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device), noise=model_output, use_original_steps=True)

                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, controlnet_canny, control, unconditional_guidance_scale=scale,unconditional_conditioning=uc,)
                        
                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))
                                # output.save(os.path.join(outpath, content_name+'-'+prompt+f'-{grid_count:04}.png'))
                output.save(os.path.join(outpath, content_name + 'stylized.jpg'))
                grid_count += 1

                toc = time.time()
    return output 
model.cpu()
model.embedding_manager.load('./logs/berthe-morisot2023-12-25T08-48-51_test/checkpoints/embeddings_gs-99999.pt')
model = model.to(device)
for i in range(2650):
    contentdir = "./comparison/" + str(i) + ".jpg"
    # main(prompt = '*', content_dir = contentdir, style_dir = contentdir, ddim_steps = 50, strength = 0.7, seed=42, model = model)
    main(prompt = '*', \
     content_dir = contentdir, \
     ddim_steps = 50, \
     strength = 0.7, \
     seed=41, \
     height = 512, \
     width = 768, \
     prospect_words = ['*'],
     model = model,\
     )
    

