import os

import model_management
import torch
import comfy.sd
import comfy.utils
import folder_paths
import comfy.samplers
from nodes import common_ksampler
from comfy_extras.chainner_models import model_loading

from PIL import Image, ImageOps, ImageFilter, ImageDraw
from PIL.PngImagePlugin import PngInfo
import numpy as np
from torchvision.transforms import ToPILImage
import cv2
from deepface import DeepFace

import re
import latent_preview
from datetime import datetime
import json
import re
import piexif
import piexif.helper


MAX_RESOLUTION=8192

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            
# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# model io
class WLSH_Checkpoint_Loader_Model_Name:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE","STRING",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "WLSH Nodes/loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        name = self.parse_name(ckpt_name)
        
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        new_out = list(out)
        new_out.pop()
        new_out.append(name)
        out = tuple(new_out)

        return (out)

    def parse_name(self, ckpt_name):
        path = ckpt_name
        filename = path.split("/")[-1]
        filename = filename.split(".")[:-1]
        filename = ".".join(filename)
        return filename

# sampling
class WLSH_KSamplerAdvancedMod:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "seed": ("SEED", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "WLSH Nodes/sampling"

    def sample(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise):
        noise_seed = seed['seed']
        force_full_denoise = False
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
        denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


class WLSH_Alternating_KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "seed": ("SEED", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "clip": ("CLIP", ),
                    "positive_prompt": ("STRING", {"forceInput": True }),
                    "negative_prompt": ("STRING", {"forceInput": True }),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "WLSH Nodes/sampling"

    def sample(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler, clip, positive_prompt, negative_prompt, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise):
        noise_seed = seed['seed']
        force_full_denoise = False
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        # alternating prompt parser
        # syntax: {A|B} will sequentially alternate between A and B
        def parse_prompt(input_string, stepnum):
            def replace_match(match):
                options = match.group(1).split('|')
                return options[(stepnum - 1) % len(options)]

            pattern = r'<(.*?)>'
            parsed_string = re.sub(pattern, replace_match, input_string)
            return parsed_string
        
        latent_input = latent_image
        for step in range(0,steps):
            positive_txt = parse_prompt(positive_prompt,step+1)
            positive = [[clip.encode(positive_txt), {}]]
            negative_txt = parse_prompt(negative_prompt,step+1)
            negative = [[clip.encode(negative_txt), {}]]

            if(step < steps):
                force_full_denoise = False
            if(step > 0):
                disable_noise=True

            latent_image = common_ksampler(model, noise_seed, 1, cfg, sampler_name, scheduler, positive, negative, latent_input,
             denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, 
             force_full_denoise=force_full_denoise)
            
            latent_input = latent_image[0]
        
        return latent_image
        # return alternating_ksampler(clip, model, noise_seed, steps, cfg, sampler_name, scheduler, positive_prompt, negative_prompt, latent_image, 
        # denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

# utilities
class WLSH_Seed_to_Number:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("SEED",),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "number_to_seed"

    CATEGORY = "WLSH Nodes"

    def number_to_seed(self, seed):
        return (int(seed["seed"]), )

class WLSH_SDXL_Steps:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precondition": ("INT", {"default": 3, "min": 1, "max": 10000}),
                "base": ("INT", {"default": 12, "min": 1, "max": 10000}),
                "total": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }
    RETURN_TYPES = ("INT","INT","INT",)
    FUNCTION = "set_steps"

    CATEGORY="WLSH Nodes"


    def set_steps(self,precondition,base,total):
        return(precondition,base,total)

class WLSH_Int_Multiply:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": ("INT",{"default": 2, "min": 1, "max": 10000}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 10000}),
            }
        }
    RETURN_TYPES = ("INT",)
    FUNCTION = "multiply"

    CATEGORY="WLSH Nodes/number"


    def multiply(self,number,multiplier):
        result = number*multiplier
        return (int(result),)

class WLSH_Time_String:
    time_format = ["%Y%m%d%H%M%S","%Y%m%d%H%M","%Y%m%d","%Y-%m-%d-%H%M%S", "%Y-%m-%d-%H%M", "%Y-%m-%d"]
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style": (s.time_format,),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_time"

    CATEGORY = "WLSH Nodes/text"

    def get_time(self, style):
        now = datetime.now()
        timestamp = now.strftime(style)

        return (timestamp,)

class WLSH_SDXL_Resolutions:
    resolution = ["1024x1024","1152x896","1216x832","1344x768","1536x640"]
    direction = ["landscape","portrait"]    
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (s.resolution,),
                "direction": (s.direction,),
            }
        }
    RETURN_TYPES = ("INT","INT",)
    FUNCTION = "get_resolutions"

    CATEGORY="WLSH Nodes"


    def get_resolutions(self,resolution, direction):
        width,height = resolution.split('x')
        width = int(width)
        height = int(height)
        if(direction == "portrait"):
            width,height = height,width
        return(width,height)

# latent
class WLSH_Empty_Latent_Image_By_Ratio:
    aspects = ["1:1","5:4","4:3","3:2","16:10","16:9","21:9","2:1","3:1","4:1"]
    direction = ["landscape","portrait"]

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "aspect": (s.aspects,),
                              "direction": (s.direction,),
                              "shortside": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "WLSH Nodes/latent"

    def generate(self, aspect, direction, shortside, batch_size=1):
        x,y = aspect.split(':')
        x = int(x)
        y = int(y)
        ratio = x/y
        width = int(shortside * ratio)
        width = (width + 63) & (-64)
        height = shortside
        if(direction == "portrait"):
            width,height = height,width
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples":latent}, )

class WLSH_SDXL_Quick_Empty_Latent:
    resolution = ["1024x1024","1152x896","1216x832","1344x768","1536x640"]
    direction = ["landscape","portrait"]

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "resolution": (s.resolution,),
                              "direction": (s.direction,),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "WLSH Nodes/latent"

    def generate(self, resolution, direction, batch_size=1):
        width,height = resolution.split('x')
        width = int(width)
        height = int(height)
        if(direction == "portrait"):
            width,height = height,width
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples":latent}, )

# conditioning
class WLSH_CLIP_Text_Positive_Negative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("STRING", {"multiline": True}),
         "negative": ("STRING", {"multiline": True}),
         "clip": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","STRING","STRING")
    FUNCTION = "encode"

    CATEGORY = "WLSH Nodes/conditioning"

    def encode(self, clip, positive, negative):
        return ([[clip.encode(positive), {}]],[[clip.encode(negative), {}]],positive,negative) 

class WLSH_CLIP_Positive_Negative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "positive_text": ("STRING",{"default": f'', "multiline": True}),
            "negative_text": ("STRING",{"default": f'', "multiline": True})
            }}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "WLSH Nodes/conditioning"

    def encode(self, clip, positive_text, negative_text):
        return ([[clip.encode(positive_text), {}]],[[clip.encode(negative_text), {}]] )
# upscaling
class WLSH_Image_Scale_By_Factor:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "original": ("IMAGE",), "upscaled": ("IMAGE",),
                              "upscale_method": (s.upscale_methods,),
                              "factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1})
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "WLSH Nodes/upscaling"

    def upscale(self, original, upscaled, upscale_method, factor, crop):
        old_width = original.shape[2]
        old_height = original.shape[1]
        new_width= int(old_width * factor)
        new_height = int(old_height * factor)
        print("Processing image with shape: ",old_width,"x",old_height,"to ",new_width,"x",new_height)
        samples = upscaled.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
        s = s.movedim(1,-1)
        return (s,)

class WLSH_SDXL_Quick_Image_Scale:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    resolution = ["1024x1024","1152x896","1216x832","1344x768","1536x640"]
    direction = ["landscape","portrait"]
    crop_methods = ["disabled", "center"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "original": ("IMAGE",),
                              "upscale_method": (s.upscale_methods,),
                              "resolution": (s.resolution,),
                              "direction": (s.direction,),
                              "crop": (s.crop_methods,),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "WLSH Nodes/upscaling"

    def upscale(self, original, upscale_method, resolution, direction, crop):
        width,height = resolution.split('x')
        new_width = int(width)
        new_height = int(height)
        if(direction == "portrait"):
            new_width,new_height = new_height,new_width
        old_width = original.shape[2]
        old_height = original.shape[1]
        #print("Processing image with shape: ",old_width,"x",old_height,"to ",new_width,"x",new_height)
        samples = original.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop)
        s = s.movedim(1,-1)
        return (s,)


class WLSH_Upscale_By_Factor_With_Model:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",), "image": ("IMAGE",),
                              "upscale_method": (s.upscale_methods,),
                              "factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1})
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "WLSH Nodes/upscaling"

    def upscale(self, image, upscale_model, upscale_method, factor):
        # upscale image using upscaling model
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=128 + 64, tile_y=128 + 64, overlap = 8, upscale_amount=upscale_model.scale)
        upscale_model.cpu()
        upscaled = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)

        # get dimensions of orginal image
        old_width = image.shape[2]
        old_height = image.shape[1]

        # scale dimensions by provided factor
        new_width= int(old_width * factor)
        new_height = int(old_height * factor)
        print("Processing image with shape: ",old_width,"x",old_height,"to ",new_width,"x",new_height)

        # apply simple scaling to image
        samples = upscaled.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
        s = s.movedim(1,-1)
        
        return (s,)
      

# outpainting
class WLSH_Outpaint_To_Image:
    directions = ["left","right","up","down"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "direction": (s.directions,),
                              "pixels": ("INT", {"default": 128, "min": 32, "max": 512, "step": 32}),
                              "mask_padding": ("INT",{"default": 12, "min": 0, "max": 64, "step": 4})
                              }}
    RETURN_TYPES = ("IMAGE","MASK")
    FUNCTION = "outpaint"

    CATEGORY = "WLSH Nodes/inpainting"

    def convert_image(self, im, direction, mask_padding):
        width, height = im.size
        im = im.convert("RGBA")
        alpha = Image.new('L',(width,height),255)
        im.putalpha(alpha)
        return im


    def outpaint(self, image, direction, mask_padding, pixels):
        image = tensor2pil(image)
        # i = 255. * image.cpu().numpy()
        # image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        image = self.convert_image(image, direction, mask_padding)
        if direction == "right":
            border = (0,0,pixels,0)
            new_image = ImageOps.expand(image,border=border,fill=(0,0,0,0))
        elif direction == "left":
            border = (pixels,0,0,0)
            new_image = ImageOps.expand(image,border=border,fill=(0,0,0,0))
        elif direction == "up":
            border = (0,pixels,0,0)
            new_image = ImageOps.expand(image,border=border,fill=(0,0,0,0))
        elif direction == "down":
            border = (0,0,0,pixels)
            new_image = ImageOps.expand(image,border=border,fill=(0,0,0,0))
       
        image = new_image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in new_image.getbands():
            mask = np.array(new_image.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        
        #print("bands: ", new_image.getbands())
        # if 'A' in new_image.getbands():
        #     mask = np.array(new_image.getchannel('A')).astype(np.float32) / 255.0
        #     mask = 1. - torch.from_numpy(mask)
        #     #print("getting mask from alpha")
        # else:
        #     mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        #    # print("generating mask")
        # new_image = new_image.convert("RGB")
        # new_image = pil2tensor(new_image)

        return (image,mask)

class WLSH_VAE_Encode_For_Inpaint_Padding:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", ), "mask": ("MASK", ),
                              "mask_padding": ("INT",{"default": 24, "min": 6, "max": 128, "step": 2})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "WLSH Nodes/inpainting"

    def encode(self, vae, pixels, mask, mask_padding=3):
        x = (pixels.shape[1] // 64) * 64
        y = (pixels.shape[2] // 64) * 64
        mask = torch.nn.functional.interpolate(mask[None,None,], size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")[0][0]

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            pixels = pixels[:,:x,:y,:]
            mask = mask[:x,:y]

        #grow mask by a few pixels to keep things seamless in latent space
        kernel_tensor = torch.ones((1, 1, mask_padding, mask_padding))
        mask_erosion = torch.clamp(torch.nn.functional.conv2d((mask.round())[None], kernel_tensor, padding=3), 0, 1)
        m = (1.0 - mask.round())
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        t = vae.encode(pixels)

        return ({"samples":t, "noise_mask": (mask_erosion[0][:x,:y].round())}, )

class WLSH_Generate_Second_Mask:
    directions = ["left","right","up","down"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "direction": (s.directions,),
                              "pixels": ("INT", {"default": 128, "min": 32, "max": 512, "step": 32}),
                              "overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16})
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gen_second_mask"

    CATEGORY = "WLSH Nodes/inpainting"

    def gen_second_mask(self, direction, image, pixels, overlap):
        image = tensor2pil(image)
        new_width,new_height = image.size
        
        # generate new image fully un-masked
        mask2 = Image.new('RGBA',(new_width,new_height),(0,0,0,255))
        mask_thickness = overlap
        if (direction == "up"):
            # horizontal mask width of new image and height of 1/4 padding
            new_mask = Image.new('RGBA',(new_width, mask_thickness),(0,122,0,255))
            mask2.paste(new_mask,(0,(pixels-int(mask_thickness/2))))
        elif (direction == "down"):
            # horizontal mask width of new image and height of 1/4 padding
            new_mask = Image.new('RGBA',(new_width, mask_thickness),(0,122,0,255))
            mask2.paste(new_mask,(0,new_height-pixels - int(mask_thickness/2)))
        elif (direction == "left"):
            # vertical mask height of new image and width of 1/4 padding
            new_mask = Image.new('RGBA',(mask_thickness,new_height),(0,122,0,255))
            mask2.paste(new_mask,(pixels - int(mask_thickness/2),0))
        elif (direction == "right"):
            # vertical mask height of new image and width of 1/4 padding
            new_mask = Image.new('RGBA',(mask_thickness,new_height),(0,122,0,255))
            mask2.paste(new_mask,(new_width - pixels - int(mask_thickness/2),0))
        mask2 = mask2.filter(ImageFilter.GaussianBlur(radius=5))
        mask2 = np.array(mask2).astype(np.float32) / 255.0
        mask2 = torch.from_numpy(mask2)[None,]
        return (mask2,)

class WLSH_Generate_Face_Mask:
    detectors = ["opencv", "retinaface", "ssd", "mtcnn"]
    channels = ["red", "blue", "green"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                             "detector": (s.detectors,),
                             "channel": (s.channels,),
                             "mask_padding": ("INT",{"default": 6, "min": 0, "max": 32, "step": 2}) 
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gen_face_mask"

    CATEGORY = "WLSH Nodes/inpainting"
    def gen_face_mask(self, image, mask_padding, detector, channel):
        image = tensor2pil(image)
        
        faces = DeepFace.extract_faces(np.array(image),detector_backend=detector)
        # cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # # Convert to grayscale
        # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # # Detect faces in the image
        # face_cascade = cv2.CascadeClassifier('custom_nodes/haarcascade_frontalface_default.xml')
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        mask = Image.new('RGB',image.size)

        # Draw a rectangle on the PIL Image object
        colors = {"red": "RGB(255,0,0)", "green": "RGB(0,255,0)", "blue": "RGB(0,0,255)"}
        draw = ImageDraw.Draw(mask)
        for face in faces:
            x,y,w,h = face['facial_area'].values()
            draw.rectangle((x-mask_padding,y-mask_padding,x+w+mask_padding,y+h+mask_padding), outline=colors[channel], fill=colors[channel])
        mask = mask.filter(ImageFilter.GaussianBlur(radius=6))

        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
        return (mask,)

# image I/O
class WLSH_Image_Save_With_Prompt_Info:
    def __init__(self):
        self.output_dir = os.path.join(os.getcwd()+'/ComfyUI', "output")

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "images": ("IMAGE", ),
                        "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                        "output_path": ("STRING", {"default": './output', "multiline": False}),
                        "extension": (['png', 'jpeg', 'tiff', 'gif'], ),
                        "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                    },
                    "optional": {
                        "positive": ("STRING",{"default": '', "multiline": True}),
                        "negative": ("STRING",{"default": '', "multiline": True}),
                        "seed": ("SEED",),
                        "modelname": ("STRING",{"default": '', "multiline": False}),
                    },
                    "hidden": {
                        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "WLSH Nodes/IO"

    def save_files(self, images, positive="uknown", negative="uknown", seed=0, modelname="uknown", filename=f'%time', output_path="./output",
     extension='png', quality=100, prompt=None, extra_pnginfo=None):
        filename = self.make_filename(seed, modelname, filename)
        comment = self.make_comment(positive, negative, modelname, seed)
        paths = self.save_images(images, output_path,filename,comment, extension, quality, prompt, extra_pnginfo)
        #return
        return { "ui": { "images": paths } }

    def make_comment(self, positive, negative, modelname, seed):
        comment = "Positive Prompt:\n" + positive + "\nNegative Prompt:\n" + negative + "\nModel: " + modelname + "\nSeed: " + str(seed['seed'])
        return comment

    def make_filename(self, seed, modelname, filename):
        # generate datetime string
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

        # parse input string
        filename = filename.replace("%time",timestamp)

        filename = filename.replace("%model",modelname)
        filename = filename.replace("%seed",str(seed['seed']))

        return (filename)   
    def save_images(self, images, output_path='', filename_prefix="ComfyUI", comment="", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
        # Setup custom path or default
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'\033[34mWAS NS\033[0m Error: The path `{output_path.strip()}` specified doesn\'t exist! Defaulting to `{self.output_dir}` directory.')
            else:
                self.output_dir = os.path.normpath(output_path.strip())

        imgCount = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo() 
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            prefix = filename_prefix
            if(images.size()[0] > 1):
                prefix = filename_prefix + "_{:02d}".format(imgCount)

            file = f"{prefix}.{extension}"
            if extension == 'png':
                img.save(os.path.join(self.output_dir, file), comment=comment, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(os.path.join(self.output_dir, file), quality=quality)
            elif extension == 'jpeg':
                img.save(os.path.join(self.output_dir, file), quality=quality, comment=comment, optimize=True)
            elif extension == 'tiff':
                img.save(os.path.join(self.output_dir, file), quality=quality, optimize=True)
            else:
                img.save(os.path.join(self.output_dir, file))
            paths.append(file)
            imgCount += 1
        return(paths)

class WLSH_Image_Save_With_Prompt_File:
    def __init__(self):
        self.output_dir = os.path.join(os.getcwd()+'/ComfyUI', "output")

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "images": ("IMAGE", ),                   
                        "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                        "output_path": ("STRING", {"default": './output', "multiline": False}),
                        "extension": (['png', 'jpeg', 'tiff', 'gif'], ),
                        "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                    },
                    "optional": {
                        "positive": ("STRING",{"default": ' ', "multiline": True}),
                        "negative": ("STRING",{"default": ' ', "multiline": True}),
                        "seed": ("SEED",),
                        "modelname": ("STRING",{"default": 'sd', "multiline": False}),
                    },
                    "hidden": {
                        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "WLSH Nodes/IO"

    def save_files(self, images, positive="unknown", negative="unknown", seed=0, modelname="unknown", filename=f'%time', output_path="./output",
     extension='png', quality=100, prompt=None, extra_pnginfo=None):
        filename = self.make_filename(seed, modelname, filename)
        comment = "Positive Prompt:\n" + positive + "\nNegative Prompt:\n" + negative + "\nModel: " + modelname + "\nSeed: " + str(seed['seed'])
        paths = self.save_images(images, output_path,filename,comment, extension, quality, prompt, extra_pnginfo)
        self.save_text_file(positive, negative, seed, modelname, output_path, filename)
        #return
        return { "ui": { "images": paths } }

    def make_filename(self, seed, modelname, filename):
        # generate datetime string
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

        # parse input string
        filename = filename.replace("%time",timestamp)
        filename = filename.replace("%model",modelname)
        filename = filename.replace("%seed",str(seed['seed']))

        return (filename)   
    def save_images(self, images, output_path='', filename_prefix="ComfyUI", comment="", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
        # Setup custom path or default
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'\033[34mWAS NS\033[0m Error: The path `{output_path.strip()}` specified doesn\'t exist! Defaulting to `{self.output_dir}` directory.')
            else:
                self.output_dir = os.path.normpath(output_path.strip())

        imgCount = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if(images.size()[0] > 1):
                filename_prefix += "_{:02d}".format(imgCount)

            file = f"{filename_prefix}.{extension}"
            if extension == 'png':
                img.save(os.path.join(self.output_dir, file), comment=comment, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(os.path.join(self.output_dir, file), quality=quality)
            elif extension == 'jpeg':
                img.save(os.path.join(self.output_dir, file), quality=quality, comment=comment, optimize=True)
            elif extension == 'tiff':
                img.save(os.path.join(self.output_dir, file), quality=quality, optimize=True)
            else:
                img.save(os.path.join(self.output_dir, file))
            paths.append(file)
            imgCount += 1
        return(paths)

    def save_text_file(self, positive, negative, seed, modelname, path, filename):
        
        # Ensure path exists
        if not os.path.exists(path):
            print(f'\033[34mWAS NS\033[0m Error: The path `{path}` doesn\'t exist!')
            
        # Ensure content to save
        if filename.strip == '':
            print(f'\033[34mWAS NS\033[0m Error: There is no text specified to save! Text is empty.')
        text = "Positive Prompt:\n" + positive + "\nNegative Prompt:\n" + negative + "\nModel: " + modelname + "\nSeed: " + str(seed['seed'])
        
        
        filename = self.make_filename(seed, modelname, filename)   
        # Write text file
        self.writeTextFile(os.path.join(path, filename + '.txt'), text)
        
        return( text, )

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print(f'\033[34mWAS Node Suite\033[0m Error: Unable to save file `{file}`')    

class WLSH_Save_Prompt_File:
    def __init__(s):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "filename": ("STRING",{"default": 'info', "multiline": False}),
                        "path": ("STRING", {"default": './output', "multiline": False}),
                        "positive": ("STRING",{"default": '', "multiline": True}),
                    },
                    "optional": {
                        "negative": ("STRING",{"default": '', "multiline": True}),
                        "modelname": ("STRING",{"default": '', "multiline": False}),
                        "seed": ("SEED",),
                    }
                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"

    CATEGORY = "WLSH Nodes/IO"
    
    def save_text_file(self, positive="", negative="", seed=0, path="./output", modelname="", filename="info"):
        
        # Ensure path exists
        if not os.path.exists(path):
            print(f'\033[34mWAS NS\033[0m Error: The path `{path}` doesn\'t exist!')
            
        # Ensure content to save
        if filename.strip == '':
            print(f'\033[34mWAS NS\033[0m Error: There is no text specified to save! Text is empty.')
        text = "Positive Prompt:\n" + positive + "\nNegative Prompt:\n" + negative + "\nSeed: " + str(seed['seed'])
        
        
        filename = self.make_filename(seed, modelname, filename)   
        # Write text file
        self.writeTextFile(os.path.join(path, filename + '.txt'), text)
        
        return( text, )

    def make_filename(self, seed, modelname="", filename=""):
        # generate datetime string
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

        # parse input string
        filename = filename.replace("%time",timestamp)
        filename = filename.replace("%model",modelname)
        filename = filename.replace("%seed",str(seed['seed']))
        

        return (filename)   

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print(f'\033[34mWAS Node Suite\033[0m Error: Unable to save file `{file}`')

class WLSH_Save_Positive_Prompt_File:
    def __init__(s):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "filename": ("STRING",{"default": 'info', "multiline": False}),
                        "path": ("STRING", {"default": './output', "multiline": False}),
                        "positive": ("STRING",{"default": '', "multiline": True}),
                    }
                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"

    CATEGORY = "WLSH Nodes/IO"
    
    def save_text_file(self, positive="", path="./output", filename="info"):
        
        # Ensure path exists
        if not os.path.exists(path):
            print(f'\033[34mWAS NS\033[0m Error: The path `{path}` doesn\'t exist!')
            
        # Ensure content to save
        if filename.strip == '':
            print(f'\033[34mWAS NS\033[0m Error: There is no text specified to save! Text is empty.')
      
        
        # filename = self.make_filename(seed, modelname, filename)   
        # Write text file
        self.writeTextFile(os.path.join(path, filename + '.txt'), positive)
        
        return( positive, )

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print(f'\033[34mWAS Node Suite\033[0m Error: Unable to save file `{file}`')

NODE_CLASS_MAPPINGS = {
    "Checkpoint Loader w/Name (WLSH)": WLSH_Checkpoint_Loader_Model_Name,
    "KSamplerAdvanced (WLSH)": WLSH_KSamplerAdvancedMod,
    # "Alternating KSampler (WLSH)": WLSH_Alternating_KSamplerAdvanced,
    "Seed to Number (WLSH)": WLSH_Seed_to_Number,
    "SDXL Steps (WLSH)": WLSH_SDXL_Steps,
    "SDXL Resolutions (WLSH)": WLSH_SDXL_Resolutions,
    "Multiply Integer (WLSH)": WLSH_Int_Multiply,
    "Time String (WLSH)": WLSH_Time_String,
    "Empty Latent by Ratio (WLSH)" : WLSH_Empty_Latent_Image_By_Ratio,
    "SDXL Quick Empty Latent (WLSH)" : WLSH_SDXL_Quick_Empty_Latent,
    "CLIP Positive-Negative (WLSH)": WLSH_CLIP_Positive_Negative,
    "CLIP Positive-Negative w/Text (WLSH)": WLSH_CLIP_Text_Positive_Negative,
    "Outpaint to Image (WLSH)": WLSH_Outpaint_To_Image,
    "VAE Encode for Inpaint Padding (WLSH)": WLSH_VAE_Encode_For_Inpaint_Padding,
    "Generate Second Mask (WLSH)": WLSH_Generate_Second_Mask,
    # "Generate Face Mask (WLSH)": WLSH_Generate_Face_Mask,
    "Image Scale By Factor (WLSH)": WLSH_Image_Scale_By_Factor,
    "Upscale by Factor with Model (WLSH)": WLSH_Upscale_By_Factor_With_Model,
    "SDXL Quick Image Scale (WLSH)": WLSH_SDXL_Quick_Image_Scale,
    "Image Save with Prompt Data (WLSH)": WLSH_Image_Save_With_Prompt_Info,
    "Save Prompt Info (WLSH)": WLSH_Save_Prompt_File,
    "Image Save with Prompt File (WLSH)": WLSH_Image_Save_With_Prompt_File,
    "Save Positive Prompt File (WLSH)": WLSH_Save_Positive_Prompt_File,
}

