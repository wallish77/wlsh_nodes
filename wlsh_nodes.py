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
#import cv2
#from deepface import DeepFace

import re
import random
import latent_preview
from datetime import datetime
import json
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
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "modelname")
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
class WLSH_KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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

    RETURN_TYPES = ("LATENT","INFO",)
    FUNCTION = "sample"

    CATEGORY = "WLSH Nodes/sampling"

    def sample(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise):
        force_full_denoise = False
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        info = {"Seed: ": seed, "Steps: ": steps, "CFG scale: ": cfg, "Sampler: ": sampler_name, "Scheduler: ": scheduler, "Start at step: ": start_at_step, "End at step: ": end_at_step, "Denoising strength: ": denoise}    
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
        denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

        return (samples[0], info)


class WLSH_Alternating_KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
        noise_seed = seed
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
                force_full_denoise = True
            if(step > 0):
                # disable_noise=True
                denoise=(steps-step)/(steps)

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

    CATEGORY = "WLSH Nodes/number"

    def number_to_seed(self, seed):
        return (int(seed["seed"]), )

class WLSH_Seed_and_Int:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0,
                          "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("INT","SEED",)
    FUNCTION = "seed_and_int"

    CATEGORY = "WLSH Nodes/number"

    def seed_and_int(self, seed):
        return (seed,{"seed": seed} )

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
    RETURN_NAMES = ("pre", "base", "total")
    FUNCTION = "set_steps"

    CATEGORY="WLSH Nodes/number"


    def set_steps(self,precondition,base,total):
        return(precondition,base,total)

class WLSH_Int_Multiply:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": ("INT",{"default": 2, "min": 1, "max": 10000, "forceInput": True}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 10000}),
            }
        }
    RETURN_TYPES = ("INT",)
    FUNCTION = "multiply"

    CATEGORY="WLSH Nodes/number"


    def multiply(self,number,multiplier):
        result = number*multiplier
        return (int(result),)

class WLSH_Res_Multiply:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "forceInput": True}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "forceInput": True}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 10000}),
            }
        }
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width","height",)
    FUNCTION = "multiply"

    CATEGORY="WLSH Nodes/number"


    def multiply(self,width, height,multiplier):
        adj_width = width*multiplier
        adj_height = height*multiplier
        return (int(adj_width),int(adj_height),)

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
    RETURN_NAMES = ("time_format",)
    FUNCTION = "get_time"

    CATEGORY = "WLSH Nodes/text"

    def get_time(self, style):
        now = datetime.now()
        timestamp = now.strftime(style)

        return (timestamp,)


# Takes an input string and a list string, uses pattern and
# delimiter from inputs to parse the list_string and replace pattern
# in the input_string
class WLSH_Simple_Pattern_Replace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": True, "forceInput": True}),
                "list_string": ("STRING", {"default": f''}),
                "pattern": ("STRING", {"default": f'$var'}),
                "delimiter": ("STRING", {"default": f','}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "replace_string"

    CATEGORY = "WLSH Nodes/text"

    def replace_string(self, input_string, list_string, pattern, delimiter, seed):
        # escape special characters and strip whitespace from pattern
        pattern = re.escape(pattern).strip()

        # find all pattern entries from input and create list
        regex = re.compile(pattern)
        matches = regex.findall(input_string)

        # return input if nothing found
        if not matches:
            return (input_string,)
        
        if seed is not None:
            random.seed(seed)
        
        # if provided delimiter not present in input, will try to use whole list
        # we do not want that to happen...
        if delimiter not in list_string:
            raise ValueError("Delimiter not found in list_string")
        
        # if pattern appears more than once each entry will have a different random choice
        def replace(match):
            return random.choice(list_string.split(delimiter))

        new_string = regex.sub(replace, input_string)

        return (new_string,)


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
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_resolutions"

    CATEGORY="WLSH Nodes/number"


    def get_resolutions(self,resolution, direction):
        width,height = resolution.split('x')
        width = int(width)
        height = int(height)
        if(direction == "portrait"):
            width,height = height,width
        return(width,height)

class WLSH_Resolutions_by_Ratio:
    aspects = ["1:1","5:4","4:3","3:2","16:10","16:9","21:9","2:1","3:1","4:1"]
    direction = ["landscape","portrait"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "aspect": (s.aspects,),
                              "direction": (s.direction,),
                              "shortside": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64})}}
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_resolutions"
    CATEGORY="WLSH Nodes/number"

    def get_resolutions(self, aspect, direction, shortside):
        x,y = aspect.split(':')
        x = int(x)
        y = int(y)
        ratio = x/y
        width = int(shortside * ratio)
        width = (width + 63) & (-64)
        height = shortside
        if(direction == "portrait"):
            width,height = height,width
        return(width,height)

class WLSH_Empty_Latent_Image_By_Resolution:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT","INT","INT",)
    RETURN_NAMES = ("latent", "width", "height",)
    FUNCTION = "generate"

    CATEGORY = "WLSH Nodes/latent"

    def generate(self, width, height, batch_size=1):
        adj_width = width // 8
        adj_height = height // 8
        latent = torch.zeros([batch_size, 4, adj_height, adj_width])
        return ({"samples":latent}, adj_width,adj_height, )

# latent
class WLSH_Empty_Latent_Image_By_Ratio:
    aspects = ["1:1","5:4","4:3","3:2","16:10","16:9","19:9","21:9","2:1","3:1","4:1"]
    direction = ["landscape","portrait"]

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "aspect": (s.aspects,),
                              "direction": (s.direction,),
                              "shortside": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT","INT","INT",)
    RETURN_NAMES = ("latent", "width", "height",)
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
        adj_width = width // 8
        adj_height = height // 8
        latent = torch.zeros([batch_size, 4, adj_height, adj_width])
        return ({"samples":latent}, adj_width,adj_height, )

class WLSH_Empty_Latent_Image_By_Pixels:
    aspects = ["1:1","5:4","4:3","3:2","16:10","16:9","19:9","21:9","2:1","3:1","4:1"]
    direction = ["landscape","portrait"]

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "aspect": (s.aspects,),
                              "direction": (s.direction,),
                              "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT","INT","INT",)
    RETURN_NAMES = ("latent", "width", "height",)
    FUNCTION = "generate"

    CATEGORY = "WLSH Nodes/latent"

    def generate(self, aspect, direction, megapixels, batch_size=1):
        x,y = aspect.split(':')
        x = int(x)
        y = int(y)
        ratio = x/y
        
        total = int(megapixels * 1024 * 1024)

        width = int(np.sqrt(ratio * total))
        width = (width + 63) & (-64)
        height = int(np.sqrt(1/ratio * total))
        height = (height + 63) & (-64)
        if(direction == "portrait"):
            width,height = height,width
        adj_width = width // 8
        adj_height = height // 8
        latent = torch.zeros([batch_size, 4, adj_height, adj_width])
        return ({"samples":latent}, adj_width, adj_height, )


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
    RETURN_TYPES = ("LATENT","INT","INT", )
    RETURN_NAMES = ("latent", "width", "height",)
    FUNCTION = "generate"

    CATEGORY = "WLSH Nodes/latent"

    def generate(self, resolution, direction, batch_size=1):
        width,height = resolution.split('x')
        width = int(width)
        height = int(height)
        if(direction == "portrait"):
            width,height = height,width
        adj_width = width // 8
        adj_height = height // 8
        latent = torch.zeros([batch_size, 4, adj_height, adj_width])
        return ({"samples":latent}, adj_width, adj_height,)

class WLSH_SDXL_Resolution_Multiplier:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "target_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "forceInput": True}),
                              "target_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "forceInput": True}),
                              "multiplier": ("INT", {"default": 2, "min": 1, "max": 12})}}
    RETURN_TYPES = ("INT","INT", )
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "multiply_res"

    CATEGORY = "WLSH Nodes/number"

    def multiply_res(self, target_width=1024, target_height=1024, multiplier=2):
        return (target_width*2, target_height*2,)

# conditioning
class WLSH_CLIP_Text_Positive_Negative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("STRING", {"multiline": True}),
         "negative": ("STRING", {"multiline": True}),
         "clip": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","STRING","STRING")
    RETURN_NAMES = ("positive", "negative","positive_text","negative_text")
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
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode"

    CATEGORY = "WLSH Nodes/conditioning"

    def encode(self, clip, positive_text, negative_text):
        return ([[clip.encode(positive_text), {}]],[[clip.encode(negative_text), {}]] )

class WLSH_CLIP_Text_Positive_Negative_XL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "positive_g": ("STRING", {"multiline": True, "default": "POS_G"}),
            "positive_l": ("STRING", {"multiline": True, "default": "POS_L"}),
            "negative_g": ("STRING", {"multiline": True, "default": "NEG_G"}), 
            "negative_l": ("STRING", {"multiline": True, "default": "NEG_L"}), 
            "clip": ("CLIP", ),
            }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","STRING","STRING")
    RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text")
    FUNCTION = "encode"

    CATEGORY = "WLSH Nodes/conditioning"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, positive_g, positive_l,negative_g, negative_l):
        tokens = clip.tokenize(positive_g)
        tokens["l"] = clip.tokenize(positive_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        condP, pooledP = clip.encode_from_tokens(tokens, return_pooled=True)
        tokensN = clip.tokenize(negative_g)
        tokensN["l"] = clip.tokenize(negative_l)["l"]
        if len(tokensN["l"]) != len(tokensN["g"]):
            empty = clip.tokenize("")
            while len(tokensN["l"]) < len(tokensN["g"]):
                tokensN["l"] += empty["l"]
            while len(tokensN["l"]) > len(tokensN["g"]):
                tokensN["g"] += empty["g"]
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)
        
        #combine pos_l and pos_g prompts
        positive_text = positive_g + ", " + positive_l
        negative_text = negative_g + ", " + negative_l
        return ([[condP, {"pooled_output": pooledP, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]],[[condN, {"pooled_output": pooledP, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]], positive_text, negative_text, )


class WLSH_CLIP_Positive_Negative_XL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "positive_g": ("STRING", {"multiline": True, "default": "POS_G"}),
            "positive_l": ("STRING", {"multiline": True, "default": "POS_L"}), 
            "negative_g": ("STRING", {"multiline": True, "default": "NEG_G"}), 
            "negative_l": ("STRING", {"multiline": True, "default": "NEG_L"}), 
            "clip": ("CLIP", ),
            }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode"

    CATEGORY = "WLSH Nodes/conditioning"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, positive_g, positive_l,negative_g, negative_l):
        tokens = clip.tokenize(positive_g)
        tokens["l"] = clip.tokenize(positive_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        condP, pooledP = clip.encode_from_tokens(tokens, return_pooled=True)
        tokensN = clip.tokenize(negative_g)
        tokensN["l"] = clip.tokenize(negative_l)["l"]
        if len(tokensN["l"]) != len(tokensN["g"]):
            empty = clip.tokenize("")
            while len(tokensN["l"]) < len(tokensN["g"]):
                tokensN["l"] += empty["l"]
            while len(tokensN["l"]) > len(tokensN["g"]):
                tokensN["g"] += empty["g"]
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)
        return ([[condP, {"pooled_output": pooledP, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]],[[condN, {"pooled_output": pooledP, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]], )


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

class WLSH_Generate_Edge_Mask:
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

# class WLSH_Generate_Face_Mask:
#     detectors = ["opencv", "retinaface", "ssd", "mtcnn"]
#     channels = ["red", "blue", "green"]
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "image": ("IMAGE",),
#                              "detector": (s.detectors,),
#                              "channel": (s.channels,),
#                              "mask_padding": ("INT",{"default": 6, "min": 0, "max": 32, "step": 2}) 
#                               }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "gen_face_mask"

#     CATEGORY = "WLSH Nodes/inpainting"
#     def gen_face_mask(self, image, mask_padding, detector, channel):
#         image = tensor2pil(image)
        
#         faces = DeepFace.extract_faces(np.array(image),detector_backend=detector)
#         # cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         # # Convert to grayscale
#         # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

#         # # Detect faces in the image
#         # face_cascade = cv2.CascadeClassifier('custom_nodes/haarcascade_frontalface_default.xml')
#         # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#         mask = Image.new('RGB',image.size)

#         # Draw a rectangle on the PIL Image object
#         colors = {"red": "RGB(255,0,0)", "green": "RGB(0,255,0)", "blue": "RGB(0,0,255)"}
#         draw = ImageDraw.Draw(mask)
#         for face in faces:
#             x,y,w,h = face['facial_area'].values()
#             draw.rectangle((x-mask_padding,y-mask_padding,x+w+mask_padding,y+h+mask_padding), outline=colors[channel], fill=colors[channel])
#         mask = mask.filter(ImageFilter.GaussianBlur(radius=6))

#         mask = np.array(mask).astype(np.float32) / 255.0
#         mask = torch.from_numpy(mask)[None,]
#         return (mask,)

# image I/O
def get_timestamp(time_format="%Y-%m-%d-%H%M%S"):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return(timestamp)

def make_filename(filename="ComfyUI", seed={"seed":0}, modelname="sd", counter=0, time_format="%Y-%m-%d-%H%M%S"):
    '''
    Builds a filename by reading in a filename format and returning a formatted string using input tokens
    Tokens:
    %time - timestamp using the time_format value
    %model - modelname using the modelname input
    %seed - seed from the seed input
    %counter - counter integer from the counter input
    '''
    timestamp = get_timestamp(time_format)

    # parse input string
    filename = filename.replace("%time",timestamp)
    filename = filename.replace("%model",modelname)
    filename = filename.replace("%seed",str(seed))
    filename = filename.replace("%counter",str(counter))  

    if filename == "":
        filename = timestamp
    return(filename)  

def make_comment(positive, negative, modelname="unknown", seed=-1, info=None):
    comment = ""
    if(info is None):
        comment = "Positive prompt:\n" + positive + "\nNegative prompt:\n" + negative + "\nModel: " + modelname + "\nSeed: " + str(seed)
        return comment
    else:
        # reformat to stop long precision
        try:
            info['CFG scale: '] = "{:.2f}".format(info['CFG scale: '])
        except:
            pass
        try:
            info['Denoising strength: '] = "{:.2f}".format(info['Denoising strength: '])
        except:
            pass

        comment = "Positive prompt:\n" + positive + "\nNegative prompt:\n" + negative + "\nModel: " + modelname
        for key in info:
            newline = "\n" + key + str(info[key])
            comment += newline
    # print(comment)
    return comment

# version without INFO input for TTN compatability
class WLSH_Image_Save_With_Prompt:
    def __init__(self):
        # get default output directory
        self.type = "output"
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "images": ("IMAGE", ),
                        "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "extension": (['png', 'jpeg', 'tiff', 'gif'], ),
                        "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                    },
                    "optional": {
                        "positive": ("STRING",{ "multiline": True, "forceInput": True}, ),
                        "negative": ("STRING",{"multiline": True, "forceInput": True}, ),
                        "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                        "modelname": ("STRING",{"default": '', "multiline": False, "forceInput": True}),
                        "counter": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                    },
                    "hidden": {
                        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "WLSH Nodes/IO"

    def save_files(self, images, positive="unknown", negative="unknown", seed=-1, modelname="unknown", counter=0, filename='', path="",
    time_format="%Y-%m-%d-%H%M%S",  extension='png', quality=100, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed, modelname, counter, time_format)
        comment = make_comment(positive, negative, modelname, seed, info=None)
        # comment = "Positive Prompt:\n" + positive + "\nNegative Prompt:\n" + negative + "\nModel: " + modelname + "\nSeed: " + str(seed)

        output_path = os.path.join(self.output_dir,path)
        
        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    
                
        paths = self.save_images(images, output_path,path, filename,comment, extension, quality, prompt, extra_pnginfo)
        
        return { "ui": { "images": paths } }

    def save_images(self, images, output_path, path, filename_prefix="ComfyUI", comment="", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
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
            metadata.add_text("parameters", comment)
            metadata.add_text("comment", comment)
            if(images.size()[0] > 1):
                filename_prefix += "_{:02d}".format(imgCount)

            file = f"{filename_prefix}.{extension}"
            if extension == 'png':
                # print(comment)
                img.save(os.path.join(output_path, file), comment=comment, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(os.path.join(output_path, file), quality=quality)
            elif extension == 'jpeg':
                img.save(os.path.join(output_path, file), quality=quality, comment=comment, optimize=True)
            elif extension == 'tiff':
                img.save(os.path.join(output_path, file), quality=quality, optimize=True)
            else:
                img.save(os.path.join(output_path, file))
            paths.append({
                "filename": file,
                "subfolder": path,
                "type": self.type
            })
            imgCount += 1
        return(paths)

class WLSH_Image_Save_With_Prompt_Info:
    def __init__(self):
        # get default output directory
        self.type = "output"
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "images": ("IMAGE", ),
                        "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "extension": (['png', 'jpeg', 'tiff', 'gif'], ),
                        "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                    },
                    "optional": {
                        "positive": ("STRING",{ "multiline": True, "forceInput": True}, ),
                        "negative": ("STRING",{"multiline": True, "forceInput": True}, ),
                        "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                        "modelname": ("STRING",{"default": '', "multiline": False, "forceInput": True}),
                        "counter": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                        "info": ("INFO",)
                    },
                    "hidden": {
                        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "WLSH Nodes/IO"

    def save_files(self, images, positive="unknown", negative="unknown", seed=-1, modelname="unknown", info=None, counter=0, filename='', path="",
    time_format="%Y-%m-%d-%H%M%S",  extension='png', quality=100, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed, modelname, counter, time_format)
        comment = make_comment(positive, negative, modelname, seed, info)
        # comment = "Positive Prompt:\n" + positive + "\nNegative Prompt:\n" + negative + "\nModel: " + modelname + "\nSeed: " + str(seed)

        output_path = os.path.join(self.output_dir,path)
        
        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    
                
        paths = self.save_images(images, output_path,path, filename,comment, extension, quality, prompt, extra_pnginfo)
        
        return { "ui": { "images": paths } }

    def save_images(self, images, output_path, path, filename_prefix="ComfyUI", comment="", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
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
            metadata.add_text("parameters", comment)
            metadata.add_text("comment", comment)
            if(images.size()[0] > 1):
                filename_prefix += "_{:02d}".format(imgCount)

            file = f"{filename_prefix}.{extension}"
            if extension == 'png':
                # print(comment)
                img.save(os.path.join(output_path, file), comment=comment, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(os.path.join(output_path, file), quality=quality)
            elif extension == 'jpeg':
                img.save(os.path.join(output_path, file), quality=quality, comment=comment, optimize=True)
            elif extension == 'tiff':
                img.save(os.path.join(output_path, file), quality=quality, optimize=True)
            else:
                img.save(os.path.join(output_path, file))
            paths.append({
                "filename": file,
                "subfolder": path,
                "type": self.type
            })
            imgCount += 1
        return(paths)        

class WLSH_Image_Save_With_File_Info:
    def __init__(self):
        # get default output directory
        self.output_dir = folder_paths.output_directory
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "images": ("IMAGE", ),                   
                        "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "extension": (['png', 'jpeg', 'tiff', 'gif'], ),
                        "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                    },
                    "optional": {
                        "positive": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                        "negative": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                        "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                        "modelname": ("STRING",{"default": '', "multiline": False, "forceInput": True}),
                        "counter": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                        "info": ("INFO",)
                    },
                    "hidden": {
                        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "WLSH Nodes/IO"

    def save_files(self, images, positive="unknown", negative="unknown", seed=-1, modelname="unknown", info=None, counter=0, filename='', path="",
    time_format="%Y-%m-%d-%H%M%S", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed, modelname, counter, time_format)
        comment = make_comment(positive, negative, modelname, seed, info)
        output_path = os.path.join(self.output_dir,path)
        
        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    
                
        paths = self.save_images(images, output_path,filename,comment, extension, quality, prompt, extra_pnginfo)
        self.save_text_file(filename,path, output_path, comment, seed, modelname)
        #return
        return { "ui": { "images": paths } }

    def save_images(self, images, output_path, path, filename_prefix="ComfyUI", comment="", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
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
            metadata.add_text("parameters", comment)
            if(images.size()[0] > 1):
                filename_prefix += "_{:02d}".format(imgCount)

            file = f"{filename_prefix}.{extension}"
            if extension == 'png':
                # print(comment)
                img.save(os.path.join(output_path, file), comment=comment, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(os.path.join(output_path, file), quality=quality)
            elif extension == 'jpeg':
                img.save(os.path.join(output_path, file), quality=quality, comment=comment, optimize=True)
            elif extension == 'tiff':
                img.save(os.path.join(output_path, file), quality=quality, optimize=True)
            else:
                img.save(os.path.join(output_path, file))
            paths.append({
                "filename": file,
                "subfolder": path,
                "type": self.type
            })
            imgCount += 1
        return(paths)

    def save_text_file(self, filename, output_path, comment="", seed=0, modelname=""):
        # Write text file
        self.writeTextFile(os.path.join(output_path, filename + '.txt'), comment)
        
        return

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print('Unable to save file `{file}`')

class WLSH_Image_Save_With_File:
    def __init__(self):
        # get default output directory
        self.output_dir = folder_paths.output_directory
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "images": ("IMAGE", ),                   
                        "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "extension": (['png', 'jpeg', 'tiff', 'gif'], ),
                        "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                    },
                    "optional": {
                        "positive": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                        "negative": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                        "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                        "modelname": ("STRING",{"default": '', "multiline": False, "forceInput": True}),
                        "counter": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                    },
                    "hidden": {
                        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                    },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "WLSH Nodes/IO"

    def save_files(self, images, positive="unknown", negative="unknown", seed=-1, modelname="unknown", counter=0, filename='', path="",
    time_format="%Y-%m-%d-%H%M%S", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed, modelname, counter, time_format)
        comment = make_comment(positive, negative, modelname, seed, info=None)
        output_path = os.path.join(self.output_dir,path)
        
        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    
                
        paths = self.save_images(images, output_path,filename,comment, extension, quality, prompt, extra_pnginfo)
        self.save_text_file(filename,path, output_path, comment, seed, modelname)
        #return
        return { "ui": { "images": paths } }

    def save_images(self, images, output_path, path, filename_prefix="ComfyUI", comment="", extension='png', quality=100, prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(filename_prefix)
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
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
            metadata.add_text("parameters", comment)
            if(images.size()[0] > 1):
                filename_prefix += "_{:02d}".format(imgCount)

            file = f"{filename_prefix}.{extension}"
            if extension == 'png':
                # print(comment)
                img.save(os.path.join(output_path, file), comment=comment, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(os.path.join(output_path, file), quality=quality)
            elif extension == 'jpeg':
                img.save(os.path.join(output_path, file), quality=quality, comment=comment, optimize=True)
            elif extension == 'tiff':
                img.save(os.path.join(output_path, file), quality=quality, optimize=True)
            else:
                img.save(os.path.join(output_path, file))
            paths.append({
                "filename": file,
                "subfolder": path,
                "type": self.type
            })
            imgCount += 1
        return(paths)

    def save_text_file(self, filename, output_path, comment="", seed=0, modelname=""):
        # Write text file
        self.writeTextFile(os.path.join(output_path, filename + '.txt'), comment)
        
        return

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print('Unable to save file `{file}`')                    

class WLSH_Save_Prompt_File:
    def __init__(self):
        # get default output directory
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "filename": ("STRING",{"default": 'info', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "positive": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                    },
                    "optional": {
                        "negative": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                        "modelname": ("STRING",{"default": '', "multiline": False, "forceInput": True}),
                        "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                        "counter": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                    }
                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"

    CATEGORY = "WLSH Nodes/IO"
    
    def save_text_file(self, positive="", negative="", seed=-1, modelname="unknown", path="", counter=0, time_format="%Y-%m-%d-%H%M%S", filename=""):
        
        output_path = os.path.join(self.output_dir,path)

        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    

        text_data = make_comment(positive, negative, modelname, seed, info=None)
        
        
        filename = make_filename(filename, seed, modelname, counter, time_format)
        # Write text file
        self.writeTextFile(os.path.join(output_path, filename + '.txt'), text_data)
        
        return( text_data, )

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print(f'Error: Unable to save file `{file}`')

class WLSH_Save_Prompt_File_Info:
    def __init__(self):
        # get default output directory
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "filename": ("STRING",{"default": 'info', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "positive": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                    },
                    "optional": {
                        "negative": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                        "modelname": ("STRING",{"default": '', "multiline": False, "forceInput": True}),
                        "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                        "counter": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                        "info": ("INFO",)
                    }
                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"

    CATEGORY = "WLSH Nodes/IO"
    
    def save_text_file(self, positive="", negative="", seed=-1, modelname="unknown", info=None, path="", counter=0, time_format="%Y-%m-%d-%H%M%S", filename=""):
        
        output_path = os.path.join(self.output_dir,path)

        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    

        text_data = make_comment(positive, negative, modelname, seed, info)
        
        
        filename = make_filename(filename, seed, modelname, counter, time_format)
        # Write text file
        self.writeTextFile(os.path.join(output_path, filename + '.txt'), text_data)
        
        return( text_data, )

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print(f'Error: Unable to save file `{file}`')

class WLSH_Save_Positive_Prompt_File:
    def __init__(self):
        # get default output directory
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "filename": ("STRING",{"default": 'info', "multiline": False}),
                        "path": ("STRING", {"default": '', "multiline": False}),
                        "positive": ("STRING",{"default": '', "multiline": True, "forceInput": True}),
                    }
                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"

    CATEGORY = "WLSH Nodes/IO"
    
    def save_text_file(self, positive="", path="", filename=""):
        
        output_path = os.path.join(self.output_dir,path)

        # create missing paths - from WAS Node Suite
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    

        # Ensure content to save, use timestamp if no name given
        if filename.strip == '':
            print(f'Warning: There is no text specified to save! Text is empty.  Saving file with timestamp')
            filename = get_timestamp('%Y%m%d%H%M%S')
        
        # Write text file after checking for empty prompt
        if positive == "":
            positive ="No prompt data"
        
        self.writeTextFile(os.path.join(output_path, filename + '.txt'), positive)
        
        return( positive, )

    # Save Text FileNotFoundError
    def writeTextFile(self, file, content):
        try:
            with open(file, 'w') as f:
                f.write(content)
        except OSError:
            print(f'Error: Unable to save file `{file}`')
# images

class WLSH_Image_Grayscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "original": ("IMAGE",), }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grayscale",)
    FUNCTION = "make_grayscale"

    CATEGORY = "WLSH Nodes/image"

    def make_grayscale(self, original):
        image = tensor2pil(original)
        image = ImageOps.grayscale(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)


class WLSH_Read_Prompt:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                        "verbose": (["true", "false"],),
                        "image": (sorted(files),  {"image_upload": True}),   
                    },
                }
    CATEGORY = "WLSH Nodes/image"
    ''' Return order:
        positive prompt(string), negative prompt(string), seed(int), steps(int), cfg(float), 
        width(int), height(int)
    '''
    RETURN_TYPES = ("IMAGE", "STRING","STRING","INT", "INT", "FLOAT", "INT","INT")
    RETURN_NAMES = ("image", "positive", "negative", "seed", "steps", "cfg", "width", "height")
    FUNCTION = "get_image_data"
    
    def get_image_data(self, image, verbose):
        image_path = folder_paths.get_annotated_filepath(image)
        with open(image_path,'rb') as file:
            img = Image.open(file)
            extension = image_path.split('.')[-1]
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

        parameters = ""
        comfy = False
        if extension.lower() == 'png':
            try:
                parameters = img.info['parameters']
                if not parameters.startswith("Positive prompt"):
                    parameters = "Positive prompt: " + parameters
            except:
                parameters = ""
                print("Error loading prompt info from png")
                # return "Error loading prompt info."
        elif extension.lower() in ("jpg", "jpeg", "webp"):
            try:
                exif = piexif.load(img.info["exif"])
                parameters = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
                parameters = piexif.helper.UserComment.load(parameters)
                if not parameters.startswith("Positive prompt"):
                    parameters = "Positive prompt: " + parameters
            except:
                try:
                    parameters = str(img.info['comment'])
                    comfy = True
                    # legacy fixes
                    parameters = parameters.replace("Positive Prompt", "Positive prompt")
                    parameters = parameters.replace("Negative Prompt", "Negative prompt")
                    parameters = parameters.replace("Start at Step", "Start at step")
                    parameters = parameters.replace("End at Step", "End at step")
                    parameters = parameters.replace("Denoising Strength", "Denoising strength")
                except:
                    parameters = ""
                    print("Error loading prompt info from jpeg")
                    # return "Error loading prompt info."

        if(comfy and extension.lower() == 'jpeg'):
            parameters = parameters.replace('\\n',' ')
        else:
            parameters = parameters.replace('\n',' ')


        patterns = [
            "Positive prompt: ",
            "Negative prompt: ",
            "Steps: ",
            "Start at step: ",
            "End at step: ",
            "Sampler: ",
            "Scheduler: ",
            "CFG scale: ",
            "Seed: ",
            "Size: ",
            "Model: ",
            "Model hash: ",
            "Denoising strength: ",
            "Version: ",
            "ControlNet 0",
            "Controlnet 1",
            "Batch size: ",
            "Batch pos: ",
            "Hires upscale: ",
            "Hires steps: ",
            "Hires upscaler: ",
            "Template: ",
            "Negative Template: ",
        ]
        if(comfy and extension.lower() == 'jpeg'):
            parameters = parameters[2:]
            parameters = parameters[:-1]

        keys = re.findall("|".join(patterns), parameters)
        values = re.split("|".join(patterns), parameters)
        values = [x for x in values if x]
        results = {}
        result_string = ""
        for item in range(len(keys)):
            result_string += keys[item] + values[item].rstrip(', ')
            result_string += "\n"
            results[keys[item].replace(": ","")] = values[item].rstrip(', ')
            
        if(verbose == "true"):
            print(result_string)

        try:
            positive = results['Positive prompt']
        except:
            positive = ""
        try:
            negative = results['Negative prompt']
        except:
            negative = ""
        try:
            seed = int(results['Seed'])
        except:
            seed = -1
        try:
            steps = int(results['Steps'])
        except:
            steps = 20
        try:
            cfg = float(results['CFG scale'])
        except:
            cfg = 8.0
        try:
            width,height = img.size
        except:
            width,height = 512,512
        
        ''' Return order:
            positive prompt(string), negative prompt(string), seed(int), steps(int), cfg(float), 
            width(int), height(int)
        '''

        return(image, positive, negative, seed, steps, cfg, width, height)
    
    @classmethod
    def IS_CHANGED(s, image, verbose):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, verbose):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class WLSH_Build_Filename_String:
    def __init__(s):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "filename": ("STRING",{"%time_%seed": 'info', "multiline": False}),
                    },
                    "optional": {
                        "modelname": ("STRING",{"default": '', "multiline": False}),
                        "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                        "counter": ("SEED",{"default": 0}),
                        "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                    }
                }
                
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "build_filename"

    CATEGORY = "WLSH Nodes/text"

    def build_filename(self, filename="ComfyUI", modelname="model", time_format="%Y-%m-%d-%H%M%S", seed=0, counter=0):

        filename = make_filename(filename,seed,modelname,counter,time_format)
        return(filename)

NODE_CLASS_MAPPINGS = {
    #loaders
    "Checkpoint Loader w/Name (WLSH)": WLSH_Checkpoint_Loader_Model_Name,
    #samplers
    "KSamplerAdvanced (WLSH)": WLSH_KSamplerAdvanced,
    # "Alternating KSampler (WLSH)": WLSH_Alternating_KSamplerAdvanced,
    #conditioning
    "CLIP Positive-Negative (WLSH)": WLSH_CLIP_Positive_Negative,
    "CLIP Positive-Negative w/Text (WLSH)": WLSH_CLIP_Text_Positive_Negative,
    "CLIP Positive-Negative XL (WLSH)": WLSH_CLIP_Positive_Negative_XL,
    "CLIP Positive-Negative XL w/Text (WLSH)": WLSH_CLIP_Text_Positive_Negative_XL,
    #latent
    "Empty Latent by Pixels (WLSH)": WLSH_Empty_Latent_Image_By_Pixels,
    "Empty Latent by Ratio (WLSH)" : WLSH_Empty_Latent_Image_By_Ratio,
    "Empty Latent by Size (WLSH)": WLSH_Empty_Latent_Image_By_Resolution,
    "SDXL Quick Empty Latent (WLSH)" : WLSH_SDXL_Quick_Empty_Latent,
    #image
    "Image Load with Metadata (WLSH)": WLSH_Read_Prompt,
    "Grayscale Image (WLSH)": WLSH_Image_Grayscale,
    #inpainting
    "Generate Border Mask (WLSH)": WLSH_Generate_Edge_Mask,
    "Outpaint to Image (WLSH)": WLSH_Outpaint_To_Image,
    "VAE Encode for Inpaint w/Padding (WLSH)": WLSH_VAE_Encode_For_Inpaint_Padding,
    #upscaling
    "Image Scale By Factor (WLSH)": WLSH_Image_Scale_By_Factor,
    "SDXL Quick Image Scale (WLSH)": WLSH_SDXL_Quick_Image_Scale,
    "Upscale by Factor with Model (WLSH)": WLSH_Upscale_By_Factor_With_Model,
    #numbers
    "Multiply Integer (WLSH)": WLSH_Int_Multiply,
    "Quick Resolution Multiply (WLSH)": WLSH_Res_Multiply,
    "Resolutions by Ratio (WLSH)": WLSH_Resolutions_by_Ratio,
    "Seed to Number (WLSH)": WLSH_Seed_to_Number,
    "Seed and Int (WLSH)": WLSH_Seed_and_Int,
    "SDXL Steps (WLSH)": WLSH_SDXL_Steps,
    "SDXL Resolutions (WLSH)": WLSH_SDXL_Resolutions,
    #text
    "Build Filename String (WLSH)": WLSH_Build_Filename_String,
    "Time String (WLSH)": WLSH_Time_String,
    "Simple Pattern Replace (WLSH)": WLSH_Simple_Pattern_Replace,
    #IO
    "Image Save with Prompt (WLSH)": WLSH_Image_Save_With_Prompt,
    "Image Save with Prompt/Info (WLSH)": WLSH_Image_Save_With_Prompt_Info,
    "Image Save with Prompt File (WLSH)": WLSH_Image_Save_With_File,
    "Image Save with Prompt/Info File (WLSH)": WLSH_Image_Save_With_File_Info,
    "Save Prompt (WLSH)": WLSH_Save_Prompt_File,
    "Save Prompt/Info (WLSH)": WLSH_Save_Prompt_File_Info,
    "Save Positive Prompt(WLSH)": WLSH_Save_Positive_Prompt_File
}