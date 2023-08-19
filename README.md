# WLSH ComfyUI Nodes
A set of custom nodes for ComfyUI created for personal use to solve minor annoyances or implement various features.

## Node descriptions
### Checkpoint Loader with Name
The regular checkpoint loader with an output that provides the name of the loaded model as a string for use in saving filenames

### KSamplerAdvanced (WLSH)
Modified version of the advanced KSampler to return the `denoising` option for experimentation.  Also provides the INFO output that provides sampler settings for use in other nodes.

### Seed to Number
Converts a `SEED` type to an `INT`, which was needed for some third party nodes

### Seed and Int
Generates a seed and outputs as an `INT` and `SEED` type to allow the number to be used with nodes looking for varying inputs

### SDXL Steps
A single node that outputs three different integers for use as step count in SDXL workflows.

### SDX Resolutions
Outputs height and width integers from a list of SDXL optimal resoultions

### Multiple Integer
Simple math node to multiple an integer by an integer.

### Time String
Returns a formatted string of the current time in the selected format.  Useful for filenames or debugging

### Empty Latent by Ratio
Quickly generate an empty latent by selecting an aspect ratio, an orientation (landscape or portrait) and the length of the 'shorter' side

### Resolutions by Ratio
Similar to Empty Latent by Ratio, but returns integer `width` and `height` for use with other nodes.

### SDXL Quick Empty Latent
Quickly generate an empty latent of a size from a list of ideal SDXL image sizes

### CLIP Positive-Negative
Simple CLIP box containing positive and negative prompt boxes.  Mostly a space saver

### Clip Postive-Negative w/Text
Same as the above, but with two output ndoes to provide the positive and negative inputs to other nodes

### Outpaint to Image
Extends an image in a selected direction by a number of pixels and outputs the expanded image and a mask of the outpainted region with some blurred border padding.

### Generate Edge Mask
Used to generate a 'stripe' of mask with blurred edges that overlaps an original iamge with the outpainted new space.  Use on a second pass of an image to 'redo' the masked stripe region and help smooth out the edges of an outpainted image. Pass output to a `Convert Image to Mask` node using the `green` channel.

### VAE Encode for Inpaint Padding
A combined node that takes an image and mask and encodes for the sampler.  Can apply some padding/blur to improve inpainting behavior.

### Image Scale by Factor
Scale an image by a predefined factor without a model

### Upscale by Factor with Model
Does what it says on the tin.  Scales using an upscale model, but lets you define the multiplier factor rather than take from the model.

### SDXL Quick Image Scale
Take an input image and do a quick simple scale (or scale & crop) to one of the ideal SDXL resolutions

### Image Save with Prompt Data
Save an image with prompt information embedded as a comment or exif data similar to how Auto1111 does it.  Need to feed it various pieces of information and then specify the filename syntax for replacement if desired.  Filename ptions include `%time` for timestamp, `%model` for model name (via input node or text box), `%seed` for the seed (via input node), and `%counter` for the integer counter (via primative node with 'increment' option ideally).  Writes the above as well as positive and negative prompts (**convert their widgets to inputs to connect to a prompt string box for easy automation**).  As far as I can tell, does not remove the ComfyUI 'embed workflow' feature for PNG.

Many options are optional for saving.

### Save Prompt Info
As the above, but to a txt file

### Image Save with Prompt File
Combination of the above two

### Save Positive Prompt File
Saves just the connected text info (expected to be the positive prompt) to help with training.  Make sure to do matching filenames!

### Read Prompt Data from Image
Extra image metadata from Auto1111-sourced images or from images saved with the nodes in this pack.  Can then connect these outputs to other nodes.

### Build Filename String
Takes inputs similar to the Image Save node, used to build a filename which can be passed to multiple other nodes

## Disabled or Non-Functional Nodes

### Alternating KSampler
Work in progress.  Attempting to recreate the alternating prompts of Auto1111.  Provide a prompt with the syntax "<foo|bar>" for it to alternate between "foo" and "bar" (and any number of others) before looping back around to the first entry.  Currently not functional due to sampling issues, but the prompt alternating logic works.

### Generate Face Mask
Use a face detection library to create a mask box on a desired color channel to then be passed to the `Convert Image to Mask` node, but currently not working.


## Example Workflow
Here's an example hi-res workflow using various WLSH nodes.

![hi-res](./img/hires-workflow-example.png)
