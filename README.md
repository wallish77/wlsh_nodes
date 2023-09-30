# WLSH ComfyUI Nodes
A set of custom nodes for ComfyUI created for personal use to solve minor annoyances or implement various features.  Includes nodes to read or write metadata to saved images in a similar way to Automatic1111 and nodes to quickly generate latent images at resolutions by pixel count and aspect ratio.

## Node descriptions

## Loaders

| Node Name | Description |
|-----------|-------------|
| Checkpoint Loader with Name | The regular checkpoint loader with an output that provides the name of the loaded model as a string for use in saving filenames

## Samplers

| Node Name | Description |
|-----------|-------------|
|KSamplerAdvanced (WLSH) | Modified version of the advanced KSampler to return the `denoising` option for experimentation.  Also provides the INFO output that provides sampler settings for use in other nodes.

## Conditioning

| Node Name | Description |
|-----------|-------------|
|CLIP Positive-Negative| Simple CLIP box containing positive and negative prompt boxes.  Mostly a space saver.  Also available as an SDXL version.|
|CLIP Postive-Negative w/Text| Same as the above, but with two output ndoes to provide the positive and negative inputs to other nodes.  Also available as an SDXL version|

## Latent

| Node Name | Description |
|-----------|-------------|
|Empty Latent by Pixels|Quickly generate an empty latent by selecting an aspect ratio, an orientation (landscape or portrait) and the number of megapixels (1MP = 1024 * 1024 in this case).  Also provides the height and width as integers for plugging in elsewhere. |
|Empty Latent by Ratio|Quickly generate an empty latent by selecting an aspect ratio, an orientation (landscape or portrait) and the length of the 'shorter' side.  Also provides the height and width as integers for plugging in elsewhere. |
|Empty Latent by Size |Nearly identical to the default empty latent node, but also provides the width and height as integer outputs for plugging in elsewhere. |
|SDXL Quick Empty Latent| Quickly generate an empty latent of a size from a list of ideal SDXL image sizes.  Also provides the height and width as integers for plugging in elsewhere.|


## Image

| Node Name | Description |
|-----------|-------------|
|Image Load with Metadata| Extract image metadata from Auto1111-sourced images or from images saved with the nodes in this pack.  Can then connect these outputs to other nodes.   Requires comfyUI commit d7b3b0f8c11c6261d0d8b859ea98f2d818b7e67d to show preview of selected image|

## Inpainting

| Node Name | Description |
|-----------|-------------|
|Generate Border Mask| Used to generate a 'stripe' of mask with blurred edges that overlaps an original iamge with the outpainted new space.  Use on a second pass of an image to 'redo' the masked stripe region and help smooth out the edges of an outpainted image. Pass output to a `Convert Image to Mask` node using the `green` channel.|
|Outpaint to Image|Extends an image in a selected direction by a number of pixels and outputs the expanded image and a mask of the outpainted region with some blurred border padding.|
|VAE Encode for Inpaint Padding|A combined node that takes an image and mask and encodes for the sampler.  Can apply some padding/blur to improve inpainting behavior.|

## Upscaling

| Node Name | Description |
|-----------|-------------|
|Image Scale by Factor|Scale an image by a predefined factor without a model|
|SDXL Quick Image Scale|Take an input image and do a quick simple scale (or scale & crop) to one of the ideal SDXL resolutions|
|Upscale by Factor with Model|Does what it says on the tin.  Scales using an upscale model, but lets you define the multiplier factor rather than take from the model.|

## Numbers

| Node Name | Description |
|-----------|-------------|
|Mulitply Integer | Simple math node to multiple an integer by an integer|
|Quick Resolution Multiplier|Takes in an integer width and height and returns width and height times the multiplier.  Useful for SDXL height (multiplied) vs. target_height (actual resolution)|
|Resolutions by Ratio|Similar to Empty Latent by Ratio, but returns integer `width` and `height` for use with other nodes.  Possibly deprecated now that the latent equivalent node returns width and height as well|
|Seed to Number| Converts a `SEED` type to an `INT`, which was needed for some third party nodes|
|Seed and Int | Generates a seed and outputs as an `INT` and `SEED` type to allow the number to be used with nodes looking for varying inputs|
|SDXL Steps|A single node that outputs three different integers for use as step count in SDXL workflows.|
|SDXL Resolutions| Outputs height and width integers from a list of SDXL optimal resolutions|

## Text

| Node Name | Description |
|-----------|-------------|
|Build Filename String|Takes inputs similar to the Image Save node, used to build a filename which can be passed to multiple other nodes|
|Time String|Returns a formatted string of the current time in the selected format.  Useful for filenames or debugging|
|Simple Pattern Replace|Takes in an input string and a list string and does a simple find/replace and returns a new string.  Can choose a custom delimiter for the list string in case you want to use commas in the list.  If multiple entries of the pattern are in the input, each will get their own random entry.|


## IO
Various nodes for file saving.  Most of these come with an "info" variant that reads in the `INFO` output from the `KSamplerAdvanced (WLSH)` node, which contains additional metadata such as sampler.  The nodes are separate due to compatibility issues found with some other third party node suites such as TTN.

| Node Name | Description |
|-----------|-------------|
|Image Save with Prompt | Save an image with prompt information embedded as a comment or exif data similar to how Auto1111 does it.  Need to feed it various pieces of information and then specify the filename syntax for replacement if desired.  Filename options include `%time` for timestamp, `%model` for model name (via input node or text box), `%seed` for the seed (via input node), and `%counter` for the integer counter (via primitive node with 'increment' option ideally). As far as I can tell, does not remove the ComfyUI 'embed workflow' feature for PNG.|
|Image Save with Prompt File| Save an image with embedded metadata along with a text file containing the metadata. Text file will have a matching filename|
|Save Prompt| Saves the prompts to a text file, along with info on model name and seed|
|Save Positive Prompt| Same as the previous one, but only the positive prompt is saved|


## Disabled or Non-Functional Nodes
| Node Name | Description |
|-----------|-------------|
|Alternating KSampler| Work in progress.  Attempting to recreate the alternating prompts of Auto1111.  Provide a prompt with the syntax "<foo\|bar>" for it to alternate between "foo" and "bar" (and any number of others) before looping back around to the first entry.  Currently not functional due to sampling issues, but the prompt alternating logic works.|

## Example Workflow
Here's an example hi-res workflow using various WLSH nodes.

![hi-res](./img/hires-workflow-example.png)
