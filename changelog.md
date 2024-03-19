# Changelog

## 2024-03-19

- Fixed imports based on changes in comfy

## 2023-12-31

- Added additional resolution to ratio-based nodes.

## 2023-11-24

- Added width and height to outputs of Image Scale by Shortside

## 2023-11-22

- Fixed issue with Image Save with Prompt/Info & File node

## 2023-11-21

- Fixed issue with integer width/height return from quick latent nodes.

## 2023-11-20

- Added Image Scale by Shortside node

## 2023-10-30

- Added unified SD1.5/SDXL CLIP-Text node

## 2023-10-25

- Added Prompt Weight node
- Added SDXL workflow

## 2023-10-18

- Added string combine node


## 2023-10-14

- Added image preprocessing node to make images grayscale

- Fixed "Image Upscale by Factor" node logic

## 2023-09-30

- Added non-INFO variant of file IO nodes to quick-fix a compatibility issue that seems to crop up with the TTN suite

- Renamed "Read Image with Prompt" to "Image Load with Metadata" and moved it to the 'image' category

- Re-ordered node loading to make categories of WLSH submenu be in a more reasonable order

- Redo of README for readability

- Updated workflow png

## 2023-09-25

- Added "Empty Latent by Size" node to set latent size directly but return width and height

- Added "Quick Resolution Multiplier" node to quickly multiply an input height and width (for SDXL)

## 2023-09-20

- Added 19:9 aspect ratio to empty latent by ratio node (and others)

- Added width/height output to empty latents (so you don't need a separate node)

- Added "Empty Latent by Pixels" which lets you set aspect ratio and megapixels total

- Moved the resolution number nodes to a numbers category


## 2023-09-01

- Added 'Simple Pattern Replace' node

## 2023-08-28
Changed input types for several nodes to default to 'input' instead of 'widget'.  For example, the file save nodes won't have prompt text boxes.

## 2023-08-22

- **Updated Read Prompt Data From Image node.** Using comfyUI commit d7b3b0f8c11c6261d0d8b859ea98f2d818b7e67d, allows for preview of selected image.

- **SDXL version of CLIP Positive-Negative nodes**.

- **Image save nodes preview**. Now shows previews of images saved in image output nodes

## 2023-08-19

- **Added "Read Prompt Data from Image" node.** Use this to read various metadata from Auto1111 or from images saved with tthe prompt medatada from this node pack.

- **Changed image metadata save format**  Modified the part of the image the metadata is saved to  for png types when using the various image output/prompt output nodes

- **Added INFO as output to custom KSamplerAdvanced**  INFO is a dictionary containing a bunch of the configuration data for that sampler node.

- **Added INFO input to Image/Prompt save nodes** Can be used to append more information to the metadata.

- **Converted SEED input types to INT** Fixed need for `SEED` type inputs on sampler and file saves.  May require re-routing old workflows.
