# Changelog

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
