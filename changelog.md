# Changelog

## 2023-08-19

- **Added "Read Prompt Data from Image" node.** Use this to read various metadata from Auto1111 or from images saved with tthe prompt medatada from this node pack.

- **Changed image metadata save format**  Modified the part of the image the metadata is saved to  for png types when using the various image output/prompt output nodes

- **Added INFO as output to custom KSamplerAdvanced**  INFO is a dictionary containing a bunch of the configuration data for that sampler node.

- **Added INFO input to Image/Prompt save nodes** Can be used to append more information to the metadata.
