# SailFish

A software to perform image processing operations on large 3D datasets of CT-scans.

image operations can be performed using the *compute_from_json.py* script. 
For large images the dataset can be split into chunks along its X-axis which will be computed one at a time. 
The chunks will be saved to temporary files which will automatically reassembled and deleted, unless otherwise specified.
if the chunks are to be further processed individually or for some other reason not reassembled immediately the can be assembled into a full image with the *reassemble_chunks.py* script.

the *load_image.py* script can be used to visualize datasets or images computed with this software

