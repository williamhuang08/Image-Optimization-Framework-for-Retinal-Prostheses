# Image-Optimization-Framework-for-Retinal-Prostheses

# Abstract
Retinal prostheses are currently limited to low-resolution grayscale images that lack color and spatial information. This study develops a novel real-time image optimization framework and tools to encode maximum information to the prostheses which are constrained by the number of electrodes. One key idea is to localize main objects in images while reducing unnecessary background noise through region-contrast saliency maps. A novel color depth mapping technique was developed through MiniBatchKmeans clustering and color space selection. The resulting image was downsampled using bicubic inter-polation to reduce image size while preserving color quality. In comparison to current schemes, the proposed framework demonstrated better visual quality in tested images. The use of the region-contrast saliency map showed improvements in efficacy up to 30 %. Finally, the computational speed of this algorithm is less than 380 ms on tested cases, making real-time retinal prostheses feasible.

# Using this Repo
For additional guidance, please contact the author.
<ul>
  <li> As the first step, find a dataset/datasets of images with varying objects in clear contrast to their background. Alternatively, if you have a webcam, you may connect it to your device to create your own dataset.
  <li> Next, use the script entitled <b>videocam.py</b> to create optimized images for retinal prostheses patients (100 x 100 pixels). The code will automatically output five different images for each input image in your dataset (detailed below). If you chose to use existing datasets, the model will output a processed version of all images in the dataset. If you chose to capture images, you must click on the space bar to capture an image. Once completed, press esc.
    <ul>
      <li> 1. !([Cam Images/VPU_frame_0.jpeg](https://github.com/williamhuang08/Image-Optimization-Framework-for-Retinal-Prostheses/blob/dbbe0c9e8bf3a7ea2d0e679657e5c3a53e72bae8/Cam%20Images/VPU_frame_0.jpeg))
    <ul>
<ul>



# Paper
Check out my paper published in IEEE Xplore titled "Enhancing the Bionic Eye: A Real-Time Image Optimization Framework to Encode Color and Spatial Information into Retinal Prostheses" (https://ieeexplore.ieee.org/document/9701618)
