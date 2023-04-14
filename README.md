# AI Image Upscaling
 
This project aims to enhance the performance of a compression pipeline used in image processing. In the first phase, a simple compression pipeline was constructed using conventional techniques, while in the second phase, the pipeline is improved by implementing machine learning methods. 

Bilinear or bicubic upsampling methods were used to recover downsampled images in the initial pipeline, but they cannot restore fine details and edges of original images. Therefore, in this final phase, a Convolutional Neural Network (CNN) is implemented to upsample Y, U, and V channels, taking into account the entire image and resulting in a more accurate decoded image. 

The CNN is trained and validated using the DIV2K dataset and outputs integer values within the range of 0 to 255. The project also includes learning about image quality metrics such as SSIM and PSNR, with the recognition that sometimes PSNR may not align with human perception of image quality.
