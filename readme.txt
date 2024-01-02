1. Install packages from requirements.txt (if you are running the code in Colab, use "!pip install bm3d" for installation of bm3d)
2. Open file "Project LIME.ipynb"
3. Run all code blocks (Processing of all given images takes approximately 7 min)
4. Processing results are saved in Output folders in ./Photos/HDR and ./Photos/ExDark directories (if you are running the code in Colab, all these directories should be created and all images which are supposed to be processed should be uploaded there in advance before the code launch)

Note: 
The first block of code contains all necessary functions for running the algorithm.

The second block performs enhancement of images from the original paper in order to evaluate the lightness order error (LOE). The list of images names is specified in case of LOE calculation. So, if you'd like to calculate LOE for different images, you should place the image and its reference in the folder HRD, name the reference as "imagename_HDR", and add the name of the image (imagename) into the list variable "name_list". If the image has an extension different from ".tif", it should be modified in this block. 

The third block performs enhancement of several images from the Exclusive Dark Dataset. The list of images names is not specified in this case. The code just reads all the files from the ExDark directory and processes those of them which have .jpg, .bmp, .png or .tif extensions. Thus, if you would like to process other images, it is sufficient just to place them into ExDark folder.

Weight strategy number (1, 2 or 3), gamma and standard deviation for BM3D algorithm could be changed at the top part of the second and the third block. Atmospheric light alpha-parameter and parameters of the gaussian kernel for the third weight strategy (sigma and kernel size) could be changed in the first block of code inside the corresponding functions. Alpha could be found in "update_illumination_map" and kernel parameters are in "initialize_weights".

