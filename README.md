# cifar1984
cifar1984 is a custom cifar10 model designed using Flask and PyTorch. cifar1984 calculates the likelihood of an object in a given image to be either three of the following object classes: 'Plane', 'Car', 
'Bird ', 'Cat ', 'Deer ', 'Dog ', 'Frog ', 'Horse', 'Ship', 'Truck'.

Mock up live-webpage with in production predictor turned off: http://ccsw.pythonanywhere.com/

(Mock up model's predictions will almost always be false and predictions have low confidence. It's purpose is merely to demonstrate the core functions and structure of this project. )

## Local machine installation and usage:
1. Set up a virtual environment with packages in requirements.txt. (pip install flask, torch, torchvision, matplotlib.)
2. Include cifar1984 files and folders above the virtual environments directory.
3. Activate cifar1984 virtual environment.
4. Run cifar1984app.py

## Files and Folders info:

1. [cifar1984app.py] :Main flask file to handle backend's predictions and frontend's(html) request and presentation.

2. [Net.py] : Class structure of Cifar model wirtten in PyTorch format. A convolutional-neural-network that accepts one 32x32px image into 3 layers which outputs 36 5x5p
convoluted matices, which is then flatten into a 36x5x5 "string" and feed into a 3 layers Fully-Connected-NN. The final result is a prediction of 10 classes.

3. [img_to_tensor.py] : Convets an image of Jpeg or PNG, of any size smaller than 10mb into a 1x32x32 tensor for prediction calculation.
This module includes a built-in procedure to "focus-crop" images with aspect ratio higher than 1.5 either h/w or w/h to better center the
subject in image for predictions calculation.

6. [predictor.py] : Receives input from img_to_tensor.py to calculate the likelihood of the object class in the image. Uses specified model from built from Net.py with
variables from new_model_varb.pth.

7. [new_model_varb.pth] : Variables for model built using CNN class Net.py, contains roughly 670k parameters.

8. [static] : Mainly css files to properly render HTML pages.

9. [templates] : HTML files to receive request image and present prediction results. Includes webpages to provide futher information on this project.
