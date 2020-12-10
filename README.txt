README FILE


The code here was run on a google colab notebook.
You will have to change the directory to your own, containing the dataset and the saved models for the demo.

Required:
keras
tensorflow
pytorch
pandas
numpy
PIL

Minimal pip requirements:

pip install matplotlib

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

pip install torchvision

pip install tensorflow

pip install Keras

pip install Pillow

pip install opencv-python

pip install pandas

pip install numpy

pip install scipy

pip install tqdm

pip install pathlib

pip install scikit-learn

pip install seaborn



There are two notebooks, one is where all the training took place and the other is for the demo, namely Car_Truck_Recognition.ipynb and Demo.ipynb

If you are also using google, you will need to have your dataset in your drive and then mount it onto the notebook session you're on so that it can access your dataset. Make sure to use the GPU runtime so as to speed things up a bit, as the normal runtime can be very slow.

What the project does:

Classifies various car and truck images using the following CNN architectures:

- ResNet50 (pretrained with ImageNet weights readily available online) 
- VGG16 (also pretrained with ImageNet weights)
- LeNet
- A basic CNN model


Why is it useful?:

- At a larger scale, it could be used for things like traffic surveillance systems, traffic monitoring systems, driver assistance systems, automatic toll collection etc.


Getting started:

Have a basic knowledge about CNNs
 
NB. The saved models are inside the datasets folder  #demo notebook and saved models too big to upload on github, even after compression (*to upload demo video instead)


notable source/dataset source: https://www.kaggle.com/helibu/cnn-and-resnet-car-and-truck-classification/
