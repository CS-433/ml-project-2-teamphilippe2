# Machine Learning - Project 2 - RoadSegmentation - TeamPhilippe2

### Team Members
- Eloïse Doyard (272300)
- Alessio Verardo (282634)
- Cyrille Pittet (282445)


### Environment 
A list of all the packages (and the corresponding version) required in order to run our code can be found in the requirements.txt file. 
To quickly install all the packages with the corresponding version, you can simply run the command 
```
pip install -r requirements.txt
```

We used the following frameworks for our models :
 - Sklearn : baseline models
 - PyTorch : deep learning models

### Structure of the project
We separated the code into multiple files instead of having all the implementations in a single huge unreadable file.
```
├── data
├── helper
    ├── autoencoder_helper.py             # Helper function to extract feature from patches using the auto-encoder
    ├── const.py                          # All the constants used throughout the project
    ├── datasets_image.py                 # Data augmentation for the FCN and Unet neural networks
    ├── datasets_patch.py                 # Datasets for the Autoencoder network
    ├── image.py                          # Functions to handle images
    ├── loading.py                        # all the loading functions for training and testing dataset
    ├── metrics.py                        # Functions to assess performance of our datasets
    ├── neural_net.py                     # Functions related to the training of the datasets
    ├── predictions.py                    # functions to predict whether a pixel is a road or background in the test data
    ├── submission.py                     # Functions used to create submissions for AICrowd
    ├── visualisations.py                 # Functions used to make differnet visualisations of the images
├── models
    ├── autoencoder.py                    # The architecture of the auto-encoder neural network described in section 2.2.2 of the report
    ├── FCNet.py                          # The Fully convolutional network architecture described in part 2.2.4 of the report
    ├── features_extraction.py            # Functions used to extract features from image or image patch
    ├── UNet.py                           # The UNet architecture architecture described in part 2.2.3 of the report
├── output
    ├── features                          # Features outputed by the autoencoder neural network. 
    ├── weights                           # Save of the different weights output during the training of our networks
├── autoencoder.ipynb                     # Notebook detailling the implementation and results of the autoencoder
├── baseline.ipynb                        # Initial data exploration and baseline model training
├── neural_net_training.ipynb             # Notebook used to train our neural netwokrs
├── report.pdf                            # The report of the project
├── requirements.txt                      # List of all the packages (and versions) needed to run our project
└── README.md
```

### Reproduce our best run
We provide a run.py file which does the following :
1. Load the test set data and resize the test data (as they do not match the size of the training set)
2. Instantiate the model and load saved weights from the corresponding folder.
3. Predict the roads on each test image
4. Resize back the images to their original size.
5. Create the submission in the expected format and save it as a csv file called 'submission.csv'

To run this script, you will need Python 3, numpy, zipfile and matplotlib installed. To execute it you can :
- Simply run it in a terminal : ```python3 run.py```
- Or call the main function from within a notebook : 
```python
from run import main
main()
```
