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
### Structure of the project
We separated the code into multiple files instead of having all the implementations in a single huge unreadable file.

- report.pdf : the report for this project
- autoencoder.ipynb:
- baseline.ipynb: 
- neural_net_training.ipynb:
- run.py : script to reproduce our results
- data :
    - data.zip : file containing the training and test data
- helper :
    - autoencoder_helper.py : Helper function to extract feature from patches using the auto-encoder
    - baseline.py : Function used to fit the baseline classification models
    - const.py : all the constants used in the project
    - data_augmentation.py : Datasets and data augmentation used for the training of the neural networks 
    - image.py : functions to handle images
    - loading.py : all the loading functions for training and testing dataset
    - metrics.py : function to assess performance of our datasets
    - neural_net.py : functions related to the training of the datasets
    - submission.py : functions used to create submissions for AICrowd
    - visualisations.py : functions used to make sdiffernet visualisations of the images 
- models :
  - autoencoder.py : The architecture of the auto-encoder neural network, i.e. the architectures of the encoder and the decoder
  - features_extraction.py : functions used to extract features from image or image patch
  - NNET : The Fully connected network architecture described in part 2.2.2 of the report
  - predictions.py : functions to predict whether a pixel is a road or background in the test data.
  - UNet: The UNet architecture architecture described in part 2.2.1 of the report
- output: 
  - features: saves of the features extracted by the auto-encoder.
  - weights: folder containing saves of the different model weights.
- plots:
  - Folder to save different visualisations.
  

### How to run the code
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
