from datetime import datetime
from helper.data_augmentation import *
from helper.metrics import *
from models.ConvNet import ConvNet
from models.FCNet import *
from models.UNet import *
from models.autoencoder import AutoEncoder
from helper.predictions import predict_test_set_nn

def run_experiment(model_str, loss_fct_str, optimizer_str, image_dir, gt_dir, num_epochs=10, learning_rate=1e-3,
                   momentum=0.0, batch_size=16, save_weights=True, ratio_train=0.8, seed=1, im_patch=None, 
                   lr_scheduler=False, lr_schedule=(10, 0.1), verbose=True, pos_weight=False):
    """
    Fully train the asked neural network, save the weights and test the accuracy on the test set. 
    Parameters:
    -----------
        - model_str: 
            The name of the model we want to train
        - loss_fct_str:
            The name of the loss function we want to use
        - optimizer_str:
            The name of the optimizer we want to use
        - image_dir : 
             the folder containing all the training images
        - gt_dir : 
            The folder containing all the groundtruth train images
        - num_epochs:
            The number of training step we want to do
        - learning_rate:
            The learning rate we want to apply during training
        - momentum:
            The momentum we want to include in the optimiser (supporting it)
        - batch_size
            The number of data at the same time in training
        - save_weights:
            Boolean to tell if we need to save the weights of the model or not
        - ratio_test:
            The ratio of data we use during training
        - seed:
            The seed to use during splitting
        - im_patch:
            String indicating whether we are training an autoencoder,
            and if we use patches or complete images as input. None
            if we do not train an autoencoder.
        - lr_scheduler:
            Boolean indicating whether to use a learning rate scheduler
        - lr_schedule:
            Tuple (epochs step, division factor) for the learning rate scheduler
        - verbose:
            Boolean indicating whether to print the loss in each epoch
        - pos_weight:
            Boolean to indicate whether to use class balancing during training
    Returns: 
    -----------
        - Train losses: the loss on the train set obtained during training
        - Test losses : the test losses obtained during training or an empty list if we do not use the autoencoder
        - f1s_test: the f1 score on the test set obtained during training or an empty list if we use the autoencoder
        - accuracies_test: the accuracies on the test set obtained during training or an empty list if we use the autoencoder
    """
    
    # Load the datasets and the device
    dataset_train, dataset_test, ds, dstest = get_datasets(image_dir, gt_dir, ratio_train, batch_size, im_patch, seed)
    device = get_device()
    
    pos_weight_tensor = None
    if pos_weight:
        weight = ds.compute_pos_weight()
        pos_weight_tensor = torch.Tensor(weight).to(device)

    # Get the model
    model = model_from_string(model_str, device)

    # Get the loss
    criterion, categorical = loss_function_from_string(loss_fct_str, pos_weight_tensor=pos_weight_tensor)
    
    threshold = 0.0 if loss_fct_str == "bcelogit" else 0.5

    # Get the optimiser
    optimizer = optimizer_from_string(optimizer_str, model.parameters(), learning_rate, momentum)

    scheduler = None
    if lr_scheduler:
       # Use a step lr scheduler
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_schedule[0], gamma=lr_schedule[1])
    
    train_losses, test_losses, f1s_test, accuracies_test = train(model, criterion, dataset_train, dataset_test,\
                                                                 device, optimizer, num_epochs, im_patch=im_patch,\
                                                                 scheduler=scheduler, print_iterations=verbose,\
                                                                 categorical=categorical, threshold = threshold)
    
    if save_weights:
        save_model(model_str, model, im_patch)

    if im_patch != "patches":
        print("Predicting pixels...")
        # Compute scores on the local test set 
        img_test, gt_test = ds.get_test_set()
        
        gt_test = [gt.numpy() for gt in gt_test]

        preds = predict_test_set_nn(img_test, model, threshold=threshold)
            
        # Display scores
        _, _, _, _ = compute_scores(gt_test, preds)
    
    return train_losses, test_losses, f1s_test, accuracies_test


def train(model, criterion, dataset_train, dataset_test, device, optimizer, num_epochs, print_iterations=True, 
          im_patch=None, scheduler=None, categorical=False, threshold=0.5):

    """
    Fully train a neural network
    Parameters:
    -----------
        - model: 
            The model we want to train (torch.nn.Module)
        - criterion:
            The loss function we need to use to assess performances (torch.nn.Module)
        - dataset_train:
            The train dataset to use (torch.utils.data.DataLoader)
        - dataset_text:
            The test dataset to use (torch.utils.data.DataLoader)
        - device:
            The device in which the model is stored
        - optimizer:
            The optimizer used during training phase (torch.optim.Optimizer)
        - num_epochs:
            The number of training passes over the whole dataset we need to make (int)
        - print_iterations:
            Boolean inidicating if we want to print a summary of performance after 10 epoch
        - im_patch:
            String indicating whether we are training an autoencoder,
            and if we use patches or complete images as input. None
            if we do not train an autoencoder.
        - scheduler:
            Learning rate scheduler (None if do not use a scheduler)
        - categorical: 
            Boolean indicating whether the loss needs categorical feature
        - threshold: 
            value at which to threshold the output of the model to obtain the pixel values
    Returns: 
    -----------
        - Train losses: the loss on the train set obtained during training
        - Test losses : the test losses obtained during training or an empty list if we do not use the autoencoder
        - f1s_test: the f1 score on the test set obtained during training or an empty list if we use the autoencoder
        - accuracies_test: the accuracies on the test set obtained during training or an empty list if we use the autoencoder
    """
    train_losses = []
    test_losses = []
    f1s_test = []
    accuracies_test = []

    print("Starting training")
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        loss_sum = 0
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            
            if categorical:
                # if the loss want categorical tensor, need to convert the tensor in NxCxHxW into NxHxW
                batch_y = batch_y.squeeze(dim=1).type(torch.int64)
            
            # Evalute the network on the training groundtruth and store loss
            loss = criterion(prediction, batch_y)
            loss_sum += loss.item()

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()
        
        train_losses.append(loss_sum)
        
        if im_patch=="patches":
            # Evaluate autoencoder performance
            test_loss = evaluate_autoencoder(model, criterion, dataset_test, device, print_iterations, epoch)
            test_losses.append(test_loss)
        else:
            # Evalute classifier performance
            f1, acc = evalute_classifier(model, criterion, dataset_test, device, print_iterations, epoch, threshold)
            f1s_test.append(f1)
            accuracies_test.append(acc)

        # Update the lr scheduler
        if scheduler is not None:
            scheduler.step()

    print("End of training")
    return train_losses, test_losses, f1s_test, accuracies_test


def evaluate_autoencoder(model, criterion, dataset_test, device, print_iterations, epoch):
    """
    Evaluate the autoencoder performance on the test set 
    Parameters:
    -----------
        - model:
            The model we wish to assess performances (torch.nn.Module)
        - criterion:
            The loss we use during training (torch.nn.Module)
        - dataset_test:
            The test dataset on which we assess performance (torch.utils.data.DataLoader)
        - device: 
            The device in which the model is stored 
        - print_iterations:
            Boolean inidicating if we want to print a summary of performance after 10 epochs
        - epoch:
            The current epoch number
        - threshold: 
            value at which to threshold the output of the model to obtain the pixel values
    Returns: 
    -----------
        - test_losses_sum:
            The loss on the test set for the current epoch
    """
    # Test the quality on the test set
    # Disable dropout and batch-norm layers for instance
    model.eval()

    test_losses_sum = 0
    for batch_x, batch_y in dataset_test:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        prediction = model(batch_x)
        t_loss = criterion(prediction, batch_y).item()
        test_losses.append(t_loss)
        test_losses_sum += t_loss

    if print_iterations and epoch%10 ==0:
        print(f"Epoch {epoch} | Avg test loss: {test_losses_sum / len(dataset_test):.5f}")
    return test_losses_sum

def evalute_classifier(model, criterion, dataset_test, device, print_iterations, epoch, threshold):
    """
    Evaluate the autoencoder performance on the test set 
    Parameters:
    -----------
        - model:
            The model we wish to assess performances (torch.nn.Module)
        - criterion:
            The loss we use during training (torch.nn.Module)
        - dataset_test:
            The test dataset on which we assess performance (torch.utils.data.DataLoader)
        - device: 
            The device in which the model is stored 
        - print_iterations:
            Boolean inidicating if we want to print a summary of performance after 10 epochs
        - epoch:
            The current epoch number
    Returns: 
    -----------
        - mean_f1:
            The f1 score on the test set for the current epoch
        - mean_acc:
            The accuracy on the test set for the current epoch 
    """
    # Test the quality on the test set
    # Disable dropout and batch-norm layers for instance
    model.eval()
    
    f1s = []
    accuracies = []
    with torch.no_grad():
        # Evaluate the accuracy on the test set for the other models
        
        for batch_x, batch_y in dataset_test:
            # Predict the class of each pixel in the test set 
            preds = predict_test_set_nn(batch_x, model, threshold=threshold)
            # Convert the groundtruth to numpy as it is the only format allowed by sklearn
            gt_test = [gt.numpy() for gt in batch_y]
            # Compute accuracy and f1 score on the test set
            f1, _, _, accuracy = compute_scores(gt_test, preds, print_values=False)
            f1s.append(f1)
            accuracies.append(accuracy)
            
    mean_f1, mean_acc = np.mean(f1s), np.mean(accuracies)
    
    if print_iterations and epoch%10 ==0:
        print("Epoch {} | Test F1: {:.5f} | Test accuracy: {:.5f}".format(epoch, mean_f1, mean_acc))
    return mean_f1, mean_acc

def loss_function_from_string(loss_fct_str, pos_weight_tensor=None):
    """
    Return the loss function corresponding to the given string
    Parameters: 
    -----------
        - loss_fct_str: 
            The name of the loss function we want to get
        - pos_weight:
            Array of weights for the different classes in the loss
    Returns: 
    -----------
        - The corresponding loss function
        - Boolean indicating whether the loss accept only a two-dimensional array with the category of the pixel or not
    """
    if loss_fct_str == "cross-entropy":
        return torch.nn.CrossEntropyLoss(), True
    elif loss_fct_str == "bce":
        return torch.nn.BCELoss(), False
    elif loss_fct_str == "mse":
        return torch.nn.MSELoss(), False
    elif loss_fct_str == "bcelogit":
        if pos_weight_tensor is not None:
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor), False
        else:
            return torch.nn.BCEWithLogitsLoss(), False
    elif loss_fct_str == "dice":
        return torchgeometry.losses.DiceLoss(), True
    elif loss_fct_str == "tversky":
        return torchgeometry.losses.TverskyLoss(alpha=0.3, beta=0.7), True
    else:
        raise ValueError(f"Unexpected value {loss_fct_str}")


def model_from_string(model_str, device):
    """
    Return the model corresponding to the given string and print statistics about this model
    Parameters: 
    -----------
        - model_str: 
            The name of the model we want to get
        - device: 
            The device on which to store the model
    Returns: 
    -----------
        - The corresponding model
    """
    if model_str == "unet":
        model = UNet(3, 64)
    elif model_str == "nnet":
        model = FCNet()
    elif model_str == "autoencoder":
        model = AutoEncoder()
    elif model_str == "convnet":
        model = ConvNet()
    else:
        raise ValueError(f"Unexpected value {model_str}")
    
    # Move the models to the correct device
    model = model.to(device)
    # Display model statistics
    print(f"Number of parameters in the model {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if torch.cuda.is_available():
        print(f"Cuda memory allocated for training {torch.cuda.memory_allocated()/1024/1024} MB")
    return model


def optimizer_from_string(optimizer_str, params, lr, momentum):
    """
    Return the optimiser corresponding to the given string
    Parameters: 
    -----------
        - optimizer_str: 
            The name of the optimizer we want to get
        - params:
            The parameters of the model
        - lr:
            The learning rate we want to apply during training
        - momentum: 
            The momentum we want to include in the optimiser
    Returns: 
    -----------
        - The corresponding optimiser
    """
    if optimizer_str == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif optimizer_str == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unexpected value {optimizer_str}")
        

def get_datasets(image_dir, gt_dir, ratio_train, batch_size, im_patch,seed):
    """
    """
    if im_patch == 'patches':
        # Do not use supervision for autoencoder
        # Training dataset
        ds = AutoencoderTrainingRoadPatches(image_dir)
        dataset_train = torch.utils.data.DataLoader(ds,
                                                    batch_size=batch_size,
                                                    shuffle=True
                                                    )

        # For autoencoder, test set on true "test" (used for AICrowd) set
        # as it is unsupervised
        dstest = AutoencoderTestRoadPatches(gt_dir)
        dataset_test = torch.utils.data.DataLoader(dstest,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    elif im_patch == 'patches_sup':
        # Use supervision
        ds = ConvNetTrainingRoadPatches(image_dir, gt_dir, ratio_train=ratio_train)
        dataset_train = torch.utils.data.DataLoader(ds,
                                                    batch_size=batch_size,
                                                    shuffle=True
                                                    )

        dstest = ConvNetTestRoadPatches(ds)
        dataset_test = torch.utils.data.DataLoader(dstest,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    else:
        # Training dataset
        ds = AugmentedRoadImages(image_dir, gt_dir, ratio_train, seed)
        dataset_train = torch.utils.data.DataLoader(ds,
                                                    batch_size=batch_size,
                                                    shuffle=True
                                                    )
        # Local test set
        dstest = RoadTestImages(ds)
        dataset_test = torch.utils.data.DataLoader(dstest, batch_size=batch_size, shuffle=True)
    return dataset_train, dataset_test, ds, dstest

def get_device():
    """
    Return the device on which to train the neural network, i.e. CUDA if available or CPU
    Return:
        - device: the device on which to train the neural network
    """
    torch.cuda.empty_cache()
    # Try to move the model to the GPU or the CPU
    device = torch.device("cuda")

    # If a GPU is available, use it
    if not torch.cuda.is_available():
        print("Things will go much quicker with a GPU")
        device = torch.device("cpu")
    return device
        
def save_model(model_str, model, im_patch):
    """
    Save the model weights in the weights folder
    Parameters:
        - model_str: the model name
        - model: the model object containing the weights
        - im_patch:
            String indicating whether we are training an autoencoder,
            and if we use patches or complete images as input. None
            if we do not train an autoencoder
    """
    # weights_folder and ext_weight_model are two constants defined in helper/const.py
    
    # Get the current time to have a unique file name
    now = datetime.now()
    time = now.strftime("%Y-%m-%d_%H-%M-%S")

    if im_patch == "patches":
        # If we used an autoencoder,
        # separate the weights files
    
        # Save encoder weights
        torch.save(model.encoder.state_dict(),
                   weights_folder + model_str + "/encoder_" + time + ext_weight_model)

        # Save decoder weights
        torch.save(model.decoder.state_dict(),
                   weights_folder + model_str + "/decoder_" + time + ext_weight_model)
    else:
        torch.save(model.state_dict(),
                   weights_folder + model_str + "/" + time + ext_weight_model)



