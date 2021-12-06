import torch
from datetime import datetime
from helper.metrics import *
from helper.const import *
from models.UNet import *
from models.NNET import *
from helper.data_augmentation import *
from models.autoencoder import AutoEncoder
from models.predictions import predict_test_set_nn


def train(model, criterion, dataset_train, dataset_test, device, optimizer, num_epochs, print_iteration=True,
          autoencoder=False, scheduler=None):
    """
    Fully train a neural network
    Parameters:
    -----------
        model: 
            The model we want to train (torch.nn.Module)
        criterion:
            The loss function we need to use to assess performances (torch.utils.data.DataLoader
        dataset_train:
            The train dataset to use (torch.utils.data.DataLoader)
        dataset_text:
            The test dataset to use (torch.utils.data.DataLoader)
        optimizer:
            The optimizer used during training phase (torch.optim.Optimizer)
        num_epochs:
            The number of training passes over the whole dataset we need to make (int)
        print_iterations:
            If we want to print a summary of performance after an epoch
        autoencoder:
            Whether we are training the autoencoder
        scheduler:
            Learning rate scheduler (None if do not use a scheduler)
    Returns:
    -----------
        model: The trained model
        train_losses: Losses on the train set
        test_losses: Losses on the test set
    """
    test_losses = []
    train_losses = []
    
    print("Starting training")
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            train_losses.append(loss.item())

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        # Test the quality on the test set
        # Disable dropout and batch-norm layers for instance
        model.eval()

        # Disable gradients computation
        with torch.no_grad():
            # Evaluate the loss on the test set for the autoencoder
            if autoencoder:
                test_losses_sum = 0
                for batch_x, batch_y in dataset_test:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    prediction = model(batch_x)

                    t_loss = criterion(prediction,  batch_y).item()
                    test_losses.append(t_loss)
                    test_losses_sum += t_loss

                if print_iteration:
                    print(f"Epoch {epoch} | Avg test loss: {test_losses_sum / len(dataset_test):.5f}")
            else:
                # Evaluate the accuracy on the test set for the autoencoder
                accuracies_test = []
                for batch_x, batch_y in dataset_test:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Evaluate the network (forward pass)
                prediction = model(batch_x)
                prediction[prediction >= 0.5] = 1
                prediction[prediction < 0.5] = 0

                    accuracies_test.append((batch_y.detach().numpy() == prediction.detach().numpy()).mean())
                if print_iteration:
                    print("Epoch {} | Test accuracy: {:.5f}".format(epoch, sum(accuracies_test).item() / len(accuracies_test)))
                    

        # Update the lr scheduler
        if scheduler is not None:
            scheduler.step()
    
    print("End of training")
    return model, train_losses, test_losses


def loss_function_from_string(loss_fct_str):
    """
    Return the loss function corresponding to the given string
    Parameters: 
    -----------
        - loss_fct_str: 
            The name of the loss function we want to get
    Returns: 
    -----------
        - The corresponding loss function
    """
    if loss_fct_str == "cross-entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_fct_str == "bce":
        return torch.nn.BCELoss()
    elif loss_fct_str == "mse":
        return torch.nn.MSELoss()
    else:
        return None


def model_from_string(model_str):
    """
    Return the model corresponding to the given string
    Parameters: 
    -----------
        - model_str: 
            The name of the model we want to get
    Returns: 
    -----------
        - The corresponding model
    """
    if model_str == "unet":
        return UNet(400, 64)
    elif model_str == "nnet":
        return NNet()
    elif model_str == "autoencoder":
        return AutoEncoder()
    else:
        return None


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
        return None


def run_experiment(model_str, loss_fct_str, optimizer_str, image_dir, gt_dir, num_epochs=10, learning_rate=1e-3,
                   momentum=0.0, batch_size=16, save_weights=True, ratio_test=0.2, seed=1, autoencoder=False,
                  lr_scheduler=False, lr_schedule=(10, 0.1)):

    """
    Fully train the asked neural network, save the weights and test the accuracy on the test set. 
    Parameters:
    -----------
        - model_str: 
            The name of the model we want to get
        - loss_fct_str:
            The name of the loss function we want to get
        - optimizer_str:
            The name of the optimizer we want to get
        - image_dir : 
             the folder containing all the training images
        - gt_dir : 
            The folder containing all the groundtruth train images
        - num_epochs:
            The number of training step we want to do
        - learning_rate:
            The learning rate we want to apply during training
        - momentum:
            The momentum we want to include in the optimiser
        - batch_size
            The number of data we want to use in each epoch of training
        - save_weights:
            If we need to save the weights of the model or not
        - ratio_test:
            The train-test set split ratio
        - seed:
            The seed to use during splitting
        - autoencoder:
            Boolean indicating whether we are training an autoencoder
        - lr_scheduler:
            Boolean indicating whether to use a learning rate scheduler
        - lr_schedule:
            Tuple (epochs step, division factor) for the learning rate scheduler
    Returns: 
    -----------
        - Train losses
        - Test losses
    """


    if autoencoder:
        # Training dataset
        ds = OriginalTrainingRoadPatches(image_dir)
        dataset_train = torch.utils.data.DataLoader(ds,
                                                    batch_size=batch_size,
                                                    shuffle=True
                                                    )

        # Test set on true "test" (used for AICrowd) set
        # as autoencoder is unsupervised
        dstest = OriginalTestRoadPatches(gt_dir)
        dataset_test = torch.utils.data.DataLoader(dstest,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    else:
        ds = AugmentedRoadImages(image_dir, gt_dir, ratio_test, seed)
        dataset_train = torch.utils.data.DataLoader(ds,
                                                    batch_size=batch_size,
                                                    shuffle=True
                                                    )
        dstest = RoadTestImages(ds)
        dataset_test = torch.utils.data.DataLoader(dstest, batch_size=batch_size, shuffle=True)


    device = torch.device("cuda")

    # If a GPU is available, use it
    if not torch.cuda.is_available():
        print("Things will go much quicker with a GPU")
        device = torch.device("cpu")

    # Train the logistic regression model with the Adam optimizer
    criterion = loss_function_from_string(loss_fct_str)
    torch.cuda.empty_cache()
    model = model_from_string(model_str).to(device)
    optimizer = optimizer_from_string(optimizer_str, model.parameters(), learning_rate, momentum)
    
    scheduler = None
    if lr_scheduler:
       # Use a step lr scheduler
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_schedule[0], gamma=lr_schedule[1])

    _, train_losses, test_losses = train(model, criterion, dataset_train, dataset_test, device, optimizer, num_epochs, autoencoder=autoencoder, scheduler=scheduler)

    if save_weights:
        now = datetime.now()

        # If we used an autoencoder,
        # separate the weights files
        if autoencoder:
            # Save encoder weights
            torch.save(model.encoder.state_dict(),
                       weights_folder + model_str + "/encoder_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ext_weight_model)

            # Save decoder weights
            torch.save(model.decoder.state_dict(),
                       weights_folder + model_str + "/decoder_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ext_weight_model)
        else:
            torch.save(model.state_dict(),
                       weights_folder + model_str + "/" + now.strftime("%Y-%m-%d_%H-%M-%S") + ext_weight_model)

    if not autoencoder:
        # Compute scores on the local test set 
        img_test, gt_test = ds.get_test_set()
        preds = predict_test_set_nn(img_test, model)

        # Display scores
        _, _, _, _ = compute_scores(preds, gt_test)
    
    return train_losses, test_losses


def load_model_weights(model, weights_path):
    """
    Loads the weights from the given file
    Parameters:
    -----------
        - model:
            Model of the network
        - weights_path:
            Path of the file
    Returns:
    -----------
    """
    model.load_state_dict(torch.load(weights_path))
