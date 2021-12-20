from datetime import datetime
from helper.datasets_image import *
from helper.metrics import *
from models.ConvNet import ConvNet
from models.NNET import *
from models.UNet import *
from models.UNet_orig import *
from models.autoencoder import AutoEncoder
from models.predictions import predict_test_set_nn
from tqdm import tqdm


def train(model, criterion, dataset_train, dataset_test, device, optimizer, num_epochs, print_iteration=True,
          im_patch=None, scheduler=None, categorical=False, threshold=0.5):
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
        im_patch:
            String indicating whether we are training an autoencoder,
            and if we use patches or complete images as input. None
            if we do not train an autoencoder.
        scheduler:
            Learning rate scheduler (None if do not use a scheduler)
    Returns:
    -----------
        model: The trained model
        train_losses: Losses on the train set
        test_losses: Losses on the test set
    """
    accuracies_test = []
    test_losses = []
    train_losses = []

    print("Starting training")
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        loss_sum = 0
        for batch_x, batch_y in tqdm(dataset_train):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            if categorical:
                batch_y = batch_y.squeeze(dim=1).type(torch.int64)

            loss = criterion(prediction, batch_y)
            loss_sum += loss.item()

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()
        train_losses.append(loss_sum)

        # Test the quality on the test set
        # Disable dropout and batch-norm layers for instance
        model.eval()

        # Disable gradients computation
        with torch.no_grad():
            # Evaluate the loss on the test set for the autoencoder
            if im_patch == 'patches':
                test_losses_sum = 0
                for batch_x, batch_y in dataset_test:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    prediction = model(batch_x)
                    t_loss = criterion(prediction, batch_y).item()
                    test_losses.append(t_loss)

                    test_losses_sum += t_loss

                test_losses.append(test_losses_sum)
                if print_iteration:
                    print(f"Epoch {epoch} | Avg test loss: {test_losses_sum / len(dataset_test):.5f}")
            else:
                # Evaluate the accuracy on the test set for the other models
                for batch_x, batch_y in dataset_test:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    # Evaluate the network (forward pass)
                    prediction = model(batch_x)

                    prediction[prediction >= threshold] = 1
                    prediction[prediction < threshold] = 0

                    if categorical:
                        best_pred = prediction[:, 1] > prediction[:, 0]
                        accuracies_test.append(
                            (batch_y.cpu().detach().numpy() == best_pred.cpu().detach().numpy()).mean())
                    else:
                        accuracies_test.append(
                            (batch_y.cpu().detach().numpy() == prediction.cpu().detach().numpy()).mean())
                if print_iteration:
                    print("Epoch {} | Test accuracy: {:.5f}".format(epoch,
                                                                    sum(accuracies_test).item() / len(accuracies_test)))

        # Update the lr scheduler
        if scheduler is not None:
            scheduler.step()

    print("End of training")
    return train_losses, test_losses, accuracies_test


def loss_function_from_string(loss_fct_str, pos_weight=None):
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
    """
    if loss_fct_str == "cross-entropy":
        return torch.nn.CrossEntropyLoss(), True
    elif loss_fct_str == "bce":
        return torch.nn.BCELoss(), False
    elif loss_fct_str == "mse":
        return torch.nn.MSELoss(), False
    elif loss_fct_str == "bcelogit":
        if pos_weight is not None:
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight), False
        else:
            return torch.nn.BCEWithLogitsLoss(), False
    elif loss_fct_str == "dice":
        return torchgeometry.losses.DiceLoss(), True
    elif loss_fct_str == "tversky":
        return torchgeometry.losses.TverskyLoss(alpha=0.3, beta=0.7), True
    else:
        raise ValueError(f"Unexpected value {loss_fct_str}")


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
        return UNet(3, 64)
    elif model_str == "nnet":
        return NNet()
    elif model_str == "autoencoder":
        return AutoEncoder()
    elif model_str == "convnet":
        return ConvNet()
    else:
        raise ValueError(f"Unexpected value {model_str}")


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
    elif optimizer_str == "lbfgs":
        return torch.optim.LBFGS(params, lr=lr)
    else:
        raise ValueError(f"Unexpected value {optimizer_str}")


def run_experiment(model_str, loss_fct_str, optimizer_str, image_dir, gt_dir, num_epochs=10, learning_rate=1e-3,
                   momentum=0.0, batch_size=16, save_weights=True, ratio_train=0.8, seed=1, im_patch=None,
                   lr_scheduler=False, lr_schedule=(10, 0.1), verbose=True, pos_weight=False):
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
            Whether to use pos_weight or not
    Returns: 
    -----------
        - Train losses
        - Test losses
    """
    torch.cuda.empty_cache()
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
        # Test set on true "test" (used for AICrowd) set
        dstest = RoadTestImages(ds)
        dataset_test = torch.utils.data.DataLoader(dstest, batch_size=batch_size, shuffle=True)

    # Try to move the model to the GPU or the CPU
    device = torch.device("cuda")

    # If a GPU is available, use it
    if not torch.cuda.is_available():
        print("Things will go much quicker with a GPU")
        device = torch.device("cpu")

    pos_weight_tensor = None
    if pos_weight:
        pos_weight_tensor = ds.compute_pos_weights()
        pos_weight_tensor = pos_weight_tensor.to(device)

    # Create the model
    model = model_from_string(model_str).to(device)

    print(f"Number of parameters in the model {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if torch.cuda.is_available():
        print(f"Cuda memory for the model {torch.cuda.memory_allocated()}")

    # Create the asked loss
    criterion, categorical = loss_function_from_string(loss_fct_str, pos_weight=pos_weight_tensor)

    # load the optimiser
    optimizer = optimizer_from_string(optimizer_str, model.parameters(), learning_rate, momentum)

    scheduler = None
    if lr_scheduler:
        # Use a step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_schedule[0], gamma=lr_schedule[1])

    threshold = 0.0 if loss_function_from_string == "bcelogit" else 0.5

    train_losses, test_losses, accuracies_test = train(model, criterion, dataset_train, dataset_test, device, optimizer,
                                                       num_epochs, im_patch=im_patch, scheduler=scheduler,
                                                       print_iteration=verbose, categorical=categorical,
                                                       threshold=threshold)

    if save_weights:
        now = datetime.now()

        # If we used an autoencoder,
        # separate the weights files
        if im_patch == 'patches':
            # Save encoder weights
            torch.save(model.encoder.state_dict(),
                       weights_folder + model_str + "/encoder_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ext_weight_model)

            # Save decoder weights
            torch.save(model.decoder.state_dict(),
                       weights_folder + model_str + "/decoder_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ext_weight_model)
        else:
            torch.save(model.state_dict(),
                       weights_folder + model_str + "/" + now.strftime("%Y-%m-%d_%H-%M-%S") + ext_weight_model)

    if im_patch != 'patches':
        # Compute scores on the local test set 
        img_test, gt_test = ds.get_test_set()

        gt_test = [gt.numpy() for gt in gt_test]

        preds = predict_test_set_nn(img_test, model, threshold=threshold)

        # Display scores
        _, _, _, _ = compute_scores(gt_test, preds)

    return train_losses, test_losses, accuracies_test


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
