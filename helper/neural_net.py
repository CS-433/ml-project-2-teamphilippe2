import torch
from datetime import datetime
from helper.metrics import *
from helper.const import *

def train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs, print_iteration=True):
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
    Returns:
    -----------
        model: The trained model
    """
    print("Starting training")
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        # Test the quality on the test set
        # Disable dropout and batch-norm layers for instance
        model.eval()
        
        # Disable gradients computation
        torch.no_grad()

        accuracies_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies_test.append(accuracy(prediction, batch_y))
        if print_iteration:
            print("Epoch {} | Test accuracy: {:.5f}".format(epoch, sum(accuracies_test).item()/len(accuracies_test)))
    print("End of training")
    return model


def loss_function_from_string(loss_fct_str):
    if loss_fct_str == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    else:
        return None
    
def model_from_string(model_str):
    if model_str == "unet":
        return torch.nn.CrossEntropyLoss()
    else:
        return None

def optimizer_from_string(optimizer_str, params, lr,momentum):
    if optimizer_str == "Adam":
        return torch.optim.Adam(params, lr=learning_rate, momentum=momentum)
    elif optimizer_str=="SGD":
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
    else:
        return None

def run_experiment(model_str, loss_fct_str, optimizer_str, image_dir, gt_dir, num_epochs=1, learning_rate=1e-3, momentum=0.0, batch_size = 1000, save_weights=True,ratio_test=0.2, seed=1):
    num_epochs = 10
    learning_rate = 1e-3
    batch_size = 1000
    
    ds = AugmentedRoadImages(image_dir, gt_dir, ratio_test)
    
    dataset_train = torch.utils.data.DataLoader(ds,
      batch_size=batch_size,
      shuffle=True
    )

    # If a GPU is available, use it
    if not torch.cuda.is_available():
        print("Things will go much quicker with a GPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Train the logistic regression model with the Adam optimizer
    criterion = loss_function_from_string(loss_fct_str)
    model = LogisticRegressionModel().to(device)
    optimizer = optimizer_from_string(optimizer_str, model.parameters(), learning_rate,momentum)
    
    train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs)
    
    if save_weights:
        now = datetime.now()
        torch.save(model.state_dict(), weights_folder+model_str+"/"+now.strftime("%Y-%m-%d_%H-%M-%S")+ext_weight_model)
    
    # Compute scores on the local test set 
    img_test, gt_test = ds.get_test_set()
    predict_test_set_nn(img_test, model)
    
    # Display scores
    _, _, _, _ = compute_scores(preds, gt_test)
    
