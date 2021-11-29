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