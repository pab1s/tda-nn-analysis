import math
import torch
from typing import Tuple, List

def find_lr(model, train_loader, criterion, optimizer_class, optimizer_params, init_value=1e-8, final_value=1e-1, beta=0.98, device=None) -> Tuple[List[float], List[float]]:
    """
    Finds the learning rate range for training a model using the method proposed by Leslie N. Smith.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        criterion: The loss function.
        optimizer_class: The optimizer class to use.
        optimizer_params (dict): Additional parameters for the optimizer.
        init_value (float, optional): The initial learning rate value. Defaults to 1e-8.
        final_value (float, optional): The final learning rate value. Defaults to 1e-1.
        beta (float, optional): The smoothing factor for computing the average loss. Defaults to 0.98.
        device (torch.device, optional): The device to use for training. If None, uses CUDA if available, otherwise CPU.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing the logarithm of the learning rates and the corresponding losses.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Number of batches
    num = len(train_loader) - 1
    if num <= 0:
        raise ValueError("The training loader must contain more than one batch to compute the learning rate range test.")

    lr = init_value
    optimizer_params['lr'] = lr
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    # Obtain the multiplicative factor
    mult = (final_value / init_value) ** (1 / num)
    avg_loss = 0.
    best_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []

    for data in train_loader:
        batch_num += 1
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()

        # Update the learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses
