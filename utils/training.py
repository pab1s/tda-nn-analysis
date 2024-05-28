import math
import torch

def find_lr(model, train_loader, criterion, optimizer_class, optimizer_params, init_value=1e-8, final_value=10, beta=0.98, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num = len(train_loader) - 1
    if num <= 0:
        raise ValueError("The training loader must contain more than one batch to compute the learning rate range test.")

    lr = init_value
    optimizer_params['lr'] = lr
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

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

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()

        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses
