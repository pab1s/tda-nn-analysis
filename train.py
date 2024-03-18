from tqdm import tqdm
from utils.plotting import plot_loss
from utils.logging import log_to_csv


def train(model, train_loader, device, criterion, optimizer, num_epochs, log_path, plot_path, verbose=True):
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        if verbose:
            progress_bar = tqdm(enumerate(train_loader, 1), total=len(
                train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        else:
            progress_bar = enumerate(train_loader, 1)

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if verbose:
                progress_bar.set_postfix({'loss': running_loss / batch_idx})

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        log_to_csv(epoch_losses, log_path)

    if verbose:
        plot_loss(epoch_losses, plot_path)
