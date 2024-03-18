import csv

def log_to_csv(epoch_losses, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])
        for epoch, loss in enumerate(epoch_losses, 1):
            writer.writerow([epoch, loss])
