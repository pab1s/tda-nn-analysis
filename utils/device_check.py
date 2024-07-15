import torch

def check() -> None:
    """
    Checks the availability of CUDA and prints the current device being used.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

if __name__ == "__main__":
    check()
