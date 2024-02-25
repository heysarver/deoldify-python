import torch

def check_cuda_availability():
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        print("CUDA GPU is available for torch functions.")
    else:
        print("CUDA GPU is not available for torch functions.")

if __name__ == "__main__":
    check_cuda_availability()
