import torch
import pandas as pd

def set_device():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    return device


def get_data(file_path):
    """Make sure dataframe has two column: i.e. text and labels

    Args:
        file_path (string): file path of csv column
    """
    
    df = pd.read_csv(file_path)
    