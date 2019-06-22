import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np

DENSE_SIZE = 50 

def init_device():
    """
    set up to run models / predictions on cuda if available, or cpu if not
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda: 
        torch.cuda.empty_cache()
    return device

def deviations_to_decoded(deviations, batch_size=1):
    """
    Take an array of deviations from the mean for each of the variables, 
    and run it through the model to produce a decoded image. 
    """
    if isinstance(deviations, np.ndarray): 
        actual_deviations = torch.from_numpy(deviations).type(torch.FloatTensor).to(device)
    elif isinstance(deviations, torch.Tensor):
        actual_deviations = deviations.type(torch.FloatTensor).to(device)
    elif isinstance(deviations, int):
        actual_deviations = torch.FloatTensor([deviations]).repeat(DENSE_SIZE).to(device)

    decoded = model.sample(mus.repeat(batch_size, 1), 
                           std_devs.repeat(batch_size, 1), deviation=actual_deviations)
    decoded = decoded.data.numpy().reshape((28, 28))
    return decoded

device = init_device()

# create model and load trained weights 
from model import Net 
torch.manual_seed(42)
model = Net(device).to(device)

vae_weights_path = "model/vae_weight.pt"
model.load_state_dict(torch.load(vae_weights_path, map_location=lambda storage, location: storage))
model.eval()

# load trained means / std deviations for features
mus_path = "model/mus.pt"
std_dev_path = "model/std_devs.pt"

mus = torch.load(mus_path).type(torch.FloatTensor)
std_devs = torch.load(std_dev_path).type(torch.FloatTensor)

# create a sample from our means and 
an_image = deviations_to_decoded(42)