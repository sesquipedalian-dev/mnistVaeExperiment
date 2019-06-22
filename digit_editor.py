import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import pygame
from pygame.locals import *

# constants
DENSE_SIZE = 50 
BLACK = (0, 0, 0)

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
    # interpret deviations into a tensor
    if isinstance(deviations, np.ndarray): 
        actual_deviations = torch.from_numpy(deviations).type(torch.FloatTensor).to(device)
    elif isinstance(deviations, torch.Tensor):
        actual_deviations = deviations.type(torch.FloatTensor).to(device)
    elif isinstance(deviations, int):
        actual_deviations = torch.FloatTensor([deviations]).repeat(DENSE_SIZE).to(device)

    # convert to appropriate order
    actual_actual_deviations = np.zeros(DENSE_SIZE)
    for i in range(DENSE_SIZE): 
        actual_actual_deviations[std_dev_by_abs[i]] = actual_deviations[i]
    actual_actual_deviations = torch.from_numpy(actual_actual_deviations).type(torch.FloatTensor).to(device)

    # predict
    decoded = model.sample(mus.repeat(batch_size, 1), 
                           std_devs.repeat(batch_size, 1), deviation=actual_actual_deviations)

    # resize
    decoded = decoded.data.numpy().reshape((28, 28))

    # normalize 
    decoded = (decoded / 2 + .5) * 255
    
    # create RGB
    decoded = np.stack([decoded, decoded, decoded], 0)

    # flip 
    decoded = np.transpose(decoded, (2, 1, 0))

    return decoded

def draw_current_digit(image):
    image_surface = pygame.surfarray.make_surface(image)
    bigger = pygame.transform.scale(image_surface, (50, 50))
    screen.blit(bigger, (0, 0))

class EventsState(object): 
    def __init__(self): 
        self.d = {}

    def __getattribute__(self, name):
        if name != 'd': 
            return self.d.get(name, False)
        else: 
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name != 'd': 
            self.d[name] = value
        else: 
            return super().__setattr__(name, value)

def handle_events(state): 
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            print('keydown {}'.format(event.key))
            # If the Esc key has been pressed set running to false to exit the main loop
            if event.key == K_LSHIFT or event.key == K_RSHIFT:
                state.isShiftPressed = True
            elif event.key == K_ESCAPE:
                state.running = False
        # Check for QUIT event; if QUIT, set running to false
        elif event.type == KEYUP:
            if event.key == K_LSHIFT or event.key == K_RSHIFT:
                state.isShiftPressed = False
        elif event.type == QUIT:
            state.running = False

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

# sort the indices by the amount of impact they have on the resulting digit 
# (how wide the std dev).  
# we'll ues this to convert the current_settings array to the 
# desired dense array for decoding 
std_dev_by_abs = [(i, abs(std_devs[i])) for i in list(range(DENSE_SIZE))]
std_dev_by_abs = sorted(std_dev_by_abs, key=lambda p: p[1], reverse=True)
std_dev_by_abs = [p[0] for p in std_dev_by_abs]

# create the mean digit as our starting image
settings = np.zeros(DENSE_SIZE)
current_image = deviations_to_decoded(settings)

# start pygame
pygame.init()
screen = pygame.display.set_mode((100, 100))

# main loop
state = EventsState()
state.running = True
state.shouldCalculateImage = False
while state.running:
    # get input
    handle_events(state)

    if state.shouldCalculateImage: 
        current_image = deviations_to_decoded(settings)

    # draw stuff
    screen.fill(BLACK)
    draw_current_digit(current_image)

    # copy over new screen buffer
    pygame.display.flip()