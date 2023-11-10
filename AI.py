import torch
import torchvision
from PIL import Image
import json
import numpy as np

alexnet = torchvision.models.alexnet(weights=None)
alexnet.load_state_dict(torch.load('data/alexnet/alexnet.pth'))

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224),),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])
# Put the model in evaluation mode
alexnet.eval()

def reco(M):
    im = Image.fromarray(np.uint8(M))

    im_normalized = preprocess(im)[None,:]

    # Run the CNN
    y = alexnet(im_normalized)
    # Get the best class index and score
    best, bestk = y.max(dim=1)
    # Get the corresponding class name
    with open('data/imnet_classes.json') as f:
        classes = json.load(f)
    name = classes[str(bestk.item())][1]

    return name