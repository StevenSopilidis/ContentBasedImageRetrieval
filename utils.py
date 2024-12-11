import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    """
    Class that is used to extract features from image using resnet50 model

    Attributes:
        model: pretrained resnet50 model
    
    """
    def __init__(self):
        resnet = models.resnet50(pretrained=True)
        module_list = list(resnet.children())[:-1]
        model = nn.Sequential(*module_list)
        model.eval()
        self.model = model

        # preprocessing transformation applied before passing the image
        # to the model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

    def extract(self, image: Image) -> np.array:
        """
        Function that extracts features from given image

        Parameters:
            image: input image

        Returns:
            torch.tensor: features extracted from the image
        """
        X = self.transform(image)
        X = X.unsqueeze(0) # add batch dimension

        with torch.no_grad():
            features = self.model(X)
            features = features.squeeze() # remove batch dimension

        return features.numpy()
    

def load_image(path: str) -> Image:
    """
    Function for laoding an image

    Parameters:
        path: path to given image

    Returns:
        Image: image that was loaded from disk
    """
    image = Image.open(path).convert("RGB")
    return image
