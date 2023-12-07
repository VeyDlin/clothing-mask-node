import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from .u2net import U2NET
import requests
from urllib.parse import urlparse


class ClothingSegmentation:
    checkpoint_url = "https://huggingface.co/VeyDlin/u2net_clothing_segmentation/resolve/main/u2net_clothing_segmentation.pth"

    def __init__(self, normalize_mean = 0.5, normalize_std = 0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [NormalizeImage(normalize_mean, normalize_std)]
        self.transform_rgb = transforms.Compose(transforms_list)

        self.palette = [0] * 3 + [255] * 9

        self.u2net = self.__load_u2net()
        

    def segmentation(self, image):
        image_tensor = self.transform_rgb(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        output_tensor = self.u2net(image_tensor.to(self.device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)

        output_arr = output_tensor.cpu().numpy()

        output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
        output_img.putpalette(self.palette)

        return output_img


    def __load_u2net(self):
        filename = urlparse(self.checkpoint_url).path.split('/')[-1]
        checkpoint_path = os.path.join("models", filename)

        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            response = requests.get(self.checkpoint_url)
            if response.status_code == 200:
                with open(checkpoint_path, "wb") as file:
                    file.write(response.content)

        model_state_dict = torch.load(checkpoint_path, map_location=torch.device(self.device))
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        net = U2NET(in_ch=3, out_ch=4)
        net.load_state_dict(new_state_dict)
        net = net.to("cuda")
        net = net.eval()

        return net



class NormalizeImage(object):
    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"