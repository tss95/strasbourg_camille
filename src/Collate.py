import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
from time import time
from PIL import ImageOps
#import transforms from pytorch
from torchvision import transforms

class CollateFn:
    def __init__(self, spatial_flip = True, random_colormap = True, p_spatial_flip = 0.5, image_size=(256, 256)):
        self.image_size = image_size  # image_size should be (height, width)
        self.spatial_flip = spatial_flip
        self.p_spatial_flip = p_spatial_flip
        self.random_colormap = random_colormap
        if self.random_colormap:
            self.RandomColormap = ApplyRandomColormapBatch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __process_batch(self, batch):
        eb1, eb2, psd1, psd2, spec1, spec2, stalta1, stalta2, rms1, rms2 = [], [], [], [], [], [], [], [], [], []
        for item in batch:
            eb1.append(item['eb1'])
            eb2.append(item['eb2'])
            psd1.append(item['psd1'])
            psd2.append(item['psd2'])
            spec1.append(item['spec1'])
            spec2.append(item['spec2'])
            stalta1.append(item['stalta1'])
            stalta2.append(item['stalta2'])
        eb1 = torch.stack(eb1).to(self.device)
        eb2 = torch.stack(eb2).to(self.device)
        psd1 = torch.stack(psd1).to(self.device)
        psd2 = torch.stack(psd2).to(self.device)
        spec1 = torch.stack(spec1).to(self.device)
        spec2 = torch.stack(spec2).to(self.device)
        stalta1 = torch.stack(stalta1).to(self.device)
        stalta2 = torch.stack(stalta2).to(self.device)
        
        return eb1, eb2, psd1, psd2, spec1, spec2, stalta1, stalta2
    
    def __call__(self, batch):
        start = time()   
        eb1, eb2, psd1, psd2, spec1, spec2, stalta1, stalta2 = self.__process_batch(batch)
        
        eb_height = psd_height = spec_height = stalta_height = self.image_size[0] // 4

        
        eb1 = F.interpolate(eb1.unsqueeze(1), size=(eb_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        eb2 = F.interpolate(eb2.unsqueeze(1), size=(eb_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        psd1 = F.interpolate(psd1.unsqueeze(1), size=(psd_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        psd2 = F.interpolate(psd2.unsqueeze(1), size=(psd_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        spec1 = F.interpolate(spec1.unsqueeze(1), size=(spec_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        spec2 = F.interpolate(spec2.unsqueeze(1), size=(spec_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        stalta1 = F.interpolate(stalta1.unsqueeze(1), size=(stalta_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        stalta2 = F.interpolate(stalta2.unsqueeze(1), size=(stalta_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(1)
        
        
        if self.spatial_flip and random.random() > 1-self.spatial_flip:
            # TODO Verify correct flipping dimensions
            img_index = int(random.random())
            flip_dim = [1]
            
            if img_index == 0:
                eb1, psd1, stalta1 = [torch.flip(img, flip_dim) for img in [eb1, psd1, stalta1]]
            else:
                eb2, psd2, stalta2 = [torch.flip(img, flip_dim) for img in [eb2, psd2, stalta2]]
                
                        
        img1_batch = torch.cat([eb1, psd1, spec1, stalta1], dim=1)
        img2_batch = torch.cat([eb2, psd2, spec2, stalta2], dim=1)

        color_start = time()
        if self.random_colormap:
            img1_batch = self.RandomColormap(img1_batch)
            img2_batch = self.RandomColormap(img2_batch)
        print(f"Time taken for color transform: {time() - color_start}")
        
        print(f"Time taken for collate function: {time() - start}")
        
        return img1_batch, img2_batch
    
class ApplyRandomColormapBatch:
    def __init__(self):
        self.colormaps = [
            ('black', 'white'),
            ('red', 'cyan'),
            ('blue', 'yellow'),
            ('green', 'magenta'),
            ('orange', 'purple'),
            ('brown', 'pink'),
            ('darkblue', 'lightgreen'),
            ('darkred', 'lightblue'),
            ('darkgreen', 'lightpink'),
            ('darkorange', 'plum'),  # replaced 'lightpurple' with 'plum'
        ]
        self.last_used = random.choice(self.colormaps)

    def __call__(self, batch):
        # Select one random colormap for the entire batch
        available_colormaps = [cm for cm in self.colormaps if cm != self.last_used]
        
        # Select one random colormap from the filtered list
        if available_colormaps:
            colormap = random.choice(available_colormaps)
        else:
            # If all colormaps are similar to last_used or it's the only one, reset the list or choose any
            colormap = random.choice(self.colormaps)
        
        # Apply the colormap to each sample in the batch
        batch_list = []
        for img in batch:
            # Convert tensor to PIL Image
            img = transforms.ToPILImage()(img)
            
            # Apply colormap
            img = ImageOps.colorize(img, colormap[0], colormap[1])
            
            # Convert PIL Image back to tensor
            img = transforms.ToTensor()(img)
            
            batch_list.append(img)
        
        batch = torch.stack(batch_list)
        self.last_used = colormap
        
        return batch

# DataLoader usage remains the same
