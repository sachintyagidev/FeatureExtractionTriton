import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_name, channels_last, device, distributed, local_rank):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs,512),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                            nn.Linear(512,4))
    
    if channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
        
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    return model