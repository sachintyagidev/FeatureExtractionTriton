import matplotlib.pyplot as plt
import seaborn as sns, os
from sklearn.metrics import confusion_matrix
from model import *
from data import prepare_dataloader
import pandas as pd
import numpy as np, pickle
from PIL import Image
from scipy.special import softmax
import time
import torch
from torchvision import transforms

import onnx
import onnxruntime
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=1

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

triton_dir  = './model/'

def pytorch_infer(model_name, channels_last, device, distributed, local_rank, out_weight, val_imgs):
    model = build_model(model_name, channels_last, device, distributed, local_rank)
    model.load_state_dict(torch.load(out_weight))

    input_batch_gpu = torch.from_numpy(val_imgs).to(device)
    with torch.no_grad():
        preds = np.array(model(input_batch_gpu).cpu())
    
    test_dl = prepare_dataloader()
    data_pytorch_model = eval(model, test_dl)
    return data_pytorch_model['gt'], data_pytorch_model['pred']

def onnx_infer(val_imgs, model_name, batch_size):
    session =onnxruntime.InferenceSession(triton_dir + model_name + '_onnx/1/model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    data0 = json.dumps({'data': val_imgs.tolist()})
    data0 = np.array(json.loads(data0)['data']).astype('float32')
    preds = session.run([output_name], {input_name: data0})

    dataloader, gt = prepare_dataloader()
    pred = None
    
    elapsed_time = []
    for i, val_imgs in enumerate(dataloader):
        data = json.dumps({'data': val_imgs.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        start_time = time.time()
        result = session.run([output_name], {input_name: data})
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0 and i != 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
        result = np.argmax(softmax(result[0], axis=1), axis=1)
        pred = np.concatenate((pred, result), axis=0) if pred is not None else result
    pred = pred[:len(gt)]
    print('Throughput: {:.0f} images/s'.format(i * batch_size / elapsed_time.sum()))
    return gt, pred