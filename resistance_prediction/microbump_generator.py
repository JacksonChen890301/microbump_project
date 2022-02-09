# +
import torch
import numpy as np
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


class CT_scan(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.x_train, self.y_train = np.load('TSMC/bump_x_train.npy'), np.load('TSMC/bump_y_train.npy')
        self.x_val, self.y_val = np.load('TSMC/bump_x_val.npy'), np.load('TSMC/bump_y_val.npy')

    def __getitem__(self, index):
        if self.train:
            inputs, target = self.x_train[index, :, :, :, :], self.y_train[index]
        else:
            inputs, target = self.x_val[index, :, :, :, :], self.y_val[index]
        inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.cuda.FloatTensor)
        return inputs, target

    def __len__(self):
        if self.train:
            return len(self.x_train)
        else:
            return len(self.x_val)
        
        
class bump_test(Dataset):
    def __init__(self):
        self.x, self.y = np.load('TSMC/bump_x_test.npy'), np.load('TSMC/bump_y_test.npy')

    def __getitem__(self, index):
        inputs, target = self.x[index, :, :, :, :], self.y[index]
        inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.cuda.FloatTensor)
        return inputs, target

    def __len__(self):
        return len(self.x)
    
    
class Anamoly_Detection(Dataset):
    def __init__(self, train=True):
        self.train = train
        initial = np.load('TSMC/ansys_result/N17_initial.npy')
        reflow = np.load('TSMC/ansys_result/N17_reflow.npy')
        image = np.load('TSMC/N17_ini_raw.npy')
        image = normalize(image)
        
        deviation_list = []
        for row in range(initial.shape[0]):
            for column in range(initial.shape[1]):
                deviation = (reflow[row, column]-initial[row, column])/initial[row, column]
                if math.isnan(deviation):
                    continue
                deviation_list.append(deviation*100)
        
#         sort = np.argsort(np.array(deviation_list))
#         image = image[sort, :, :, :, :]
        
        deviation_list = scaler.fit_transform(np.array(deviation_list).reshape(-1, 1))
        rnd = np.arange(image.shape[0])
        np.random.seed(11)
        np.random.shuffle(rnd)
        image = image[rnd, :, :, :, :]
        deviation_list = deviation_list[rnd]
        self.normal = image[:290, :, :, :, :]
        self.anamoly = image[290:, :, :, :, :]
        self.norm_target = deviation_list[:290, :]
        self.ana_target = deviation_list[290:, :]
                
    def __getitem__(self, index):
        if self.train:
            inputs, target = self.normal[index, :, :, :, :], self.norm_target[index]
        else:
            inputs, target = self.anamoly[index, :, :, :, :], self.ana_target[index]
        inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.cuda.FloatTensor)
        return inputs, target
    
    def __len__(self):
        if self.train:
            return len(self.normal)
        else:
            return len(self.anamoly)
        
        
class Anamoly_Test(Dataset):
    def __init__(self):
#         self.train = train
        initial = np.load('TSMC/ansys_result/N12_initial.npy')
        reflow = np.load('TSMC/ansys_result/N12_reflow.npy')
        image = np.load('TSMC/N12_ini_raw.npy')
        image = np.concatenate((image[0:10, :, :, :, :],image[20:90, :, :, :, :],
                                image[100:110, :, :, :, :],image[120:240, :, :, :, :],
                                image[250:270, :, :, :, :]), axis=0)
        image = normalize(image)
        
        deviation_list = []
        for row in range(reflow.shape[0]):
            for column in range(reflow.shape[1]):
                deviation = (reflow[row, column]-initial[row, column])/initial[row, column]
                if math.isnan(deviation):
                    continue
                deviation_list.append(deviation*100)
        
#         sort = np.argsort(np.array(deviation_list))
#         image = image[sort, :, :, :, :]
        deviation_list = scaler.transform(np.array(deviation_list).reshape(-1, 1))
        self.x = image
#         self.y = np.sort(np.array(deviation_list))
        self.y = deviation_list
                
    def __getitem__(self, index):
        inputs, target = self.x[index, :, :, :, :], self.y[index, :]
        inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.cuda.FloatTensor)
        return inputs, target
    
    def __len__(self):
        return len(self.x)

        
def normalize(volume):
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

test = CT_scan()
