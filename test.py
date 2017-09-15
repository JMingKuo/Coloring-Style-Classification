import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

class Classifier():
    '''simple image classifier'''
    
    transform = transforms.ToTensor()

    def __init__(self, weight=None):
        self.resize = (228, 228)
        self.input_size = (224, 224)
        self.num_class = 6
        self.model = self.load_model(weight)

    def load_model(self, weight):
        '''load trained model from checkpoint'''
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 6)
        model.load_state_dict(torch.load(weight)['state_dict'])
        return model.cuda()

    def center_crop(self, input):
        return cv2.resize(input, self.resize)[2:226, 2:226].transpose(2, 0, 1)

    def inference(self, inputs, batch_size=1):
        '''do inference'''
        img = self.center_crop(inputs)
        img_input = self.transform(img).unsqueeze(0)
        self.model.eval() # close BN and Dropout
        output = self.model(Variable(img_input).cuda())
        _, preds = torch.max(output.data, 1)
        return preds.cpu().numpy()

Classifier = Classifier(weight=r'model_best.pth.tar')

data_dir = r'D:\workspace\dataset\pixiv_coloring\imgdata\val'
confusion_matrix = np.zeros([6, 6], dtype=np.float32)
class_cnt = np.zeros([6, 6], dtype=np.float32)

for label in os.listdir(data_dir):
    for img in os.listdir(os.path.join(data_dir, label)):
        img_raw = cv2.imread(os.path.join(data_dir, label, img))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
        pred = int(Classifier.inference(inputs=img_rgb))
        confusion_matrix[int(label), pred] += 1
        class_cnt[int(label), :] += 1

np.set_printoptions(precision=4)

print(confusion_matrix/class_cnt)
print('Average accuracy :', np.trace(confusion_matrix/class_cnt)/6)

# show heatmap
categories = ['LineArt','Pastose','Digital','Black/White','Celluloid','WaterColor']
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in categories],
                  columns = [i for i in categories])
fig = plt.figure(figsize = (10,7))
ax = plt.axes()
heatmap_cm = sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
heatmap_cm.set_yticklabels(heatmap_cm.get_yticklabels(), rotation = 0) # fix label direction
ax.tick_params(labelbottom='off',labeltop='on')
plt.show()