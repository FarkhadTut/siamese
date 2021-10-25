import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize


device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda'



data_train = pd.read_csv('dataset/fashion-mnist_train.csv')
X_full = data_train.iloc[:,1:]
Y_full = data_train.iloc[:,:1]

x_train, x_test, y_train, y_test = train_test_split(X_full, Y_full, test_size = 0.3)


x_train = x_train.values.reshape(-1, 28, 28, 1).astype('float32')/255
x_test = x_test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = y_train.values.astype('int')
y_test = y_test.values.astype('int')



train_groups = [x_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]
test_groups = [x_test[np.where(y_test==i)[0]] for i in np.unique(y_train)] 



class Dataloader():
	def __init__(self, x_train, y_train, batch_halfsize):
		self.x_train = x_train
		self.y_train = y_train
		self.count = 0
		self.batch_halfsize = batch_halfsize
		self.groups = np.unique(y_train)

	def batch_available(self):
		return self.count*self.batch_halfsize < len(self.x_train)

	def reset(self):
		self.count = 0

	def get_batch(self):
		a_img, b_img, l = [],[],[]
		for match_group in [True, False]:
			idx = np.random.choice(list(range(len(self.y_train))), size=self.batch_halfsize)
			if match_group:
				a = self.x_train[idx]
				labels = self.y_train[idx]
				b = np.zeros_like(a)
				labels_unique = np.unique(labels)
				for i in labels_unique:
					idx_new = np.where(self.y_train == i)[0]
					quan_i = len(labels[labels==i])
					idx_b = np.where(labels==i)[0]
					idx_new = np.random.choice(idx_new, size=quan_i)
					b[idx_b] = self.x_train[idx_new]
					
				
				l = [1] * self.batch_halfsize
				a_img = a
				b_img = b
				
			else:
				labels = self.y_train[idx]
				a = self.x_train[idx]
				b = np.zeros_like(a)
				labels_unique = np.unique(labels)
				suka = []
				for i in labels_unique:
					quan_i = len(labels[labels==i])
					idx_all = np.where(self.y_train != i)[0]
					idx_new = np.random.choice(idx_all, size=quan_i)
					idx_b = np.where(labels==i)[0]
					b[idx_b] = self.x_train[idx_new]

				l = np.concatenate((l, [0]*self.batch_halfsize), axis=0)
		
		a_img = np.concatenate((a_img, a), axis=0)
		b_img = np.concatenate((b_img, b), axis=0)
		
		
		a_img = np.reshape(a_img, (a_img.shape[0],a_img.shape[-1], a_img.shape[1],a_img.shape[2]))
		b_img = np.reshape(b_img, (b_img.shape[0],b_img.shape[-1], b_img.shape[1],b_img.shape[2]))

		a_img, b_img, l = shuffle(a_img, b_img, l)

		a_img = torch.from_numpy(a_img).to(device)
		b_img = torch.from_numpy(b_img).to(device)
		l = torch.from_numpy(l).to(device)



		self.count += 1

		return a_img, b_img, l








class Model(nn.Module):
	def __init__(self, batch_halfsize):
		self.batch_size = 2*batch_halfsize
		super().__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv2d(1,64,5, stride=2, padding=1),
			nn.MaxPool2d(2,2),
			nn.Conv2d(64,128,5, stride=2, padding=1),
			nn.MaxPool2d(2,2),
			nn.Conv2d(128,256,3, stride=2, padding=1),
		)

		self.fcs = nn.Sequential(
			nn.Linear(in_features=256, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=32),
			nn.ReLU(),
			nn.Linear(in_features=32, out_features=2),
			# nn.Softmax(dim=1)
		)
		
	def forward(self, t, z_vector=False):
		if z_vector:
			t = self.fcs(t)

		else:
			t = t.to(device)
			t = self.conv_layers(t)
			t = t.flatten(start_dim=1)

		return t 



epochs = 20
batch_halfsize = 64
lr = 0.0001


testloader = Dataloader(x_test, y_test, batch_halfsize)

dataloader = Dataloader(x_train, y_train, batch_halfsize)
model = Model(batch_halfsize).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
cost_function = nn.CrossEntropyLoss()



loss_per_epoch = []
accuracy_per_epoch = []
loss_per_epoch_val = []
accuracy_per_epoch_val = []

for epoch in range(epochs):
	loss_avg = 0
	accuracy_avg = 0
	loss_avg_val = 0
	accuracy_avg_val= 0

	while(dataloader.batch_available()):
		a, b, l = dataloader.get_batch()
		a_feat = model(a)
		b_feat = model(b)
		z = abs(a_feat - b_feat)
		z = F.normalize(z)
		if torch.max(z).item() > 1:
			print('Z vector contains values larger than 1')
			print('Exited...')
			sys.exit()

		output = model(z, z_vector=True)
		optimizer.zero_grad()
		loss = cost_function(output, l)
		loss.backward()
		optimizer.step()


		loss_avg += loss.item()


		_, pred = torch.max(output.data, 1)
		pred_correct = len(list(filter(lambda x: x == True, l.eq(pred).type(torch.bool))))
		accuracy = 100*pred_correct/(2*batch_halfsize)
		accuracy_avg += accuracy
		
######################  Cross Validation ################################
		
		if not testloader.batch_available():
			testloader.reset()

		model.eval()
		a, b, l = testloader.get_batch()
		a_feat = model(a)
		b_feat = model(b)
		z = abs(a_feat - b_feat)
		z = F.normalize(z)

		output_val = model(z, z_vector=True)
		loss_val = cost_function(output_val, l)
		loss_avg_val += loss_val.item()

		_, pred_val = torch.max(output_val.data, 1)
		pred_correct_val = len(list(filter(lambda x: x == True, l.eq(pred_val).type(torch.bool))))
		accuracy_val = 100*pred_correct_val/(2*batch_halfsize)
		accuracy_avg_val += accuracy_val

		model.train()
		
########################################################################
		print(f'Epoch: {epoch}, Iter: {dataloader.count}/{len(x_train)/batch_halfsize}  Acc: {int(accuracy*10)/10},    Loss: {int(loss.item()*10000)/10000}', end='\r')
		# print(f'              , Iter: {dataloader.count}/{len(x_train)/batch_halfsize}  Acc: {int(accuracy_val*10)/10},    Loss: {int(loss_val.item()*10000)/10000}', end='\r')

	loss_avg = loss_avg/dataloader.count
	accuracy_avg = accuracy_avg/dataloader.count
	loss_per_epoch.append(loss_avg)
	accuracy_per_epoch.append(accuracy_avg)


	loss_avg_val = loss_avg_val/dataloader.count
	accuracy_avg_val = accuracy_avg_val/dataloader.count
	loss_per_epoch_val.append(loss_avg_val)
	accuracy_per_epoch_val.append(accuracy_avg_val)
	
	dataloader.reset()



	print(f'Epoch: {epoch},  Acc: {int(accuracy_avg*10)/10},    Loss: {int(loss_avg*10000)/10000}')
	print(f'Valid: {epoch},  Acc: {int(accuracy_avg_val*10)/10},    Loss: {int(loss_avg_val*10000)/10000}')
	print('-------------------------------------------------------------------------------------')











