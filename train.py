from model.Network import Network
from model.Optimizers import Adam
from model.Losses import CrossEntropy
from model.FlattenLayer import FlattenLayer
from model.InputLayer import InputLayer
from model.ActivationLayer import ReluLayer
from model.SoftmaxLayer import SoftmaxLayer
from model.DropoutLayer import DropoutLayer
from model.CONVLayer import CONVLayer
from model.MaxPoolLayer import MaxPoolLayer
from model.FCLayer import FCLayer
import pandas as pd
from tensorflow import one_hot
import numpy as np
import pickle

## Add data here
try:
    x_train = pickle.load(open('xtrains.pkl','rb'))
    y_train = pickle.load(open('ytrains.pkl','rb'))
except:
    file= 'train.csv'
    data = pd.read_csv(file,header=None,index_col=None)[:100]
    y_train = data.iloc[:,0].values
    x_train = data.iloc[:,1:].values
    x_train = [img.reshape(3,32,32) for img in x_train]
    x_train = np.asarray(x_train)
    y_train = np.asarray(one_hot(y_train,10))
    print(x_train.shape,y_train.shape)
        # pickle.dump(x_train,open('xtrain.pkl','wb'))
        # pickle.dump(y_train,open('ytrain.pkl','wb'))

model = 'n'
while model != 'c' and model != 'n':
    # Add code to ask whether to upload old network
    print("Continue training from existing model: press c")
    print("Start training a new model: press n")
    model = input()

if model == 'n':
    net = Network()
    net.add(InputLayer((3, 32, 32)))
    net.add(CONVLayer(32, [3, 3], [1, 1]))
    net.add(ReluLayer())
    net.add(MaxPoolLayer())
    net.add(CONVLayer(64, [3, 3], [1, 1]))
    net.add(ReluLayer())
    net.add(MaxPoolLayer())
    net.add(FlattenLayer())
    net.add(DropoutLayer(0.25))
    net.add(FCLayer(128))
    net.add(ReluLayer())
    net.add(DropoutLayer(0.2))
    net.add(FCLayer(10))
    net.add(SoftmaxLayer())

elif model == 'c':
    # get network
    model_file = open("network_model", "rb")
    net = pickle.load(model_file)
    model_file.close()

net.use(loss_fn=CrossEntropy(), opt=Adam(learning_rate=0.0005))

net.fit(x=x_train, y=y_train, epochs=100, batch_size=32)

pickle.dump(net,open('network_model.pkl','wb'))

    # folder = 'train/'
    # xy_list = []
    # for file in os.listdir(folder[:4000]):
    #     img = cv2.imread(os.path.join(folder,file))
    #     labels = pd.read_csv('trainLabels.csv',index_col=None)
    #     label = labels[labels.id==int(file.split('.')[0])].label.values
    #     # label = one_hot(label,10)
    #     # print(label)
    #     xy_list.append((img.transpose(2,0,1),label))
    # le=LabelEncoder()
    # xy = np.asarray(xy_list)
    # x_train = xy[:,0]/255.0
    # y_train = xy[:,1]
    # x_train = list(x_train)
    # x_train = np.asarray(x_train)
    # le.fit(list(y_train))
    # y_train=le.transform(list(y_train))
    # y_train = one_hot(y_train,10)
    # y_train = np.asarray(y_train)

    # print(y_train)
    #     # x_train = [img.reshape(3,32,32) for img in x_train]
    #     # x_train = np.asarray(x_train)
    #     # y_train = np.asarray(one_hot(y_train,10))
    #     # print(x_train.shape,y_train.shape)
    # pickle.dump(x_train,open('xtrain.pkl','wb'))
    # pickle.dump(y_train,open('ytrain.pkl','wb'))