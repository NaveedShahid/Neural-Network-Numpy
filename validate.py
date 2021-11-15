import pickle
import time
import numpy as np
import pandas as pd
from model.utils import Dataset
from tensorflow import one_hot
output = []
feature_set = []

validate_file = "validate.csv"
data = pd.read_csv(validate_file,header=None,index_col=None)
y_validate = data.iloc[:,0].values
x_validate = data.iloc[:,1:].values
x_validate = [img.reshape(3,32,32) for img in x_validate]
x_validate = np.asarray(x_validate)
y_validate = np.asarray(one_hot(y_validate,10))
print(x_validate.shape,y_validate.shape)

# get network
model_file = open("network_model.pkl", "rb")
net = pickle.load(model_file)
model_file.close()

# validate
batch_size=32

correct = 0
overall = (len(y_validate)//batch_size)*batch_size

validate = Dataset(X=x_validate, y=y_validate, batch_size=batch_size)
start = time.time()
for batch, (x_mini, y_mini) in enumerate(validate):
    print("\n====== Batch", batch, "======")            

    out = net.predict(x_mini)
    for idx, row in enumerate(out):
        if (y_mini[idx][np.argmax(row)] == 1):
            correct += 1

print('%f%%' % ((correct / overall) * 100))