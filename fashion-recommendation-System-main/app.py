import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle


model = ResNet50(weights='imagenet',include_top= False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())

def feature_extraction(img_path,model):
    

    return normalized_result

filenames =[]
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# print(len(filenames))

feature_list = []

for file in filenames:
    feature_list.append(feature_extraction(file,model))

print(np.array(feature_list).shape)

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

