import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from tensorflow.keras.layers import Dense,Conv2D,GlobalAvgPool2D,Input
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras import callbacks,optimizers
import numpy as np
from google.colab import drive 

drive.mount('/content/drive')

cd /content/drive/MyDrive/Final_project

for i in os.listdir("img"):
  print(i)

for i in os.listdir("train"):
  print(i,len(os.listdir("train/"+i)))

for i in os.listdir("test"):
  print(i,len(os.listdir("test/"+i)))

def img_Data(dir_path,target_size,batch,class_lst,preprocessing):
  if preprocessing:
    gen_object = ImageDataGenerator(preprocessing_function=preprocessing)
  else:
     gen_object = ImageDataGenerator()

  return(gen_object.flow_from_directory(dir_path,
                                        target_size=target_size,
                                        batch_size=batch,
                                        class_mode='sparse',
                                        classes=class_lst,
                                        shuffle=True))


train_data_gen = img_Data("train",(224,224),50,['Ocimum Africanum','Ocimum Gratissimum','Ocimum Tenuiflorum'],preprocess_input)
valid_data_gen = img_Data("test",(224,224),50,['Ocimum Africanum','Ocimum Gratissimum','Ocimum Tenuiflorum'],preprocess_input)

base_model=tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=(224,224,3),
    alpha=1.0,
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)

base_model.trainable= False 

model = tf.keras.models.Sequential()
model.add(base_model)
model.add(GlobalAvgPool2D())
model.add(Dense(1024,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

elst = callbacks.EarlyStopping(monitor='val_loss',patience=5,mode='min')
save_ck = callbacks.ModelCheckpoint('.mdl_wt.hdf5',save_best_only=True,monitor='val_loss',mode='min')

history=model.fit(train_data_gen,batch_size=50,validation_data=valid_data_gen,callbacks=[elst,save_ck],epochs=10)

pred1=np.argmax(model.predict(valid_data_gen,steps=6,verbose=1),axis=1)

pred1

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

print('Confusion Matrix')
print(confusion_matrix(valid_data_gen.classes,pred1))
print('Classification Report')
target_names = ['Ocimum Africanum',
 'Ocimum Gratissimum',
 'Ocimum Tenuiflorum']
print(classification_report(valid_data_gen.classes, pred1, target_names=target_names))
