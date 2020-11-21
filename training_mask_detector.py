import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


directory_for_dataset = r'C:\Users\Jeeva Nagarajan\PycharmProjects\project 10(face mask detection)\dataset'
folders_in_dataset = ['with_mask', 'without_mask']


print('[INFO] Loading Images...')


datas = []
labels = []


for folder in folders_in_dataset:
    path = os.path.join(directory_for_dataset, folder)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        x = load_img(img_path, target_size = (256, 256))
        x = img_to_array(x)
        image = preprocess_input(x)

        datas.append(image)
        labels.append(folder)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


datas = np.array(datas, dtype = 'float32')
labels = np.array(labels)


(x_train, x_test, y_train, y_test) = train_test_split(datas, labels, test_size = 0.25, stratify = labels, random_state = 42)


learning_rate = 0.0001
epochs = 20
batch_size = 32


datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')


base_model = MobileNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (256, 256, 3)))


head_model = base_model.output
head_model = AveragePooling2D(pool_size = (7, 7))(head_model)
head_model = Flatten(name = 'flatten')(head_model)
head_model = Dense(128, activation = 'relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation = 'softmax')(head_model)


model = Model(inputs = base_model.input, outputs = head_model)


for layer in base_model.layers:    
    layer.trainable = False


print('[INFO] Compiling Our Facemask_Detector Model...')


model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = learning_rate, decay = learning_rate / epochs), metrics = ['accuracy'])


print('[INFO] Training Head...')


head = model.fit(   
    datagen.flow(x_train, y_train, batch_size = batch_size),
    steps_per_epoch = len(x_train) // batch_size,
    validation_data = (x_test, y_test),
    validation_steps = len(x_test) // batch_size,
    epochs = epochs)


print('[INFO] Evaluating Network...')


prediction = model.predict(x_test, batch_size = batch_size)
prediction = np.argmax(prediction, axis = 1)


print(classification_report(y_test.argmax(axis = 1), prediction, target_names = lb.classes_))


print('[INFO] Saving the Facemask Detector Model...')


model.save('Facemask_Detector.model', save_format = 'h5')


plt.style.use('fivethirtyeight')
plt.figure()
plt.plot(np.arange(0, epochs), head.history['loss'], label = 'train_loss', marker = 'o' )
plt.plot(np.arange(0, epochs), head.history['val_loss'], label = 'val_loss', marker = 'o')
plt.plot(np.arange(0, epochs), head.history['accuracy'], label = 'train_acc', marker = 'o')
plt.plot(np.arange(0, epochs), head.history['val_accuracy'], label = 'val_acc', marker = 'o')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.tight_layout()
plt.savefig('model.png')



