from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

print("Loading the configurations...")

exec(open("config.py").read())
conv_layers = config.model.conv_layers
fully_layers = config.model.fully_connected_layers
l0 = config.l0
alphabet_size = config.alphabet_size
embedding_size = config.model.embedding_size
num_of_classes = config.num_of_classes
th = config.model.th
p = config.dropout_p

print("Configuration loaded")

print("Building the model...")

# building the model

# Input layer
inputs = Input(shape=(l0,), name='sent_input', dtype='int64')

# Embedding layer
x = Embedding(alphabet_size + 1, embedding_size, input_length=l0)(inputs)

# Convolution layers
for cl in conv_layers:
    x = Convolution1D(cl[0], cl[1])(x)
    x = ThresholdedReLU(th)(x)
    if not cl[2] is None:
        x = MaxPooling1D(cl[2])(x)

x = Flatten()(x)

# Fully connected layers
for fl in fully_layers:
    x = Dense(fl)(x)
    x = ThresholdedReLU(th)(x)
    x = Dropout(0.5)(x)

predictions = Dense(num_of_classes, activation='softmax')(x)

# model
model = Model(input=inputs, output=predictions)

optimizer = Adam()

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("Model built")

print("Loading the data sets...")

# Data operations
from data_utils import Data

train_data = Data(data_source=config.train_data_source,
                    label_source=config.train_label_source,
                    alphabet=config.alphabet,
                    l0=config.l0,
                    batch_size=0,
                    no_of_classes=config.num_of_classes)

train_data.loadData()

train_instances, train_labels = train_data.getAllData()

test_data = Data(data_source=config.test_data_source,
                    label_source=config.test_label_source,
                    alphabet=config.alphabet,
                    l0=config.l0,
                    batch_size=0,
                    no_of_classes=config.num_of_classes)

test_data.loadTestData()

X_test = test_data.getTestAllData()

seed = 7
np.random.seed(seed)
X_train, X_val, y_train, y_val = train_test_split(train_instances, train_labels,
                                                  test_size=0.2, random_state=seed)

print("Data loaded")

# Model Training
print("Model training...")

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

history = model.fit(X_train, y_train, epochs=config.training.epochs, batch_size=config.batch_size,
                    validation_data=(X_val, y_val), callbacks=[monitor])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# test data prediction
f = open('results/ytest.txt',"w+")
pred = model.predict(X_test)
for predicted in pred:
    f.write(str(np.argmax(predicted))+"\n")
f.close()
