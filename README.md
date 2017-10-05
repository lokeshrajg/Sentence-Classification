# Sentence-Classification
# Character Level CNN using Tensorflow and Keras

## Infrastructure, Technologies and Tools used
1. Windows 10
2. Python 3.5.2
3. TensorFlow 1.2.1 - CPU version
4. Keras 2.0.6

And TensorFlow and Keras Python dependencies

## Folders and Files
data ------------------------> contains provided input files <br />
xtest_obfuscated.txt <br />
xtrain_obfuscated.txt <br />
ytrain.txt <br />

instruction -----------------> contains provided instruction file <br />
instructions.txt <br />

results ---------------------> contains model's captured results <br />
model-accuracy.png ------> plot of train and test model accuracy vs epoch<br />
model-loss.png ----------> plot of train and test model loss vs epoch <br />
ytest.txt ---------------> prediction result file for provided xtest_obfuscated.txt data <br />

character_cnn.py ------------> CNN implementation <br />
config.py -------------------> Models configurations are set <br />
data_utils.py ---------------> Loading and preprocessing data <br />


## How to use

Run character_cnn.py as below:
```sh
$ python character_cnn.py
```

## Train and Validation Split

Training on 26010 samples, validation on 6503 samples

## Accuracy and Loss

Epoch 1/10 <br />
26010/26010 [=============] - 1165s - loss: 2.2681 - acc: 0.1805 - val_loss: 2.0271 - val_acc: 0.2788 <br />
Epoch 2/10 <br />
26010/26010 [=============] - 1185s - loss: 1.8671 - acc: 0.3453 - val_loss: 1.5543 - val_acc: 0.4436 <br />
Epoch 3/10 <br />
26010/26010 [=============] - 1176s - loss: 1.5235 - acc: 0.4591 - val_loss: 1.4258 - val_acc: 0.4921 <br />
Epoch 4/10 <br />
26010/26010 [=============] - 1125s - loss: 1.3150 - acc: 0.5328 - val_loss: 1.1662 - val_acc: 0.5831 <br />
Epoch 5/10 <br />
26010/26010 [=============] - 1163s - loss: 1.1655 - acc: 0.5811 - val_loss: 1.1149 - val_acc: 0.6005 <br />
Epoch 6/10 <br />
26010/26010 [=============] - 1176s - loss: 1.0428 - acc: 0.6275 - val_loss: 1.0352 - val_acc: 0.6312 <br />
Epoch 7/10 <br />
26010/26010 [=============] - 1188s - loss: 0.9147 - acc: 0.6756 - val_loss: 0.9679 - val_acc: 0.6608 <br />
Epoch 8/10 <br />
26010/26010 [=============] - 1180s - loss: 0.8041 - acc: 0.7220 - val_loss: 0.9773 - val_acc: 0.6695 <br />
Epoch 9/10 <br />
26010/26010 [=============] - 1171s - loss: 0.6996 - acc: 0.7590 - val_loss: 0.9219 - val_acc: 0.6918 <br />
Epoch 10/10 <br />
26010/26010 [=============] - 1181s - loss: 0.6047 - acc: 0.7923 - val_loss: 0.9483 - val_acc: 0.7015 <br />

## Expected accuracy on the test set

Accuracy - (70.15 +/- 5) %

## Author

Lokesh Kotimugalur Ganesha Raja
