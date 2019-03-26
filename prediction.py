from __future__ import print_function
import numpy as np
import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Initializing=- 0d4rseyt
batch_size = 128
num_classes = 10
epochs = 100
prediction = 0
pred = 0
data_augmentation = False

# input image dimensions
img_rows, img_cols = 128, 128

# The following 4 list are organized like this x_train = Training Images, y_train = Corresponding Labels of the train images.
# Same goes for x_test,y_test

x_train = []
y_train = []
x_test = []
y_test = []
for i in range(0, 10):
    files = os.listdir('Dataset/train_set/' + str(i) + '/')  # Reads each images of folders one by one
    for file in files:
        filename = 'Dataset/train_set/' + str(i) + '/' + file
        img = cv2.imread(filename)
        img = cv2.resize(img, (128, 128))
        x_train.append(img)
        y_train.append(i)
for i in range(0, 10):
    files = os.listdir('Dataset/test_set/' + str(i) + '/')  # Reads each images of folders one by one

    for file in files:
        filename = 'Dataset/test_set/' + str(i) + '/' + file
        img = cv2.imread(filename)
        img = cv2.resize(img, (128, 128))
        x_test.append(img)
        y_test.append(i)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#..........................................
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#..........................................
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#..........................................
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
#......................................
model.load_weights('Weight.h5', by_name=True)


#...............................Prediction Part...................................
import cv2
from gtts import gTTS
cap = cv2.VideoCapture(0)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    #prediction = 0
    #pred = 0
    #arr = [[0,0,0,0,0,0,0,0,0,0]]
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Saves image of the current frame in jpg file
    name = './data/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imshow('frame', frame)
    #cv2.imwrite(str(currentFrame)+".jpg",currentFrame)
    if currentFrame%240==10:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Frame Cropping

        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        kernel = np.ones((5, 5))
        dilation = cv2.dilate(mask2, kernel, iterations=4)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        filtered = cv2.GaussianBlur(erosion, (5, 5), 100)
        ret, thresh = cv2.threshold(filtered, 127, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow('Dilation',frame)
        cropped = frame[y:y + h, x:x + w]
        img1 = cv2.resize(cropped, (128, 128))
        ##### Show final image ########
        cv2.imwrite('Frame%d.jpg'% currentFrame,img1)
        img2 = img1.reshape(1, 128, 128, 3)
        print(model.predict(img2))
        arr = model.predict(img2)
        for i in range(0, 10):
            if arr[0][i] == 1:
                prediction = i
                break
        print(prediction)  # Prediction result print

        # Translating prediction to Bangla using googletrans API
        pred = str(prediction)

        text = pred
        targetLanguage = 'bn'
        tts = gTTS(text, targetLanguage)
        tts.save("9.mp3")
        os.system('9.mp3')

    # To stop duplicate images
    currentFrame += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



#Visualizations
# The ROC curve part
'''import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
print(y_test.shape)
print(Y_pred.shape)
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(10):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 10

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(["red", "orange", "yellow", "green", "blue", "purple", "black", 'aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(["red", "orange", "yellow", "green", "blue", "purple", "black", 'aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zoom view of Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Accuracy and Loss curve
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=30,
                                 shear_range=0.2,
                                 horizontal_flip=True)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
history_callback = model.fit_generator(datagen.flow(x_train, y_train,
                                                        batch_size=batch_size),
                                           epochs=100,
                                           validation_data=(x_test, y_test),
                                           workers=4)


def plot_history(logger):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")


plot_history(history_callback)
plt.show()'''