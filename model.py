# Import necessary libraries.
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# If the car's steering angle is negative, a left turn, take image from the right camera and decrease
# steering angle by 0.2. If the angle is positive, take the left image and increase by 0.2 Images with 
# no steering angle, take left image and add angle by 0.2, and take right image and subtract by 0.2.
def get_RL_angles(angles, left_train, right_train):
    new_angles, new_train = [], []
    for i in range(len(angles)):
        if angles[i] == 0:
            new_train.append(left_train[i])
            new_train.append(right_train[i])
            new_angles.append(0.2)
            new_angles.append(-0.2)
        if angles[i] < 0:
            n_train = right_train[i]
            new_train.append(n_train)
            new_angles.append(angles[i]-0.2)
        if angles[i] > 0:
            n_train = left_train[i]
            new_train.append(n_train)
            new_angles.append(angles[i]+0.2)
    return np.array(new_angles), np.array(new_train)

# Take every center image. If steering angle is negative, warp image by a random factor so it seems the car 
# is facing farther to the right and decrease angle accordingly, and visa versa. Ignore images where angle
# is 0.
def warp_persp(angle, img):
    new_angle, new_img = [], []
    n = np.random.rand(len(angle)) * 0.3
    for i in range(len(img)):
        if angle[i] < 0:
            src = np.float32([[int(160*n[i]), 0], [160 , 0], [int(160*n[i]), 33], [160, 33]])
            dst = np.float32([[0, 0], [160, 0], [0, 33], [160, 33]])
            new_angle.append(angle[i]-n[i])
            M = cv2.getPerspectiveTransform(src, dst)
            shp = (img[i].shape[1], img[i].shape[0])
            warped = cv2.warpPerspective(img[i], M, shp, flags=cv2.INTER_LINEAR)
            new_img.append(warped)
        if angle[i] > 0:
            src = np.float32([[0, 0], [int(160*(1-n[i])) , 0], [0, 33], [int(160*(1-n[i])), 33]])
            dst = np.float32([[0, 0], [160, 0], [0, 33], [160, 33]])
            new_angle.append(angle[i]+n[i])
            M = cv2.getPerspectiveTransform(src, dst)
            shp = (img[i].shape[1], img[i].shape[0])
            warped = cv2.warpPerspective(img[i], M, shp, flags=cv2.INTER_LINEAR)
            new_img.append(warped)
    return np.array(new_angle), np.array(new_img)

# Train, validation, and test split of image filenames and steering angles.
def train_test_split(fn, val_size=0.05):
    dlog = pd.read_csv(fn, header=None)
    num_images = dlog.shape[0]
    
    # This allows the test set to be processed separately. They aren't augmented and 
    # are kept in chronological order because the test set is one loop around the first track.
    if val_size == 0.:
        train = np.array([dlog[0][i][21:] for i in range(num_images)])
        train_angles = dlog[3].values
        center_valid, valid_angles = [], []
    
    # The below is for the train-validation split.
    else:
        idx = np.arange(num_images)
        np.random.shuffle(idx)
        train_idx = idx[:-int(val_size*num_images)]
        val_idx = idx[-int(val_size*num_images):]
    
        center_train = np.array([dlog[0][train_idx[i]][21:] for i in range(len(train_idx))])
        left_train = np.array([dlog[1][train_idx[i]][21:] for i in range(len(train_idx))])
        right_train = np.array([dlog[2][train_idx[i]][21:] for i in range(len(train_idx))])
        center_angles = dlog[3].values[train_idx]
        
        xtra_angles, xtra_train = get_RL_angles(center_angles, left_train, right_train)
        new_angles = np.concatenate((center_angles, xtra_angles), axis=0)
        new_train = np.concatenate((center_train, xtra_train), axis=0)
    
        num_train = len(new_angles)
        new_train_idx = np.arange(num_train)
        np.random.shuffle(new_train_idx)
    
        train = np.array([new_train[new_train_idx[i]] for i in range(num_train)])
        train_angles = np.array([new_angles[new_train_idx[i]] for i in range(num_train)])
        
        # Do not augment validation data
        center_valid = np.array([dlog[0][val_idx[i]][21:] for i in range(len(val_idx))])
        valid_angles = dlog[3].values[val_idx]
    
    return train, train_angles, center_valid, valid_angles   

# I collected the data from track 1 and 2 separately, so this function ensures both tracks are well 
# represented in the validation data.
def combine_train_data(fns):
    num_files = len(fns)
    train = 0
    for j in range(num_files-1):
        train1, angle_train1, valid1, angle_valid1 = train_test_split(fns[j])
        train2, angle_train2, valid2, angle_valid2 = train_test_split(fns[j+1])
        
        train_combined = np.concatenate((train1, train2), axis=0)
        angle_train_combined = np.concatenate((angle_train1, angle_train2), axis=0)
        valid_combined = np.concatenate((valid1, valid2), axis=0)
        angle_valid_combined = np.concatenate((angle_valid1, angle_valid2), axis=0)
        
        # This is for the first pass if you need to combine more than two directories of training data.
        if train == 0:
            train = train_combined
            angle_train = angle_train_combined
            valid = valid_combined
            angle_valid = angle_valid_combined
            
        else:
            train = np.concatenate((train, train_combined), axis=0)
            angle_train = np.concatenate((angle_train, angle_train_combined), axis=0)
            valid = np.concatenate((valid, valid_combined), axis=0)
            angle_valid = np.concatenate((angle_valid, angle_valid_combined), axis=0)

    idx = np.arange(len(train))
    np.random.shuffle(idx)
    train = np.array([train[idx[i]] for i in range(len(train))])
    angle_train = np.array([angle_train[idx[i]] for i in range(len(train))])
    
    idx_v = np.arange(len(valid))
    np.random.shuffle(idx_v)
    valid = np.array([valid[idx_v[i]] for i in range(len(valid))])
    angle_valid = np.array([angle_valid[idx_v[i]] for i in range(len(valid))])
    
    return train, angle_train, valid, angle_valid

def get_batches(images, angles, batch_size, TEST=False):
    num_images = len(images)
    while 1:
        for i in range (0, num_images, batch_size):
            # If using the generator for validation or test sets, do not augment data.
            if TEST:
                if num_images - i >= batch_size:
                    batch_y = angles[i:i+batch_size]
                    batch_x = np.array([cv2.resize(plt.imread('/home/ubuntu/data'+images[i+j])[60:140, :], dsize=(160, 40)) 
                                        for j in range(batch_size)])
                # Make sure to capture the last batch even if there aren't enough images left for a full batch.
                else:
                    batch_y = angles[i:]
                    batch_x = np.array([cv2.resize(plt.imread('/home/ubuntu/data'+images[i+j])[60:140, :], dsize=(160, 40)) 
                                        for j in range(len(batch_y))])
            
            # If using the generator for the training data, augment each batch by a factor of 4 by flipping then
            # randomly warping each image. 
            else:
                if num_images - i >= batch_size:
                    batch_y = angles[i:i+batch_size]
                    y_flip = -batch_y
                    batch_y = np.concatenate((batch_y, y_flip))

                    batch_x = np.array([cv2.resize(plt.imread('/home/ubuntu/data'+images[i+j])[60:140, :], dsize=(160, 40)) 
                                        for j in range(batch_size)])
                    x_flip = np.array([cv2.flip(batch_x[i], 1) for i in range(len(batch_x))])
                    batch_x = np.concatenate((batch_x, x_flip), axis=0)
                    
                    y_aug, x_aug = warp_persp(batch_y, batch_x)
                    
                    batch_x = np.concatenate((batch_x, x_aug), axis=0) 
                    batch_y = np.concatenate((batch_y, y_aug))

                else:
                    batch_y = angles[i:]

                    batch_x = np.array([cv2.resize(plt.imread('/home/ubuntu/data'+images[i+j])[60:140, :], dsize=(160, 40)) 
                                        for j in range(len(batch_y))])
                    x_flip = np.array([cv2.flip(batch_x[i], 1) for i in range(len(batch_x))])
                    batch_x = np.concatenate((batch_x, x_flip), axis=0)

                    y_flip = -batch_y
                    batch_y = np.concatenate((batch_y, y_flip))
                    
                    y_aug, x_aug = warp_persp(batch_y, batch_x)

                    batch_x = np.concatenate((batch_x, x_aug), axis=0)
                    batch_y = np.concatenate((batch_y, y_aug))
                
            yield batch_x, batch_y

# Read the driving logs, split into train, validation, and test sets and print out basic info about them.
fns =['/home/ubuntu/data/car_sim_data_more/driving_log.csv', '/home/ubuntu/data/car_sim_data_track2_more/driving_log.csv']
train, angle_train, valid, angle_valid = combine_train_data(fns)
fn_test = '/home/ubuntu/data/car_sim_data_test/driving_log.csv'
test, angle_test, valid3, angle_valid3 = train_test_split(fn_test, val_size=0.)

print('Number of training examples with partial augmentation: {}'.format(len(train)))
print('Number of validation examples: {}'.format(len(valid)))
print('Number of test examples: {}'.format(len(test)))
print('')
print('Number of left turns in train set: {}'.format(len([ang for ang in angle_train if ang < 0])))
print('Number of right turns in train set: {}'.format(len([ang for ang in angle_train if ang > 0])))
print('')
print('Number of left turns in validation set: {}'.format(len([ang for ang in angle_valid if ang < 0])))
print('Number of right turns in validation set: {}'.format(len([ang for ang in angle_valid if ang > 0])))
print('')
print('Number of left turns in test set: {}'.format(len([ang for ang in angle_test if ang < 0])))
print('Number of right turns in test set: {}'.format(len([ang for ang in angle_test if ang > 0])))

# Import Deep Learning dependencies.
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# Set hyperparameters.
epochs = 2
batch_size = 16
drop_rate = 0.5
num_train = len(train)
num_valid = len(valid)
num_test = len(test)
train_steps = num_train//batch_size
valid_steps = num_valid//batch_size
test_steps = num_test//batch_size
width, height, channels = 160, 40, 3

# Create train, validation, and test data generators.
train_generator = get_batches(train, angle_train, batch_size)
valid_generator = get_batches(valid, angle_valid, batch_size, TEST=True)
test_generator = get_batches(test, angle_test, batch_size, TEST=True)

# Build the model.
model = Sequential()
# Normalize the data.
model.add(Lambda(lambda x: x/255.,
                 input_shape=(height, width, channels),
                 output_shape=(height, width, channels)))
model.add(Conv2D(64, 5, strides=(1, 1), activation=None, padding='same', use_bias=True))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64, 5, strides=(2, 2), activation=None, padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Dropout(drop_rate))
model.add(Conv2D(64, 5, strides=(1, 1), activation=None, padding='same', use_bias=True))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64, 5, strides=(2, 2), activation=None, padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Dropout(drop_rate))
model.add(Conv2D(64, 3, strides=(1, 1), activation=None, padding='same', use_bias=True))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64, 3, strides=(2, 2), activation=None, padding='same', use_bias=True))
model.add(LeakyReLU(alpha=0.2))
model.add(Flatten())
model.add(Dense(512, activation=None))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(drop_rate))
model.add(Dense(512, activation=None))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(1, activation=None))
model.compile(loss='mae', optimizer='adam')
model.fit_generator(train_generator, train_steps, epochs=epochs, validation_data=valid_generator, 
                    validation_steps=valid_steps, workers=1, verbose=1)
model.save('output/model.h5')
test_loss = model.evaluate_generator(test_generator, test_steps)
print('')
print('test_loss: {:.4}'.format(test_loss))
print('')
print('Model trained and saved :)')