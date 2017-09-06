# -*- coding: utf-8 -*-
# Author: yongyuan.name

import numpy as np
from numpy import linalg as LA
import os
np.random.seed(1234)

from keras import applications
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model,Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import pdb
import argparse

'''
 Use vgg16 model to extract features
 Output normalized feature vector
'''
ap = argparse.ArgumentParser()
ap.add_argument("-aug", required = True)
ap.add_argument("-lr", dest='lr', type=float, default=0.005,help='learning rate')
ap.add_argument("-bits", dest='bits', type=int, default=128,help='binary bits')
ap.add_argument("-test", dest='test', action='store_true', default=False, help='only test step')
args = vars(ap.parse_args())

# weights: 'imagenet'
# pooling: 'max' or 'avg'
# input_shape: (width, height, 3), width and height should >= 48
def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


if __name__ == "__main__":

	aug = args['aug']
	lr = args['lr']
	bits = args['bits']
	input_shape = (100, 100, 3)
	img_height = 100
	img_width = 100
	'''
	base_model = applications.VGG16(weights='imagenet',input_shape = (input_shape[0], input_shape[1], input_shape[2]),include_top=False)
	top_model = Sequential()
	top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(11, activation='sigmoid'))
	model = Model(inputs= base_model.input, outputs= top_model(base_model.output))
	'''
	db = './oxford_flow2/train'
        db_test = './oxford_flow2/test'
        db_valid = './oxford_flow2/valid'
	classes = len(os.listdir(db))
	base_model = applications.VGG16(weights='imagenet',input_shape = (input_shape[0], input_shape[1], input_shape[2]),include_top=False)
	x = base_model.output
	x = Flatten(input_shape=base_model.output_shape[1:],name='flatten')(x)
	x = Dense(1024, activation='relu',name='F1')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu',name='F2')(x)
	x = Dropout(0.3)(x)
	x = Dense(bits, activation='sigmoid',name='F3')(x)
	x = Dropout(0.2)(x)
	output = Dense(classes, activation='sigmoid')(x)
	model = Model(inputs= base_model.input, outputs= output)
	rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0)
	model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop, #optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
	print model.summary()


	'''
	img_list = get_imlist(db)
	img_test_list = get_imlist(db_test)
	classes = 0
	name_dict = dict()

	for img_path in img_list:
		temp = img_path.split('/')[-1].split('_')
		name = ''
		for i in temp[:-1]:
			name = name + i + '_'
		name = name[:-1]
		if not name_dict.has_key(name):
			classes += 1
			name_dict[name] = classes

	sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
	x = []
	y = []
	countimg = 0
	for img_path in img_list:
		img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		x.append(img[0])
		temp = img_path.split('/')[-1].split('_')
		name = ''
		for i in temp[:-1]:
			name = name + i + '_'
		name = name[:-1]
		cl = name_dict[name]
		y0 = np.zeros((classes,),dtype=np.int)
		y0[cl-1] = 1
		y.append(y0)
		countimg += 1
		if countimg%500 == 0: print countimg
	x = np.asarray(x)
	y = np.asarray(y)

	x_val = []
	y_val = []
	countimg = 0
	for img_path in img_test_list:
		img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		x_val.append(img[0])
		temp = img_path.split('/')[-1].split('_')
		name = ''
		for i in temp[:-1]:
			name = name + i + '_'
		name = name[:-1]
		cl = name_dict[name]
		y0 = np.zeros((classes,),dtype=np.int)
		y0[cl-1] = 1
		y_val.append(y0)
		countimg += 1
		if countimg%500 == 0: print countimg
	x_val = np.asarray(x_val)
	y_val = np.asarray(y_val)
	'''
	if aug == '1':#augmentation
		nb_train_samples = 1600
		nb_epoch = 100
		nb_validation_samples = 200
		nb_test_samples = 93
		train_datagen = ImageDataGenerator(
				rescale=1./255,
				shear_range=0.2,
				zoom_range=0.2,
				horizontal_flip=True)

		test_datagen = ImageDataGenerator(rescale=1./255)

		train_generator = train_datagen.flow_from_directory(
				db,
				target_size=(img_height, img_width),
				batch_size=16,
				class_mode='categorical')

		test_generator = test_datagen.flow_from_directory(
				db_test,
				target_size=(img_height, img_width),
				batch_size=16,
				class_mode='categorical')

		validation_generator = test_datagen.flow_from_directory(
				db_valid,
				target_size=(img_height, img_width),
				batch_size=16,
				class_mode='categorical')
		if not args['test']:
		    # fine-tune the model
		    checkpointer = ModelCheckpoint(filepath='./models/finetune_%fbits_%f.hdf5'%(bits, lr),verbose=1,save_best_only=True)
		    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
		    model.fit_generator(
				    train_generator,
				    samples_per_epoch=nb_train_samples,
				    nb_epoch=nb_epoch,
				    validation_data=validation_generator,
				    callbacks=[checkpointer, earlystopper],
				    nb_val_samples=nb_validation_samples)
		model.load_weights("./models/finetune_%fbits_%f.hdf5"%(bits, lr))
		scores = model.evaluate_generator(test_generator, steps=nb_test_samples)
		print scores

	model.save_weights('weights/fine_tune_split_oxford2_%fbits_%f_8.h5'%(bits,lr))

