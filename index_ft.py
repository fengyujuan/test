# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse
import pdb
from numpy import linalg as LA

from keras import applications
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model,Sequential


from keras.applications.vgg16 import VGG16

ap = argparse.ArgumentParser()
ap.add_argument("-path", required = True,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
ap.add_argument("-weight",required = False,
	help = "Path to weights of the model")
args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
	dir_list = os.listdir(path)
	img_list = []
	for dir in dir_list:
		dir = os.path.join(path,dir)
		extend = [os.path.join(dir,d) for d in os.listdir(dir)]
		img_list.extend(extend)
	return img_list

def get_imlist_dis(path):
	return [os.path.join(path,f) for f in os.listdir(path)]

def extract_VGG_feat(model,img_path):
	input_shape = (100,100,3)
	img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	feat = model.predict(img)
	norm_feat = feat[0]/LA.norm(feat[0])
	return norm_feat

def extract_feat(model,img_path):
	# weights: 'imagenet'
	# pooling: 'max' or 'avg'
	# input_shape: (width, height, 3), width and height should >= 48	
	input_shape = (100, 100, 3)	
	img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	feat = model.predict(img)
	norm_feat = feat[0]/LA.norm(feat[0])
	return norm_feat

def construct_model(weight_path):
	classes = 11
	input_shape = (100, 100, 3)
	base_model = applications.VGG16(weights='imagenet',input_shape = (input_shape[0], input_shape[1], input_shape[2]),include_top=False)
	x = base_model.output
	x = Flatten(input_shape=base_model.output_shape[1:], name='flatten')(x)
	x = Dense(1024, activation='relu',name='F1')(x)
	x = Dropout(0.3)(x)
	x = Dense(512, activation='relu',name='F2')(x)
	x = Dropout(0.3)(x)
	x = Dense(bits, activation='sigmoid',name='F3')(x)
	x = Dropout(0.2)(x)
	output = Dense(classes, activation='sigmoid')(x)
	model = Model(inputs= base_model.input, outputs= output)
	model.load_weights(weight_path,by_name=False)
	sub_model0 = Model(input=model.input,output=model.get_layer('flatten').output)
	sub_model1 = Model(input=model.input,output=model.get_layer('F1').output)
	sub_model2 = Model(input=model.input,output=model.get_layer('F2').output)
	sub_model3 = Model(input=model.input,output=model.get_layer('F3').output)
	return [sub_model0, sub_model1, sub_model2, sub_model3]
'''
 Extract features and index the images
'''
if __name__ == "__main__":

	db0 = args["path"]
	feats_4068 = []
	feats_1024 = []
	feats_512 = []
	feats_48 = []
	names = []
	#weight_path = args["weight"]
	bits = 256
	lr = 0.05
	weight_path = './models/finetune_%fbits_%f.hdf5'%(bits, lr)
	model0, model1, model2, model3 = construct_model(weight_path)
	for p in ['train', 'test', 'valid']:
		#db_path = os.path.join(db,p)
		img_list = []
		#if p is not 'distract':
		db = os.path.join(db0,p)
		img_list = get_imlist(db)
		#else:
		#	img_list = get_imlist_dis(db_path)
		
		print "--------------------------------------------------"
		print "         feature extraction starts"
		print "--------------------------------------------------"
		#input_shape = (100,100,3)	
		#model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)

		for i, img_path in enumerate(img_list):
			norm_feat = extract_feat(model0,img_path)
			feats_4068.append(norm_feat)
			norm_feat = extract_feat(model1,img_path)
			feats_1024.append(norm_feat)
			norm_feat = extract_feat(model2,img_path)
			feats_512.append(norm_feat)
			norm_feat = extract_feat(model3,img_path)
			feats_48.append(norm_feat)
			#norm_feat_conv = extract_conv_feat(model_conv,img_path)
			#norm_feat = np.append(norm_feat,norm_feat_conv)
			#norm_feat = extract_VGG_feat(model,img_path)
			img_name = os.path.split(img_path)[1]
			names.append(img_name)
			print img_name
				#print "extracting feature from image No. %d , %d images in total" %((i+1), len(img_list))

	feats_4068 = np.array(feats_4068)
	feats_1024 = np.array(feats_1024)
	feats_512 = np.array(feats_512)
	feats_48 = np.array(feats_48)
	# directory for storing extracted features
	output = args["index"]
	
	print "--------------------------------------------------"
	print "      writing feature extraction results ..."
	print "--------------------------------------------------"
	
	
	h5f = h5py.File(output, 'w')
	h5f.create_dataset('lay_4068', data = feats_4068)
	h5f.create_dataset('lay_1024', data = feats_1024)
	h5f.create_dataset('lay_512', data = feats_512)
	h5f.create_dataset('lay_48', data = feats_48)
	h5f.create_dataset('image_names', data = names)
	h5f.close()
	
