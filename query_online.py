# -*- coding: utf-8 -*-
# Author: yongyuan.name
#from extract_features import extract_feat

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time
import pdb
import os


ap = argparse.ArgumentParser()
ap.add_argument("-query", required = False,
	help = "Path to query images which contains images to be indexed")
ap.add_argument("-path", required = False,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-result", required = False,
	help = "Path for output retrieved images")
ap.add_argument("-weight", required = False,
	help = "Path for weight")
args = vars(ap.parse_args())

result = args["result"]
f = open('./logs/%s'%(result),'w')
# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
feats_4068 = h5f['lay_4068'][:]
feats_1024 = h5f['lay_1024'][:]
feats_512 = h5f['lay_512'][:]
feats_48 = h5f['lay_48'][:]
imgNames = h5f['image_names'][:]
h5f.close()

print "--------------------------------------------------"
print "               searching starts"
print "--------------------------------------------------"

# read and show query image
queryDir = args["query"]
imageDir = args["path"]
#weight_path = args["weight"]
'''
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()
'''

def update_size(size_dict,label):
	if size_dict.has_key(label):
		size_dict[label] += 1
	else:
		size_dict[label] = 1
	return 0

def update_error(error_dict,label):
	if error_dict.has_key(label):
		error_dict[label] += 1
	else:
		error_dict[label] = 1
	return 0


# extract query image's feature, compute simlarity score and sort
if queryDir is None:
	for feats in [feats_4068, feats_1024, feats_512, feats_48]:
		print >>f, 'query results...\n'
		error_rec = dict()
		size_rec = dict()
		clock = 0
		acc = 0
		size = 0
		for names in imgNames:
			label = names.split('/')[-1].split('_')[0]
			update_size(size_rec,label)
		cnt = 0
		for queryDir,queryVec in zip(imgNames,feats):
			#queryDir = os.path.join(imageDir,queryDir)
			#queryVec = extract_feat(queryDir,weight_path)

			if 'test' in queryDir:
				cnt += 1
				time1 = time.time()
				scores = np.dot(queryVec, feats.T)
				rank_ID = np.argsort(scores)[::-1]
				rank_score = scores[rank_ID]

				time2 = time.time()
				clock = time2 - time1 + clock

				maxres = 6
				label = queryDir.split('/')[-1].split('_')[0]
				if size_rec[label] < maxres:
					maxres = size_rec[label]
				size += (maxres-1)
				imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
				query_name = queryDir.split('/')[-1]
				if cnt <= 5:
					print >>f, query_name
					print >>f, imlist
				for i in imlist[1:]:
					l = i.split('_')[0]
					if l == label:
						acc += 1
					#	print 'correct: query:' + queryDir + '    ' + i
					else:
						update_error(error_rec,label)
					#	print 'error: query:' + queryDir + '    ' + i

		#print rank_ID
		#print rank_score
		print 'acc:'+str(float(acc)/size)
		print 'total time:'+str(clock)
		for i in size_rec.keys():
			if error_rec.has_key(i):
				print i +' size:'+str(size_rec[i])+' error:'+str(error_rec[i])
			else:
				print i +' size:'+str(size_rec[i])+' error:0'
		print >>f, float(acc)/size
		print >>f, clock
	f.close()
else:
	queryVec = extract_feat(queryDir)
	scores = np.dot(queryVec, feats.T)
	rank_ID = np.argsort(scores)[::-1]
	rank_score = scores[rank_ID]
	# number of top retrieved images to show
	maxres = 6
	imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
	print "top %d images in order are: " %maxres, imlist



# show top #maxres retrieved result one by one
'''
for i,im in enumerate(imlist):
    image = mpimg.imread(args["result"]+"/"+im)
    plt.title("search output %d" %(i+1))
    plt.imshow(image)
    plt.show()
'''
