"""
main.py - module to pixelate images
"""

""" import dependencies """
import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import MiniBatchKMeans

def main(args):
	"""
	main - function to pixelate images
	Input:
		args.image_path		Path to image file
		args.output_path	Path to store pixelated image
	"""
	if args.method is 'filter':
		filter_method(args, 5, 5, 24)
	else:
		kmean_method(args, 5, 6)

def filter_method(args, pixelSize = 5, filter_radius = 7, colors = 24):
	# read iamge
	image = Image.open(args.image_path).convert('RGB')

	image = image.filter(ImageFilter.MedianFilter(filter_radius))
	image = image.convert('P', palette=Image.ADAPTIVE, colors=colors)
	image = image.resize((image.size[0]//pixelSize, image.size[1]//pixelSize), Image.CUBIC)
	image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.CUBIC)
	image.save(args.output_path)

def kmean_method(args, pixelSize = 5, n_clusters = 6):
	image = cv2.imread(args.image_path)
	(h, w) = image.shape[:2]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

	# reshape the image into a feature vector so that k-means
	# can be applied
	image = image.reshape((image.shape[0] * image.shape[1], 3))

	# apply k-means using the specified number of clusters and
	# then create the quantized image based on the predictions
	clt = MiniBatchKMeans(n_clusters = n_clusters)
	labels = clt.fit_predict(image)
	quant = clt.cluster_centers_.astype("uint8")[labels]

	# reshape the feature vectors to images
	quant = quant.reshape((h, w, 3))
	image = image.reshape((h, w, 3))

	# convert from L*a*b* to RGB
	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
	quant = cv2.resize(quant, (quant.shape[0] // pixelSize, quant.shape[1] // pixelSize), cv2.INTER_NEAREST)
	quant = cv2.resize(quant, (quant.shape[0] * pixelSize, quant.shape[1] * pixelSize), cv2.INTER_NEAREST)

	# display the images and wait for a keypress
	#cv2.imshow("image", quant)
	#cv2.waitKey(0)

	cv2.imwrite(args.output_path, quant)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Python module to pixalate images")
	parser.add_argument('--method', type = str, default = 'filter', help = 'Method to pixelate images: filter or kmean')
	parser.add_argument('--image_path', type = str, help = 'Path to image file')
	parser.add_argument('--output_path', type = str, help = 'Path to store pixelated image')

	main(parser.parse_args())
