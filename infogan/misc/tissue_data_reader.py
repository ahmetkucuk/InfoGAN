from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from infogan.misc.datasets import TissueDataset
from infogan.misc.datasets import Dataset
import numpy as np


class ImageReader(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self, size=256):
		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
		self._decode_jpeg = tf.image.resize_images(self._decode_jpeg, [size, size])

	def read_image_dims(self, sess, image_data):
		image = self.decode_jpeg(sess, image_data)
		return image.shape[0], image.shape[1]

	def decode_jpeg(self, sess, image_data):
		image = sess.run(self._decode_jpeg,
						 feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image


def _get_filenames_and_classes(dataset_dir):
	"""Returns a list of filenames and inferred class names.

	Args:
	  dataset_dir: A directory containing a set of subdirectories representing
		class names. Each subdirectory should contain PNG or JPG encoded images.

	Returns:
	  A list of image file paths, relative to `dataset_dir` and the list of
	  subdirectories, representing class names.
	"""
	directories = []
	class_names = []
	for filename in os.listdir(dataset_dir):
		path = os.path.join(dataset_dir, filename)
		if os.path.isdir(path):
			directories.append(path)
			class_names.append(filename)

	photo_filenames = []
	for directory in directories:
		for filename in os.listdir(directory):
			path = os.path.join(directory, filename)
			photo_filenames.append(path)

	return photo_filenames, sorted(class_names)


def get_data(dataset_dir, size):

	filenames, class_names = _get_filenames_and_classes(dataset_dir=dataset_dir)
	images = []
	labels = []

	with tf.Graph().as_default():
		image_reader = ImageReader(size)

		with tf.Session('') as sess:
			for i in range(len(filenames)):
				image_file = tf.gfile.FastGFile(filenames[i], 'r').read()
				image = image_reader.decode_jpeg(sess, image_file)
				images.append(image)
				class_names_to_ids = dict(zip(class_names, range(len(class_names))))
				class_name = os.path.basename(os.path.dirname(filenames[i]))
				class_id = class_names_to_ids[class_name]
				labels.append(class_id)
	split_at = int(len(images)*0.8)
	train_images = np.asarray(images[:split_at], np.int32)
	train_labels = np.asarray(labels[:split_at], np.int32)

	test_images = np.asarray(images[split_at:], np.int32)
	test_labes = np.asarray(labels[split_at:], np.int32)

	training_dataset = Dataset(images=train_images, labels=train_labels)
	testing_dataset = Dataset(images=test_images, labels=test_labes)

	return TissueDataset(train=training_dataset, test=testing_dataset)
