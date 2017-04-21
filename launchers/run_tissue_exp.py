from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime
from infogan.misc.tissue_data_reader import get_data
import sys

'''
Sample Run:
nohup python run_tissue_exp.py /home/ahmet/workspace/medical-image-extractor/data/patches_for_gan /home/ahmet/workspace/tensorflow/tensorboard/medical_gan/small_128 /home/ahmet/workspace/tensorflow/tensorboard/medical_gan/small_128/check_pt > output.txt 2>&1 &

python run_tissue_exp.py /Users/ahmetkucuk/Documents/Research/Medical/sample_patches_small logs/tissue2 ckt/tissue

'''
if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) < 3:
		print("args: data dir, log dir, checkpoint dir")
		exit()

	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

	data_dir = args[0]
	root_log_dir = args[1]
	root_checkpoint_dir = args[2]
	batch_size = 128
	updates_per_epoch = 100
	max_epoch = 50

	exp_name = "tissue_%s" % timestamp

	log_dir = os.path.join(root_log_dir, exp_name)
	checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

	mkdir_p(log_dir)
	mkdir_p(checkpoint_dir)

	dataset = get_data(dataset_dir=data_dir, size=128)

	latent_spec = [
		(Uniform(62), False),
		(Categorical(10), True),
		(Uniform(1, fix_std=True), True),
		(Uniform(1, fix_std=True), True),
	]

	model = RegularizedGAN(
		output_dist=MeanBernoulli(dataset.image_dim),
		latent_spec=latent_spec,
		batch_size=batch_size,
		image_shape=dataset.image_shape,
		network_type="mnist",
	)

	algo = InfoGANTrainer(
		model=model,
		dataset=dataset,
		batch_size=batch_size,
		exp_name=exp_name,
		log_dir=log_dir,
		checkpoint_dir=checkpoint_dir,
		max_epoch=max_epoch,
		updates_per_epoch=updates_per_epoch,
		info_reg_coeff=1.0,
		generator_learning_rate=1e-3,
		discriminator_learning_rate=2e-4,
	)

	algo.train()
