# encoding: utf-8
from __future__ import division
import tensorflow as tf
import numpy as np
from Dataset import Dataset
from time import time
import os
import pdb
import math
import heapq
from flip_gradient import flip_gradient

def get_train(u, train, num_items, num_negatives, user_input, item_input, labels):
	for i in train[u]:
		user_input.append(u)
		item_input.append(i)
		labels.append(1)
		for t in range(num_negatives):
			j = np.random.randint(num_items)
			while j in train[u]:
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	return user_input, item_input, labels

def get_train_padding(u, train, num, num_items, num_negatives, user_input, item_input, labels):
	count = 0
	while(count < num):
		for i in train[u]:
			count += 1
			if count <= num:                   
				user_input.append(u)
				item_input.append(i)
				labels.append(1)
				for t in range(num_negatives):
					j = np.random.randint(num_items)
					while j in train[u]:
						j = np.random.randint(num_items)
					user_input.append(u)
					item_input.append(j)
					labels.append(0)
	return user_input, item_input, labels

def get_cross_data(left_data, right_data):
	new_data = []
	num_data = len(left_data)
	for i in range(num_data):
		new_data.append(left_data[i])
		new_data.append(right_data[i])
	return new_data

def get_train_cutting(u, train, num, num_items, num_negatives, user_input, item_input, labels):
	count = 0
	item_index = []
	for t in range(num):
		while True:
			index = np.random.randint(len(train[u]))
			if index not in item_index:
				item_index.append(index)
				break
	for i in item_index:
		item = train[u][i]
		user_input.append(u)
		item_input.append(item)
		labels.append(1)
		for t in range(num_negatives):
			j = np.random.randint(num_items)
			while j in train[u]:
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	return user_input, item_input, labels

def get_train_instances(source_train, source_num_items, target_train, target_num_items, num_negatives):
	source_user_input, source_item_input, source_labels = [],[],[]
	target_user_input, target_item_input, target_labels = [],[],[]
	num_users = len(source_train.keys())
	for u in source_train.keys():
		if u in target_train.keys():
			s_num = len(source_train[u])
			t_num = len(target_train[u])
			if s_num > t_num:
				source_user_input, source_item_input, source_labels = get_train_cutting(u, source_train, t_num, source_num_items, num_negatives, source_user_input, source_item_input, source_labels)
				target_user_input, target_item_input, target_labels = get_train(u, target_train, target_num_items, num_negatives, target_user_input, target_item_input, target_labels)
			elif s_num < t_num:
				source_user_input, source_item_input, source_labels = get_train_padding(u, source_train, t_num, source_num_items, num_negatives, source_user_input, source_item_input, source_labels)
				target_user_input, target_item_input, target_labels = get_train(u, target_train, target_num_items, num_negatives, target_user_input, target_item_input, target_labels)
			else:
				source_user_input, source_item_input, source_labels = get_train(u, source_train, source_num_items, num_negatives, source_user_input, source_item_input, source_labels)
				target_user_input, target_item_input, target_labels = get_train(u, target_train, target_num_items, num_negatives, target_user_input, target_item_input, target_labels)

	common_user_input = get_cross_data(source_user_input, target_user_input)

	source_user_label = np.zeros(len(source_user_input)).tolist()
	target_user_label = np.ones(len(target_user_input)).tolist()    
	common_user_label = get_cross_data(source_user_label, target_user_label) 

	source_user_input = get_cross_data(source_user_input, source_user_input)
	source_item_input = get_cross_data(source_item_input, source_item_input)
	source_labels = get_cross_data(source_labels, source_labels)
	target_user_input = get_cross_data(target_user_input, target_user_input)
	target_item_input = get_cross_data(target_item_input, target_item_input)
	target_labels = get_cross_data(target_labels, target_labels)

	return source_user_input, source_item_input, source_labels, target_user_input, target_item_input, target_labels, common_user_input, common_user_label

def get_test_instances(source_testRatings, source_testNegatives, target_testRatings, target_testNegatives):
	source_user_input, source_item_input, source_negtivates = [],[],[]
	target_user_input, target_item_input, target_negtivates = [],[],[]
	common_user_input, common_user_label = [], []
	for i in range(len(source_testRatings)):
		source_user_input.append(source_testRatings[i][0])
		source_item_input.append(source_testRatings[i][1])
		source_negtivates.append(source_testNegatives[i])
		target_user_input.append(target_testRatings[i][0])
		target_item_input.append(target_testRatings[i][1])
		target_negtivates.append(target_testNegatives[i])
		common_user_input.append(target_testRatings[i][0])
	return source_user_input, source_item_input, source_negtivates, target_user_input, target_item_input, target_negtivates, common_user_input

def get_train_instance_batch_change(count, batch_size, source_user_input_train, source_item_input_train, source_labels_train, target_user_input_train, target_item_input_train, target_labels_train, common_user_input_train, common_user_label_train):
	source_user_input_batch, source_item_input_batch, source_labels_batch, target_user_input_batch, target_item_input_batch, target_labels_batch, common_user_input_batch, common_user_label_batch = [], [], [], [], [], [], [], []

	for idx in range(batch_size):
		index = (count*batch_size + idx) % len(source_user_input_train)
		source_user_input_batch.append(source_user_input_train[index])
		source_item_input_batch.append(source_item_input_train[index])
		source_labels_batch.append([source_labels_train[index]])
		target_user_input_batch.append(target_user_input_train[index])
		target_item_input_batch.append(target_item_input_train[index])
		target_labels_batch.append([target_labels_train[index]])
		common_user_input_batch.append(common_user_input_train[index])
		common_user_label_batch.append([common_user_label_train[index]])
	return source_user_input_batch, source_item_input_batch, source_labels_batch, target_user_input_batch, target_item_input_batch, target_labels_batch, common_user_input_batch, common_user_label_batch

def train_model():
	source_users = tf.placeholder(tf.int32, shape=[None])
	source_items = tf.placeholder(tf.int32, shape=[None])
	source_rates = tf.placeholder(tf.float32, shape=[None, 1])
	target_users = tf.placeholder(tf.int32, shape=[None])
	target_items = tf.placeholder(tf.int32, shape=[None])
	target_rates = tf.placeholder(tf.float32, shape=[None, 1])
	common_users = tf.placeholder(tf.int32, shape=[None])
	common_label = tf.placeholder(tf.float32, shape=[None, 1])
	grl_lambds = tf.placeholder(tf.float32, [])
	global_step = tf.Variable(tf.constant(0),trainable=False)

	source_user_one_hot = tf.one_hot(indices=source_users, depth=source_num_users, name="source_user_one_hot")
	print("source_user_one_hot: ", source_user_one_hot.get_shape())
	source_item_one_hot = tf.one_hot(indices=source_items, depth=source_num_items, name="source_item_one_hot")
	print("source_item_one_hot: ", source_item_one_hot.get_shape())
	target_user_one_hot = tf.one_hot(indices=target_users, depth=target_num_users, name="target_user_one_hot")
	print("target_user_one_hot: ", target_user_one_hot.get_shape())
	target_item_one_hot = tf.one_hot(indices=target_items, depth=target_num_items, name="target_item_one_hot")
	print("target_item_one_hot: ", target_item_one_hot.get_shape())
	common_user_one_hot = tf.one_hot(indices=common_users, depth=target_num_users, name="common_user_one_hot")
	print("common_user_one_hot: ", common_user_one_hot.get_shape())

	source_user_embed = tf.layers.dense(inputs = source_user_one_hot, units = num_factor, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='source_user_embed')
	print("source_user_embed: ", source_user_embed.get_shape())

	source_item_embed = tf.layers.dense(inputs = source_item_one_hot, units = num_factor, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='source_item_embed')
	print("source_item_embed: ", source_item_embed.get_shape())

	target_user_embed = tf.layers.dense(inputs = target_user_one_hot, units = num_factor, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='target_user_embed')
	print("target_user_embed: ", target_user_embed.get_shape())

	target_item_embed = tf.layers.dense(inputs = target_item_one_hot, units = num_factor, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='target_item_embed')
	print("target_item_embed: ", target_item_embed.get_shape())

	common_user_embed = tf.layers.dense(inputs = common_user_one_hot, units = factor_layers[0], activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='common_user_embed')
	print("common_user_embed: ", common_user_embed.get_shape())

	for idx in range(1, len(factor_layers)):
		common_user_embed = tf.layers.dense(inputs = common_user_embed, units = factor_layers[idx], activation = tf.nn.tanh, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='common_user_embed_layer%d' % idx)
		print("common_user_embed: ", common_user_embed.get_shape())

	common_user_embed_grl = flip_gradient(common_user_embed, grl_lambds)
	common_predict_label = tf.layers.dense(inputs= common_user_embed_grl,
										units = 1,
										activation=None,
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate),
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='common_predict_label')
	print("common_predict_label: ", common_predict_label.get_shape())

	source_atten = tf.layers.dense(inputs = source_user_embed, units = num_factor, activation = tf.nn.relu, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='source_atten')
	print("source_atten: ", source_atten.get_shape())

	source_atten_weight = tf.layers.dense(inputs = source_atten, units = 1, use_bias = False, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='source_atten_weight')
	source_atten_weight = tf.exp(source_atten_weight)
	print("source_atten_weight: ", source_atten_weight.get_shape())

	source_common_atten = tf.layers.dense(inputs = common_user_embed, units = num_factor, activation = tf.nn.relu, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='source_common_atten')
	print("source_common_atten: ", source_common_atten.get_shape())

	source_common_atten_weight = tf.layers.dense(inputs = source_common_atten, units = 1, use_bias = False, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='source_common_atten_weight')
	source_common_atten_weight = tf.exp(source_common_atten_weight)
	print("source_common_atten_weight: ", source_common_atten_weight.get_shape())

	source_common_weight = tf.div(source_atten_weight, source_atten_weight + source_common_atten_weight)
	common_source_weight = 1.0 - source_common_weight

	source_user_embed_final = source_common_weight*source_user_embed + common_source_weight*common_user_embed
	print("source_user_embed_final: ", source_user_embed_final.get_shape())

	source_predict = tf.multiply(source_user_embed_final, source_item_embed, name='source_predict')
	print("source_predict: ", source_predict.get_shape())

	source_predict_rate = tf.layers.dense(inputs= source_predict,
										units = 1,
										activation=None,
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate),
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='source_predict_rate')
	print("source_predict_rate: ", source_predict_rate.get_shape())

	target_atten = tf.layers.dense(inputs = target_user_embed, units = num_factor, activation = tf.nn.relu, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='target_atten')
	print("target_atten: ", target_atten.get_shape())

	target_atten_weight = tf.layers.dense(inputs = target_atten, units = 1, use_bias = False, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='target_atten_weight')
	target_atten_weight = tf.exp(target_atten_weight)
	print("target_atten_weight: ", target_atten_weight.get_shape())

	target_common_atten = tf.layers.dense(inputs = common_user_embed, units = num_factor, activation = tf.nn.relu, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='target_common_atten')
	print("target_common_atten: ", target_common_atten.get_shape())

	target_common_atten_weight = tf.layers.dense(inputs = target_common_atten, units = 1, use_bias = False, activation = None, 
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='target_common_atten_weight')
	target_common_atten_weight = tf.exp(target_common_atten_weight)
	print("target_common_atten_weight: ", target_common_atten_weight.get_shape())

	target_common_weight = tf.div(target_atten_weight, target_atten_weight + target_common_atten_weight)
	common_target_weight = 1.0 - target_common_weight

	target_user_embed_final = target_common_weight*target_user_embed + common_target_weight*common_user_embed
	print("target_user_embed_final: ", target_user_embed_final.get_shape())

	target_predict = tf.multiply(target_user_embed_final, target_item_embed, name='target_predict')
	print("target_predict: ", target_predict.get_shape())

	target_predict_rate = tf.layers.dense(inputs= target_predict,
										units = 1,
										activation=None,
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
										kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate),
										bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate), 
										name='target_predict_rate')
	print("target_predict_rate: ", target_predict_rate.get_shape())

	source_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=source_rates, logits=source_predict_rate))
	
	target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_rates, logits=target_predict_rate))

	common_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=common_label, logits=common_predict_label))

	total_loss = source_gama * source_loss + target_loss + gan_gama * common_loss

	# l_rate = tf.train.exponential_decay(learn_rate,global_step,decay_steps,decay_rate,staircase=True)
	optimizer = tf.train.AdamOptimizer(learn_rate)
	train_step = optimizer.minimize(total_loss, global_step=global_step)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		hit5_list, ndcg5_list, precision5_list, mean_ap5_list, mrr5_list = [], [], [], [], []
		hit10_list, ndcg10_list, precision10_list, mean_ap10_list, mrr10_list = [], [], [], [], []
		hit15_list, ndcg15_list, precision15_list, mean_ap15_list, mrr15_list = [], [], [], [], []
		for e in range(epochs):
			t = time()
			loss_total, loss_source, loss_target, loss_common = 0.0, 0.0, 0.0, 0.0
			count = 0.0
			# get train instances
			source_user_input_train, source_item_input_train, source_labels_train, target_user_input_train, target_item_input_train, target_labels_train, common_user_input_train, common_user_label_train = get_train_instances(source_trainDict, source_num_items, target_trainDict, target_num_items, num_negatives)
			
			# get test instances
			source_user_input_test, source_item_input_test, source_negtivates_test, target_user_input_test, target_item_input_test, target_negtivates_test, common_user_input_test = get_test_instances(source_testRatings, source_testNegatives, target_testRatings, target_testNegatives)

			train_iter_num = int(len(source_user_input_train) / batch_size) + 1
			total_global_step = epochs * train_iter_num
			for i in range(train_iter_num):
				source_user_input_batch, source_item_input_batch, source_labels_batch, target_user_input_batch, target_item_input_batch, target_labels_batch, common_user_input_batch, common_user_label_batch = get_train_instance_batch_change(i, batch_size, source_user_input_train, source_item_input_train, source_labels_train, target_user_input_train, target_item_input_train, target_labels_train, common_user_input_train, common_user_label_train)
				# print("done")
				global_step = e * train_iter_num + i + 1
				process = global_step * 1.0 / total_global_step
				grl_lambda = 2.0 / (1.0 + np.exp(-gamma *  process)) - 1.0

				_, loss, loss_s, loss_t, loss_c = sess.run([train_step, total_loss, source_loss, target_loss, common_loss] ,feed_dict={source_users: source_user_input_batch, source_items: source_item_input_batch, source_rates: source_labels_batch, target_users: target_user_input_batch, target_items: target_item_input_batch, target_rates: target_labels_batch, common_users: common_user_input_batch, common_label: common_user_label_batch, grl_lambds: grl_lambda})

				loss_total += loss
				loss_source += loss_s
				loss_target += loss_t
				loss_common += loss_c
				count += 1.0

			t1 = time()
			print("epoch%d time: %.2fs, loss_total: %.4f, loss_source: %.4f, loss_target: %.4f, loss_common: %.4f" % (e, t1 - t, loss_total/count, loss_source/count, loss_target/count, loss_common/count))
			hits5, ndcgs5, precisions5, mean_aps5, mrrs5, hits10, ndcgs10, precisions10, mean_aps10, mrrs10, hits15, ndcgs15, precisions15, mean_aps15, mrrs15 = eval_model(source_users, source_items, target_users, target_items, common_users, target_predict_rate, sess, source_user_input_test, source_item_input_test, source_negtivates_test, target_user_input_test, target_item_input_test, target_negtivates_test, common_user_input_test, topK_list)
			hitm5, ndcgm5, precisionm5, mean_apm5, mrrm5 = np.mean(hits5), np.mean(ndcgs5), np.mean(precisions5), np.mean(mean_aps5), np.mean(mrrs5)
			hitm10, ndcgm10, precisionm10, mean_apm10, mrrm10 = np.mean(hits10), np.mean(ndcgs10), np.mean(precisions10), np.mean(mean_aps10), np.mean(mrrs10)
			hitm15, ndcgm15, precisionm15, mean_apm15, mrrm15 = np.mean(hits15), np.mean(ndcgs15), np.mean(precisions15), np.mean(mean_aps15), np.mean(mrrs15)
			print("\tK=5, HR: %.4f, NDCG: %.4f, Precision: %.4f, MAP: %.4f, MRR: %.4f" % (hitm5, ndcgm5, precisionm5, mean_apm5, mrrm5))
			print("\tK=10, HR: %.4f, NDCG: %.4f, Precision: %.4f, MAP: %.4f, MRR: %.4f" % (hitm10, ndcgm10, precisionm10, mean_apm10, mrrm10))
			print("\tK=15, HR: %.4f, NDCG: %.4f, Precision: %.4f, MAP: %.4f, MRR: %.4f, time: %.2fs" % (hitm15, ndcgm15, precisionm15, mean_apm15, mrrm15, time()-t1))
			hit5_list.append(hitm5), ndcg5_list.append(ndcgm5), precision5_list.append(precisionm5), mean_ap5_list.append(mean_apm5), mrr5_list.append(mrrm5)
			hit10_list.append(hitm10), ndcg10_list.append(ndcgm10), precision10_list.append(precisionm10), mean_ap10_list.append(mean_apm10), mrr10_list.append(mrrm10)
			hit15_list.append(hitm15), ndcg15_list.append(ndcgm15), precision15_list.append(precisionm15), mean_ap15_list.append(mean_apm15), mrr15_list.append(mrrm15)
		print("End. K=5, Best_HR: %.4f, Best_NDCG: %.4f, Best_Precision: %.4f, Best_MAP: %.4f, Best_MRR: %.4f" % (max(hit5_list), max(ndcg5_list), max(precision5_list), max(mean_ap5_list), max(mrr5_list)))
		print("End. K=10, Best_HR: %.4f, Best_NDCG: %.4f, Best_Precision: %.4f, Best_MAP: %.4f, Best_MRR: %.4f" % (max(hit10_list), max(ndcg10_list), max(precision10_list), max(mean_ap10_list), max(mrr10_list)))
		print("End. K=15, Best_HR: %.4f, Best_NDCG: %.4f, Best_Precision: %.4f, Best_MAP: %.4f, Best_MRR: %.4f" % (max(hit15_list), max(ndcg15_list), max(precision15_list), max(mean_ap15_list), max(mrr15_list)))

def eval_model(source_users, source_items, target_users, target_items, common_users, target_predict_rate, sess, source_user_input_test, source_item_input_test, source_negtivates_test, target_user_input_test, target_item_input_test, target_negtivates_test, common_user_input_test, topK_list):
	hits5, ndcgs5, precisions5, mean_aps5, mrrs5 = [], [], [], [], []
	hits10, ndcgs10, precisions10, mean_aps10, mrrs10 = [], [], [], [], []
	hits15, ndcgs15, precisions15, mean_aps15, mrrs15 = [], [], [], [], []
	for idx in range(len(target_user_input_test)):
		items = target_negtivates_test[idx]
		user = target_user_input_test[idx]
		gtItem = target_item_input_test[idx]
		items.append(gtItem)
		users = [user] * len(items)
		predict = sess.run(target_predict_rate, feed_dict={target_users: users, target_items: items, common_users: users})
		predictions = predict[:, 0]
		# print("precisions: ", precisions)
		map_item_score = {}
		for i in range(len(items)):
			item = items[i]
			map_item_score[item] = predictions[i]
		items.pop()

		total_ranklist = heapq.nlargest(len(items), map_item_score, key=map_item_score.get)
		for k in topK_list:
			if k == 5:
				hits5, ndcgs5, precisions5, mean_aps5, mrrs5 = eval_rank(hits5, ndcgs5, precisions5, mean_aps5, mrrs5, k, map_item_score, gtItem)
			if k == 10:
				hits10, ndcgs10, precisions10, mean_aps10, mrrs10 = eval_rank(hits10, ndcgs10, precisions10, mean_aps10, mrrs10, k, map_item_score, gtItem)
			if k == 15:
				hits15, ndcgs15, precisions15, mean_aps15, mrrs15 = eval_rank(hits15, ndcgs15, precisions15, mean_aps15, mrrs15, k, map_item_score, gtItem)
	return hits5, ndcgs5, precisions5, mean_aps5, mrrs5, hits10, ndcgs10, precisions10, mean_aps10, mrrs10, hits15, ndcgs15, precisions15, mean_aps15, mrrs15

def eval_rank(hits, ndcgs, precisions, mean_aps, mrrs, k, map_item_score, gtItem):
	ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
	hr = getHitRatio(ranklist, gtItem)
	ndcg = getNDCG(ranklist, gtItem)
	precision = getHitRatio(ranklist, gtItem) / k
	mean_ap = get_MAP(ranklist, gtItem)
	mrr = get_MRR(ranklist, gtItem)
	hits.append(hr)
	ndcgs.append(ndcg)   
	precisions.append(precision)
	mean_aps.append(mean_ap)
	mrrs.append(mrr)
	return hits, ndcgs, precisions, mean_aps, mrrs 

def getHitRatio(ranklist, gtItem):
	for item in ranklist:
		if item == gtItem:
			return 1
	return 0

def getNDCG(ranklist, gtItem):
	for i in range(len(ranklist)):
		item = ranklist[i]
		if item == gtItem:
			return math.log(2) / math.log(i+2)
	return 0

def get_MAP(ranklist, gtItem):
	precision = 0
	for i in range(len(ranklist)):
		precision += getHitRatio(ranklist[:(i+1)], gtItem) / (i+1)
	return precision / len(ranklist)

def get_MRR(ranklist, gtItem):
	for i in range(len(ranklist)):
		item = ranklist[i]
		if item == gtItem:
			return 1 / (i+1)
	return 0

if __name__ == "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	source_path = './Data/common_movies'
	target_path = './Data/common_office'
	topK_list = [5, 10, 15]
	num_factor = 16
	factor_layers = [128, 64, 32, 16]
	learn_rate = 0.001
	decay_rate = 0.96
	decay_steps = 30000
	batch_size = 256
	epochs = 200
	num_negatives = 2
	regularizer_rate = 0.0001
	source_gama = 0.4
	gan_gama = 0.5
	gamma = 10.0

	firTime = time()
	dataset = Dataset(source_path, target_path)
	source_trainDict, source_num_users, source_num_items, source_testRatings, source_testNegatives = dataset.source_user_ratingdict, dataset.source_user_num, dataset.source_item_num, dataset.source_testRatings, dataset.source_testNegatives
	target_trainDict, target_num_users, target_num_items, target_testRatings, target_testNegatives = dataset.target_user_ratingdict, dataset.target_user_num, dataset.target_item_num, dataset.target_testRatings, dataset.target_testNegatives
	secTime = time()

	print("train_user_grl load data time: %.2fs"%(secTime - firTime))
	print("Source user num: ", source_num_users)
	print("Source item num: ", source_num_items)
	print("Target user num: ", target_num_users)
	print("Target item num: ", target_num_items)

	train_model()
