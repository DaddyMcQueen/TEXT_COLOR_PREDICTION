import numpy as np
import random
from constants import LEARNING_RATE

class NeuralNetwork(object):

	def __init__(self, in_nodes, hidden_nodes, out_nodes):
		self.in_nodes = in_nodes
		self.hidden_nodes = hidden_nodes
		self.out_nodes = out_nodes
		self.create_weights()
		self.create_biases()

	def create_weights(self):
		np.random.seed(0)
		self.wih = 0.1 * np.random.randn(self.in_nodes, self.hidden_nodes)
		self.who = 0.1 * np.random.randn(self.hidden_nodes, self.out_nodes)

	def create_biases(self):
		self.hb = np.zeros((1, self.hidden_nodes))
		self.ob = np.zeros((1, self.out_nodes))

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def d_sigmoid(self, x):
		return x * (1 - x)

	def forward_pass(self, inputs):

		self.l0 = inputs.reshape(1, 3)
		self.l1 = np.dot(inputs, self.wih) + self.hb
		self.a1 = self.sigmoid(self.l1)
		self.l2 = np.dot(self.l1, self.who) + self.ob
		self.a2 = self.sigmoid(self.l2)
		# print(self.l2)
		return self.a2

	def get_error(self):
		avg_rgb = np.sum(self.l0, axis=1)
		if avg_rgb > 1.20:
			predicted_out = 1
		else:
			predicted_out = 0
		cost = (self.a2 - predicted_out) * 2
		# print(cost)
		return cost

	def back_prop(self, error):

		delta_ob = error * self.d_sigmoid(self.a2)
		delta_who = np.dot(self.a1.T, delta_ob)

		delta_hb = np.dot(self.who, delta_ob.T).T * self.d_sigmoid(self.a1)
		delta_wih = np.dot(self.l0.T, delta_hb)

		self.ob -= delta_ob
		self.who -=  LEARNING_RATE * delta_who
		self.hb -= delta_hb
		self.wih -=  LEARNING_RATE * delta_wih

	def train(self, inputs):
		output = self.forward_pass(inputs)
		error = self.get_error()
		self.back_prop(error)

	def test_nn(self, rgb):
		test_data = np.asarray(rgb)
		output = self.forward_pass(test_data)
		return output
