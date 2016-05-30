import random
import math


class Neuron(object):
	def __init__(self, prev_n_neurons):
		self._synapticWeights = []
		for i in range(prev_n_neurons):
			self._synapticWeights.append(random.random() - 0.5)
		self.lamda = 1.5
		self._activation = 0.0

	def activate(self, inputs):
		self._activation = 0.0
		for i in range(len(inputs)):
			self._activation += inputs[i] * _synapticWeights[i]
		return 2.0 / (1.0 + float(math.exp(-_activation * lamda))) - 1.0

	def getActivationDerivative(self):
		expmlx = float(math.exp(lamda * _activation))
		return 2 * lamda * expmlx / ((1 + expmlx) * (1 + expmlx))

	def getSynapticWeights(self):
		return self._synapticWeights

	def getSynapticWeight(self, i):
		return self._synapticWeights[i]

	def setSynapticWeight(self, i, v):
		self._synapticWeights[i] = v


class Layer(object):
	def __init__(self, prev_n_neurons, n_neurons):
		self._n_neurons = n_neurons + 1
		self._prev_n_neurons = prev_n_neurons + 1
		self._neurons = []
		self._outputs = []

		for i in range(self._n_neurons):
			self._neurons.append(Neuron(self._prev_n_neurons))

	@staticmethod
	def add_bias(inp):
		outp = [1.0]
		for i in range(len(inp)):
			outp.append(inp[i])
		return outp

	def evaluate(self, inp):
		inputs = []
		if len(inp) != len(_neurons[0]._synapticWeights):
			inputs = add_bias(inp)
		else:
			inputs = inp
		for i in range(_n_neurons):
			if i > 0:
				_outputs[i] = _neurons[i].activate(inputs)
		_outputs[0] = 1.0

		return _outputs
	def size(self):
		return _n_neurons
	def getOutput(self, i):
		return _outputs[i]
	def getActivationDerivative(self, i):
		return _neurons[i].getActivationDerivative()
	def getWeight(self, i, j):
		return self._neurons[i].getSynapticWeight(j)
	def getWeights(self, i):
		return self._neurons[i].getSynapticWeights()
	def setWeight(self, i, j, v):
		self._neurons[i].setSynapticWeight(j, v)


class Mlp(object):
	def __init__(self, nn_neurons):
		self._layers = []
		for i in range(len(nn_neurons)):
			self._layers.append(Layer(nn_neurons[i] if i == 0 else nn_neurons[i-1], nn_neurons[i]))
		self._delta_w = []
		for i in range(len(nn_neurons)):
			self._delta_w.append([[None] * len(self._layers[i].getWeights(0))] * len(self._layers[i]))
		self._grad_ex = []
		for i in range(len(nn_neurons)):
			self._grad_ex.append([None] * _layers[i].size())
	def evaluate(self, inputs):
		outputs = [None] * len(inputs)
		for i in range(len(_layers)):
			outputs = _layers[i].evaluate(inputs)
			inputs = outputs
		return outputs

	def evaluateError(self, nn_output, desired_output):
		d = []
		if len(desired_output) != len(nn_output):
			d = Layer.add_bias(desired_output)
		else:
			d = desired_output
		e = 0.0
		for i in range(len(nn_output)):
			e += (nn_output[i] - d[i]) * (nn_output[i] - d[i])
		return e

	def evaluateQuadraticError(self, examples, results):
		e = 0.0
		for i in range(len(examples)):
			e += evaluateError(evaluate(examples[i]), results[i])
		return e

	def evaluateGradients(self, results):
		c = len(_layers) - 1
		while c >= 0:
			for i in range(len(_layers[c])):
				if c == len(_layers) - 1:
					_grad_ex[c][i] = 2 * (_layers[c].getOutput(i) - results[0]) * (_layers[c].getActivationDerivative(i))
				else:
					sum = 0.0
					for k in range(len(_layers[c+1])):
						if k > 0:
							sum += _layers[c+1].getWeight(k, i) * _grad_ex[c+1][k]
					_grad_ex[c][i] = _layers[c].getActivationDerivative(i) * sum
			c = c - 1

	def resetWeightsDelta(self):
		for c in range(len(_layers)):
			for i in range(len(_layers[c])):
				weights = _layers[c].getWeights[i]
				for j in range(len(weights)):
					_delta_w[c][i][j] = 0

	def evaluateWeightsDelta(self):
		for c in range(len(_layers)):
			if c > 0:
				for i in range(len(_layers[c])):
					weights = _layers[c].getWeights(i)
					for j in range(len(weights)):
						_delta_w[c][i][j] += _grad_ex[c][i] * _layers[c-1].getOutput(j)

	def updateWeights(self, learning_rate):
		for c in range(len(_layers)):
			for i in range(len(_layers[c])):
				weights = _layers[c].getWeights(i)
				for j in range(len(weights)):
					_layers[c].setWeight(i, j, _layers[c].getWeight(i, j) - (learning_rate * _delta_w[c][i][j]))

	def batchBackPropagation(self, examples, results, learning_rate):
		resetWeightsDelta()
		for l in range(len(examples)):
			evaluate(examples[l])
			evaluateGradients(results[l])
			evaluateWeightsDelta()
		updateWeights(learning_rate)

	def learn(self, examples, results, learning_rate):
		e = 10000000000.0
		while e > 0.001:
			batchBackPropagation(examples, results, learning_rate)
			e = evaluateQuadraticError(examples, results)


ex = [[-1,1], [1,1], [1,-1], [-1,-1]]
res = [1, -1, 1, -1]
nn = [len(ex[0]), len(ex[0]) * 3]
mlp = Mlp(nn)
for i in range(40000):
	mlp.learn(ex, res, 0.3)
	err = mlp.evaluateQuadraticError(ex, res)
	print(i, "->error: ", err)	
