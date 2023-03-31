module ai

import arrays

/*
Implementation of the backpropagation algorithm as described here:
https://en.wikipedia.org/wiki/Backpropagation#u.Matrix_multiplication

This algorithm works in the following way:

1. We calculate the partial gradient at layer l denoted delta(l)
2. This is done using the following recursive formula:

delta(l-1) = f_l' * W_(l+1)^T . ... . f_(L-1)' * W_L^T . f_L' grad(C, f_L)


- where f_i is the activation function at layer i
- W_k is the weight matrix of coefficients (i,j) where W_k(i,j) = the weight multiplying the input from neuron j in layer k-1 as output to neuron i in layer k
- ^T is the transpose operation
- C is the cost function

3. Once the deltas have been calculated, we can determine dC/dW_k:

dC/dW_k =delta(k) * f_(k-1)^T


4. Finally given a learning rate eta, we update each weight following the formula:

W_k(i,j) += -eta * dC/dW_k(i,j)


Note: here we have the Cost function evaluated at x comparing to the expected result y:

C(y, f_L(W_L * f_(L-1) (W_(L-1) ... f_1(W_1 * x)))
*/

fn (n Network) delta(y Vec, l int, mut deltas map[int]Vec) !Vec {
	d := Vec([]f64{})

	if e := deltas[l] {
		return e
	} else if l == n.layers.len - 1 {
		d = n.activation_fns[l].dcall(n.layers[l]) * n.loss_fn.dcall(n.layers[l], y)
	} else if 0 <= l <= n.layers.len {
		d = (n.activation_fns[l].dcall(n.layers[l]) * n.weights[l].transpose() * delta(l + 1).mat()).vec()
	} else {
		return error('cannot calculate delta for index ${l}')
	}

	deltas[l] = d
	return d
}

fn (mut n Network) update_weights(y Vec, learning_rate f64) ! {
	mut deltas := []Vec{len: n.layers.len}

	mut dW := []Matrix{len: n.layers.len}

	for i, w in n.weights {
		dW[i] = delta(y, i, deltas)!
	}

	for k, mut dw in dW {
		for i in 0 .. dw.len {
			for j in 0 .. dw[0].len {
				n.weights[k][i][j] -= learning_rate * dw[i][j]
			}
		}
	}
}

pub fn (mut n Network) fit(x []Vec, y []Vec, learning_rate f64) ! {
	assert x.len == y.len
	assert x[0].len == n.layers[0].len
	assert y[0].len == n.layers[n.layers.len - 1].len

	for i in 0 .. x.len {
		n.run(x[i])!
		n.update_weights(y[i], learning_rate)!
	}
}

pub fn (mut n Network) test(x []Vec, y []Vec) !f64 {
	assert x.len == y.len
	assert x[0].len == n.layers[0].len
	assert y[0].len == n.layers[n.layers.len - 1].len

	mut losses := []f64{}

	mean := fn (v Vec) f64 {
		return arrays.sum[f64](v) / v.len
	}

	for i in 0 .. x.len {
		n.run(x[i])!
		losses << mean(n.loss_fn.call(n.layers[l], y[i]))
	}

	return mean(losses)
}

[params]
pub struct TrainingParams {
	training      Dataset
	testing       Dataset
	learning_rate f64
	batch_size    int = 1000
}

pub fn (mut n Network) train(p TrainingParams) !Vec {
	x_train := split_data(p.training.inputs, p.batch_size)
	y_train := split_data(p.training.outputs, p.batch_size)

	mut losses := []f64{}
	for i in 0 .. x_train.len {
		n.fit(x_train[i], y_train[i], learning_rate)!
		losses << n.test(p.testing.inputs[i], p.testing.outputs[i])!
		print('loss at iteration ${i * batch_size}: ${losses[losses.len - 1]}')
	}

	return Vec(losses)
}

fn split_data(data []Vec, size int) [][]Vec {
	if size >= data.len {
		return [data]
	} else {
		mut k := size
		mut split := [][]Vec{}
		for k < data.len {
			split << data[(k - size)..k]
		}

		split << data[k..]

		return split
	}
}
