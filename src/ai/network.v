module ai

import arrays

pub struct Network {
	loss_fn LossFn
mut:
	name           string
	weights        []Matrix
	layers         []Vec
	activation_fns []ActivationFn
}

[params]
pub struct NetworkDefinition {
	name              string       [required]
	input_size        int          [required]
	output_size       int          [required]
	inner_layer_count int = 1
	input_fn          ActivationFn = sigmoid
	output_fn         ActivationFn = sigmoid
	inner_fn          ActivationFn = sigmoid
	inner_size        int
	loss_fn           LossFn = logloss
}

[params]
pub struct ModelDefinition {
	name    string
	layers  []Layer
	loss_fn LossFn
}

pub struct Layer {
	activation_fn ActivationFn
	size          int
}

pub fn new_network(def NetworkDefinition) Network {
	// form the layers
	mut input_layer := new_vec(def.input_size)
	mut outer_layer := new_vec(def.output_size)
	mut inner_layers := []Vec{len: def.inner_layer_count, init: new_vec(def.inner_size)}
	mut inner_layer_fns := []ActivationFn{len: def.inner_layer_count, init: def.inner_fn}

	// join all layers
	mut layers := [input_layer]
	layers << inner_layers
	layers << outer_layer

	mut fns := [def.input_fn]
	fns << inner_layer_fns
	fns << def.output_fn

	weights := new_weights(layers)

	return Network{
		name: def.name
		weights: weights
		loss_fn: def.loss_fn
		layers: layers
		activation_fns: fns
	}
}

fn new_weights(layers []Vec) []Matrix {
	return []Matrix{len: layers.len - 1, init: new_matrix(layers[index + 1].len, layers[index].len)}
}

pub fn new_model(mod ModelDefinition) Network {
	mut real_layers := []Vec{len: mod.layers.len, init: new_vec(mod.layers[index].size)}
	fns := []ActivationFn{len: mod.layers.len, init: mod.layers[index].activation_fn}
	weights := new_weights(real_layers)
	return Network{
		name: mod.name
		weights: weights
		loss_fn: mod.loss_fn
		layers: real_layers
		activation_fns: fns
	}
}

pub fn (mut n Network) rename(name string) {
	n.name = name
}

// runs an input through the network and outputs the value returned
pub fn (n Network) run(input Vec) !Vec {
	if input.len != n.layers[0].len {
		return error('error, incorrect input length')
	}
	for i, mut neuron in n.layers[0] {
		neuron = input[i]
	}

	for i, mut layer in n.layers[1..] {
		layer = n.activation_fns[i].call((n.weights[i] * n.layers[i].mat()).vec())
	}

	return n.layers[n.layers.len - 1]
}

// returns the loss on the previous run, comparing to the y prediction
pub fn (n Network) loss(y Vec) !Vec {
	output := n.layers[n.layers.len - 1]
	if output.len != y.len {
		return error('unmatching prediction and test lengths')
	}
	return n.loss_fn.call(output, y)
}

// returns average loss on all coordinates of the output array
pub fn (n Network) meanloss(y Vec) !f64 {
	loss := n.loss(y)!

	s := arrays.sum[f64](loss)!
	return s / loss.len
}

pub fn (mut n Network) randomize_weights() {
	for i, mut layer in n.layers[1..] {
		for j, _ in layer {
			n.weights[i][j] = random_vec(n.layers[i].len)
		}
	}
}

/*
there are some errors in the data storage process because of float imprecisions,
	this gets a list of all changed values and returns the deltas which are usually around 1e-18 in size
*/
pub fn diff(n1 Network, n2 Network) Vec {
	assert n1.layers.len == n2.layers.len

	mut diffs := []f64{}

	for i, _ in n1.layers[1..] {
		println('dimensions: ${n1.weights[i].dim()}, ${n2.weights[i].dim()}')
		n, _ := n1.weights[i].dim()
		for j in 0 .. n {
			assert n1.weights[i][j].len == n2.weights[i][j].len
			for k, weight in n1.weights[i][j] {
				if weight != n2.weights[i][j][k] {
					diffs << weight - n2.weights[i][j][k]
				}
			}
		}
	}
	return diffs
}
