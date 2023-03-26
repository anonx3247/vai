module ai

import rand
import arrays

[noinit]
struct Network {
	loss_fn fn ([]f64, []f64) []f64
mut:
	name   string
	layers []Layer
}

[params]
pub struct NetworkDefinition {
	name              string                  [required]
	input_size        int                     [required]
	output_size       int                     [required]
	inner_layer_count int  = 1
	input_fn          Func = sigmoid
	output_fn         Func = sigmoid
	inner_fn          Func = sigmoid
	inner_size        int
	loss_fn           fn ([]f64, []f64) []f64 = logloss
}

// Network Initialization

pub fn new_network(def NetworkDefinition) Network {
	// form the layers
	mut input_layer := Layer{
		activation_fn: def.input_fn
		neurons: []Neuron{len: def.input_size}
	}

	mut inner_layers := []Layer{len: def.inner_layer_count}

	for mut layer in inner_layers {
		layer = Layer{
			activation_fn: def.inner_fn
			neurons: []Neuron{len: def.inner_size}
		}
	}

	mut output_layer := Layer{
		activation_fn: def.output_fn
		neurons: []Neuron{len: def.output_size}
	}

	// join all layers
	mut layers := [input_layer]
	layers << inner_layers
	layers << output_layer

	// connect layers
	layers.connect()

	return Network{
		name: def.name
		loss_fn: def.loss_fn
		layers: layers
	}
}

pub fn (mut n Network) rename(name string) {
	n.name = name
}

// runs an input through the network and outputs the value returned
pub fn (n Network) run(input []f64) ![]f64 {
	if input.len != n.layers[0].neurons.len {
		return error('error, incorrect input length')
	}
	for i, mut neuron in n.layers[0].neurons {
		neuron.value = input[i]
	}

	for mut layer in n.layers[1..] {
		layer.run()
	}

	return layer_to_array(n.layers[n.layers.len - 1])
}

// returns the loss on the previous run, comparing to the y prediction
pub fn (n Network) loss(y []f64) ![]f64 {
	output := layer_to_array(n.layers[n.layers.len - 1])
	if output.len != y.len {
		return error('unmatching prediction and test lengths')
	}
	return n.loss_fn(output, y)
}

// returns average loss on all coordinates of the output array
pub fn (n Network) meanloss(y []f64) !f64 {
	loss := n.loss(y)!

	s := arrays.sum[f64](loss.data)!
	return s / loss.len
}

/*
returns a list of tables of weights where:

	weights[i] is the table representing the weights connecting layer i-1 to layer i

	and weights[i][j] is the list of weights of neurons in layer i-1 connected to neuron j in layer i
*/
fn (n Network) weights() [][][]f64 {
	mut w := [][][]f64{len: n.layers.len - 1, init: [][]f64{}}

	for i, layer in n.layers[1..] {
		w[i] = [][]f64{len: layer.neurons.len}
		for j, neuron in layer.neurons {
			for weight in neuron.input_weights {
				w[i][j] << weight
			}
		}
	}

	return w
}

// sets random values for all the weights of a network
pub fn (mut n Network) randomize_weights() {
	for mut layer in n.layers {
		for mut neuron in layer.neurons {
			for mut weight in neuron.input_weights {
				weight = rand.f64()
			}
		}
	}
}

/*
there are some errors in the data storage process because of float imprecisions,
	this gets a list of all changed values and returns the deltas which are usually around 1e-18 in size
*/
pub fn find_diff(n1 Network, n2 Network) []f64 {
	assert n1.layers.len == n2.layers.len

	mut diffs := []f64{}

	for i, layer in n1.layers {
		assert layer.neurons.len == n2.layers[i].neurons.len
		for j, neuron in layer.neurons {
			assert neuron.input_weights.len == n2.layers[i].neurons[j].input_weights.len
			for k, weight in neuron.input_weights {
				if weight != n2.layers[i].neurons[j].input_weights[k] {
					diffs << weight - n2.layers[i].neurons[j].input_weights[k]
				}
			}
		}
	}
	return diffs
}
