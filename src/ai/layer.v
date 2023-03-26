module ai

struct Layer {
	activation_fn Func
mut:
	neurons []Neuron
}

fn (mut l Layer) run() {
	for mut neuron in l.neurons {
		neuron.eval(l.activation_fn)
	}
}

fn new_layer(func string, weights [][]f64) !Layer {
	if func !in ['sigmoid', 'tanh', 'relu'] {
		return error('unrecognized function')
	}

	// inverted_weights := util.invert_tbl(weights)

	activ := match func {
		'sigmoid' {
			sigmoid
		}
		'tanh' {
			tanh
		}
		'relu' {
			relu
		}
		else {
			sigmoid
		}
	}

	mut neurons := []Neuron{len: weights.len}

	for i, mut neuron in neurons {
		neuron = Neuron{
			value: 0.0
			inputs: []Neuron{len: weights.len}
			input_weights: weights[i]
		}
	}

	return Layer{
		activation_fn: activ
		neurons: neurons
	}
}

fn (layers []Layer) connect() {
	for i in 1 .. layers.len {
		for mut neuron in layers[i].neurons {
			neuron.inputs = layers[i - 1].neurons
			if neuron.input_weights.len == 0 {
				neuron.input_weights = []f64{len: neuron.inputs.len}
			}
		}
	}
}
