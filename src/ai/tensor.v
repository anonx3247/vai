module ai

import rand

[noinit]
struct Tensor {
	len  int   [required]
	data []f64
}

pub fn new_tensor(len int) Tensor {
	return Tensor{
		len: len
		data: []f64{len: len}
	}
}

fn array_to_tensor(arr []f64) Tensor {
	return Tensor{
		len: arr.len
		data: arr
	}
}

fn layer_to_tensor(l Layer) Tensor {
	mut arr := []f64{len: l.neurons.len}

	for i, neuron in l.neurons {
		arr[i] = neuron.value
	}

	return array_to_tensor(arr)
}

pub fn random_tensor(len int) Tensor {
	mut arr := []f64{len: len}
	for mut e in arr {
		e = rand.f64()
	}
	return array_to_tensor(arr)
}
