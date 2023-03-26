module ai

import rand

fn layer_to_array(l Layer) []f64 {
	return []f64{len: l.neurons.len, init: l.neurons[index].value}
}

pub fn random_array(len int) []f64 {
	mut arr := []f64{len: len}
	for mut e in arr {
		e = rand.f64()
	}
	return arr
}
