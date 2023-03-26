module ai

struct Neuron {
mut:
	value  f64
	inputs []Neuron
__global:
	input_weights []f64
}

fn (mut n Neuron) eval(act_fn Func) {
	mut s := 0.0
	for i in 0 .. n.inputs.len {
		input_value := n.inputs[i].value
		s += input_value * n.input_weights[i]
	}
	n.value = act_fn(s)
}
