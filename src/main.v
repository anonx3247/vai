module main

import ai

fn main() {
	mut net := ai.new_network(
		name: 'Skynet'
		input_spec: ai.new_tensor(64)
		output_spec: ai.new_tensor(10)
		inner_size: 64
	)

	net.randomize_weights()

	t := ai.random_tensor(64)

	println('input: ${t}')

	res := net.run(t)!

	y := ai.random_tensor(10)

	loss := net.loss(y)!

	println('res: ${res}')
	println('loss: ${loss}')
	println('meanloss ${net.meanloss(y)!}')
}
