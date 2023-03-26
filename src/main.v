module main

import ai

fn main() {
	mut net := ai.new_network(
		name: 'Skynet'
		input_size: 64
		output_size: 10
		inner_size: 64
	)

	net.randomize_weights()

	t := ai.random_array(64)

	println('input: ${t}')

	res := net.run(t)!

	y := ai.random_array(10)

	loss := net.loss(y)!

	println('res: ${res}')
	println('loss: ${loss}')
	println('meanloss ${net.meanloss(y)!}')
}
