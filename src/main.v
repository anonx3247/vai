module main

import ai

fn main() {
	/*
	mut net := ai.new_network(
		name: 'Skynet'
		input_size: 64
		output_size: 10
		inner_size: 64
	)
	*/

	mut net := ai.new_model(
		name: 'Skynet'
		layers: [
			ai.Layer{ai.relu, 64},
			ai.Layer{ai.sigmoid, 50},
			ai.Layer{ai.sigmoid, 10},
		]
	)

	net.randomize_weights()
	// net.save('skynet')!

	t := ai.random_vec(64)

	println(net.run(t)!)
}
