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
	net.save('skynet')!

	// t := ai.random_vec(64)

	mut net2 := ai.net_from_file('skynet/Skynet.net')!

	net2.rename('skynet2')

	println(ai.diff(net, net2))
}
