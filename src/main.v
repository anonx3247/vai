module main

import ai
import encoding.csv
import os

fn main() {
	// open datasets

	training_set, test_set := get_datasets('train.csv', 'test.csv')

	net_location := 'writer'

	exists := os.exists(net_location)

	mut net := if !exists { ai.new_model(
			name: 'Writer'
			layers: [
				ai.Layer{ai.relu, 64},
				ai.Layer{ai.sigmoid, 50},
				ai.Layer{ai.sigmoid, 10},
			]
		) } else { ai.net_from_file(net_location)! }
	net.randomize_weights()

	losses := net.train(
		training: training_set
		testing: test_set
		learning_rate: 0.01
		batch_size: 1000
	)

	net.save(net_location, true)! // true means it will overwrite
	println(net.run(t)!)
}

fn get_datasets(train string, test string) (ai.Dataset, ai.Dataset) {
	mut dataset_train := os.read_file(train)!
	mut dataset_test := os.read_file(test)!

	mut train_reader := csv.new_reader(dataset_train)
	mut test_reader := csv.new_reader(dataset_test)

	mut raw_train := [][]string{}
	mut raw_test := [][]string{}

	for {
		raw_train << train_reader.read() or { break }
	}

	for {
		raw_test << test_reader.read() or { break }
	}

	mut train_d := raw_train.f64_tbl()
	mut test_d := raw_test.f64_tbl()

	// code specific to the MNIST datasets preprocessing

	mut train_x := []ai.Vec{}
	mut train_y := []ai.Vec{}
	mut test_x := []ai.Vec{}
	mut test_y := []ai.Vec{}

	// TODO: remove ai.f64
	val_to_vec := fn (x f64) ai.Vec {
		c := x.int()

		return ai.Vec([]ai.f64{len: 10, init: if index == c {
			1
		} else {
			0
		}})
	}

	// the data is stored with the output in column 1 and the input in the other columns
	for j in 0 .. train_d[0].len {
		train_y << val_to_vec(train_d[0][j])
		train_x << ai.Vec([]ai.f64{len: train_d.len - 1, init: train_d[index + 1][j]})
	}
	for j in 0 .. test_d[0].len {
		test_y << val_to_vec(test_d[0][j])
		test_x << ai.Vec([]ai.f64{len: test_d.len - 1, init: test_d[index + 1][j]})
	}
}
