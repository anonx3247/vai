module ai

import os
import encoding.csv
import json
import arrays
import util
import strconv
import szip

struct NetworkParameters {
	name           string
	loss_fn        string
	activation_fns []string
}

/*
Stores neural network as a zipped file in the directory `location`

	The folder contains
	- a csv holding the values of the weights for each inter-layer
	- a `params.json` file with other parameters such as activation functions and the network name
*/
pub fn (n Network) save(location string) ! {
	// open a new temporary directory
	dir := os.vtmp_dir()

	mut files := []string{}

	// folder to be compressed
	os.mkdir(location)!

	// store weights as multiple csv files
	weights := n.weights()

	for i in 0 .. weights.len {
		os.create(dir + '/${i}.csv')!
		mut writer := csv.new_writer(use_crlf: false)
		for line in weights[i].str_tbl() {
			writer.write(line)!
		}
		os.write_file(dir + '/${i}.csv', writer.str())!
		files << dir + '/${i}.csv'
	}

	// store other net params in json file
	os.create(dir + '/params.json')!
	s := json.encode(n.params())
	os.write_file(dir + '/params.json', s)!
	files << dir + '/params.json'

	szip.zip_files(files, location + '/${n.name}.net')!
}

// Utility function returning name, loss_fn, and the activation fns of a network
fn (n Network) params() NetworkParameters {
	activ := []string{len: n.layers.len, init: n.layers[index].activation_fn.str()}
	loss := if n.loss_fn == logloss { 'logloss' } else { 'square_mean' }
	return NetworkParameters{
		name: n.name
		loss_fn: loss
		activation_fns: activ
	}
}

// Extracts data from a zipped stored network (created with Network.save()) and rebuilds the network
pub fn net_from_file(filename string) !Network {
	// create a temporary directory for extracting the network contents
	path := os.vtmp_dir()
	szip.extract_zip_to_dir(filename, path)!
	mut paths := os.ls(path)!

	paths = arrays.filter_indexed(paths, fn (idx int, elem string) bool {
		if elem.len < 4 {
			return false
		} else if elem.len >= 4 && elem[elem.len - 4..] == '.csv' {
			return true
		} else if elem.len >= 5 {
			if elem[elem.len - 5..] == '.json' {
				return true
			} else {
				return false
			}
		} else {
			return false
		}
	})
	println(paths)

	params_index := arrays.binary_search(paths, 'params.json') or {
		return error('params file not found')
	}
	params := os.read_file(path + '/' + paths[params_index]) or {
		return error('unable to read params file')
	}

	// remove params from the index
	paths.delete(params_index)
	// sort ascendingly the csvs
	paths = util.sort[string](paths, fn (elem string) !int {
		assert elem.len >= 4
		return strconv.atoi(elem[..elem.len - 4])!
		/*
		mut it := 0
		// find the first digit
		for {
			if it > elem.len {
				return error('unable to locate any digit in filename')
			}
			strconv.atoi(elem[it..it + 1]) or {
				it++
				continue
			}
			break
		}

		// remove the last 4 chars '.csv'
		return strconv.atoi(elem[it..elem.len - 4])!
		*/
	})!

	for mut p in paths {
		p = path + '/' + p
	}

	weights := [][][]f64{len: paths.len, init: weights_from_csv(paths[index])!}

	parameters := json.decode(NetworkParameters, params) or {
		return error('unable to decode parameters, ${err}')
	}

	// there are n activation functions for n layers
	// but there are only n-1 csv files since weights are between layers
	if parameters.activation_fns.len != paths.len + 1 {
		return error('missing layers, expected ${parameters.activation_fns.len}, got ${paths.len + 1}\npaths=${paths}')
	}

	mut layers := []Layer{len: paths.len, init: new_layer(parameters.activation_fns[index + 1],
		weights[index])!}

	layers.prepend(new_layer(parameters.activation_fns[0], [][]f64{len: layers[0].neurons[0].input_weights.len}) or {
		return error('unable to create first layer')
	})

	layers.connect()

	return Network{
		name: parameters.name
		loss_fn: if parameters.loss_fn == 'logloss' { logloss } else { square_mean }
		layers: layers
	}
}

// Utility fn, returns weights table from a csv
fn weights_from_csv(path string) ![][]f64 {
	text := os.read_file(path) or { return error('unable to read file ${path}') }

	mut raw := [][]string{}

	mut reader := csv.new_reader(text)

	for {
		raw << reader.read() or { break }
	}

	return raw.f64_tbl()
}
