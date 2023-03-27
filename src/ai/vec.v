module ai

import rand

type Vec = []f64

pub fn random_vec(len int) Vec {
	mut arr := []f64{len: len}
	for mut e in arr {
		e = rand.f64()
	}
	return arr
}

pub fn new_vec(n int) Vec {
	return Vec([]f64{len: n})
}

pub fn (v1 Vec) * (v2 Vec) Vec {
	assert v1.len == v2.len, 'cannot multiply Vec[${v1.len}] by Vec[${v2.len}]'
	return Vec([]f64{len: v1.len, init: v1[index] * v2[index]})
}

pub fn (v1 Vec) + (v2 Vec) Vec {
	assert v1.len == v2.len, 'cannot add Vec[${v1.len}] by Vec[${v2.len}]'
	return Vec([]f64{len: v1.len, init: v1[index] + v2[index]})
}

pub fn (v1 Vec) - (v2 Vec) Vec {
	assert v1.len == v2.len, 'cannot sub Vec[${v1.len}] by Vec[${v2.len}]'
	return Vec([]f64{len: v1.len, init: v1[index] - v2[index]})
}

pub fn (v1 Vec) / (v2 Vec) Vec {
	assert v1.len == v2.len, 'cannot divide Vec[${v1.len}] by Vec[${v2.len}]'
	return Vec([]f64{len: v1.len, init: v1[index] / v2[index]})
}

pub fn unit(n int) Vec {
	return Vec([]f64{len: n, init: 1.0})
}

pub fn (v Vec) mat() Matrix {
	mut m := new_matrix(v.len, 1)

	for i, e in v {
		m[i][0] = e
	}

	return m
}
