module ai

import arrays

type Matrix = []Vec

pub fn new_matrix(n int, p int) Matrix {
	return Matrix([]Vec{len: n, init: new_vec(p)})
}

pub fn (m Matrix) vec() Vec {
	assert m[0].len == 1
	mut v := new_vec(m.len)
	for i in 0 .. m.len {
		v[i] = m[i][0]
	}

	return v
}

pub fn (tbl Matrix) is_empty() bool {
	for i in 0 .. tbl.len {
		for j in 0 .. tbl[i].len {
			if tbl[i][j] != 0 {
				return false
			}
		}
	}
	return true
}

pub fn (m Matrix) transpose() Matrix {
	mut t := new_matrix(m[0].len, m.len)
	for i in 0 .. m.len {
		for j in 0 .. m[i].len {
			t[j][i] = m[i][j]
		}
	}

	return t
}

pub fn (a Matrix) * (b Matrix) Matrix {
	n := a.len
	p := a[0].len
	q := b[0].len

	assert p == b.len, 'cannot multiply ${a.dim()} matrix with ${b.dim()} matrix'

	mut c := new_matrix(n, q)

	for i in 0 .. n {
		for k in 0 .. q {
			s := []f64{len: p, init: a[i][index] * b[index][k]}
			c[i][k] = arrays.sum(s) or {
				eprintln('cannot multiply empty matrix')
				continue
			}
		}
	}

	return c
}

pub fn (tbl Matrix) str_tbl() [][]string {
	mut str_tbl := [][]string{len: tbl.len, init: []string{len: tbl[index].len}}
	for i in 0 .. tbl.len {
		for j in 0 .. tbl[i].len {
			str_tbl[i][j] = tbl[i][j].str()
		}
	}

	return str_tbl
}

pub fn (tbl [][]string) f64_tbl() Matrix {
	mut f64_tbl := []Vec{len: tbl.len, init: Vec([]f64{len: tbl[index].len})}

	for i in 0 .. tbl.len {
		for j in 0 .. tbl[i].len {
			f64_tbl[i][j] = tbl[i][j].f64()
		}
	}

	return Matrix(f64_tbl)
}

pub fn (m Matrix) dim() (int, int) {
	return m.len, m[0].len
}
