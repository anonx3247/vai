module util

pub fn (tbl [][]f64) str_tbl() [][]string {
	mut str_tbl := [][]string{len: tbl.len, init: []string{len: tbl[index].len}}
	for i in 0 .. tbl.len {
		for j in 0 .. tbl[i].len {
			str_tbl[i][j] = tbl[i][j].str()
		}
	}

	return str_tbl
}

pub fn (tbl [][]string) f64_tbl() [][]f64 {
	mut f64_tbl := [][]f64{len: tbl.len, init: []f64{len: tbl[index].len}}

	for i in 0 .. tbl.len {
		for j in 0 .. tbl[i].len {
			f64_tbl[i][j] = tbl[i][j].f64()
		}
	}

	return f64_tbl
}

// inverts table so [i,j] = [j,i]

pub fn invert_tbl(tbl [][]f64) [][]f64 {
	mut inverted := [][]f64{len: tbl[0].len, init: []f64{len: tbl.len}}

	for i in 0 .. tbl[0].len {
		for j in 0 .. tbl.len {
			inverted[i][j] = tbl[j][i]
		}
	}

	return inverted
}

pub fn (tbl [][]f64) is_empty() bool {
	for i in 0 .. tbl.len {
		for j in 0 .. tbl[i].len {
			if tbl[i][j] != 0 {
				return false
			}
		}
	}
	return true
}

pub fn sort[T](list []T, indexer fn (T) !int) ![]T {
	mut new := []T{len: list.len}
	for elem in list {
		index := indexer(elem)!
		assert index <= list.len && index >= 0, 'index out of range: ${index} not in 0..${list.len}'
		new[index] = elem
	}
	return new
}
