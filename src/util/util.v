module util

pub fn sort_by_index[T](list []T, indexer fn (T) !int) ![]T {
	mut new := []T{len: list.len}
	for elem in list {
		index := indexer(elem)!
		assert index <= list.len && index >= 0, 'index out of range: ${index} not in 0..${list.len}'
		new[index] = elem
	}
	return new
}

/*
implement quicksort algorithm using the 'comparer' function as
the with which to compare elements
*/

pub fn sort[T](list []T, comparer ?fn (x T, y T) bool) []T {
	op := if comparer is none {
		fn (x T, y T) bool {
			if x < y {
				return true
			} else {
				return false
			}
		}
	} else {
		comparer
	}

	if list.len <= 1 {
		return list
	}

	pivot := list[0]

	mut small := []T{}
	mut big := []T{}

	for elem in list {
		if op(elem, pivot) {
			small << elem
		} else {
			big << elem
		}
	}

	mut res := []T{}

	res << small.sort(op)
	res << big.sort(op)

	return res
}
