module ai

import math

struct ActivationFn {
	self  fn (Vec) Vec
	deriv fn (Vec) Vec
	rep   string
}

struct LossFn {
	self fn (x Vec, y Vec) Vec
	rep  string
}

fn def_relu(x Vec) Vec {
	mut y := new_vec(x.len)

	for i, mut yi in y {
		if x[i] < 0 {
			yi = 0.0
		} else if x[i] >= 0 && x[i] <= 1 {
			yi = x
		} else {
			yi = 1.0
		}
	}
	return y
}

fn def_relup(x Vec) Vec {
	return []f64{len: x.len, init: if x[index] < 0 {
		0.0
	} else {
		1.0
	}}
}

fn def_sigmoid(x Vec) Vec {
	return []f64{len: x.len, init: 1 / (1 + math.exp(-x[index]))}
}

fn def_sigmoidp(x Vec) Vec {
	return def_sigmoid(x) * (unit(x.len) - def_sigmoid(x))
}

fn def_tanh(x Vec) Vec {
	return []f64{len: x.len, init: (math.exp(x[index]) - math.exp(-x[index])) / (
		math.exp(x[index]) + math.exp(-x[index]))}
}

fn def_tanhp(x Vec) Vec {
	return unit(x.len) - def_tanh(x) * def_tanh(x)
}

fn (f ActivationFn) str() string {
	return f.rep
}

fn (f ActivationFn) call(x Vec) Vec {
	return f.self(x)
}

fn (f ActivationFn) dcall(x Vec) Vec {
	return f.deriv(x)
}

fn def_logloss(pred Vec, real Vec) Vec {
	return []f64{len: pred.len, init: -(real[index] * math.log(pred[index]) +
		(1 - real[index]) * math.log(1 - pred[index]))}
}

fn (l LossFn) str() string {
	return l.rep
}

fn (l LossFn) call(x Vec, y Vec) Vec {
	return l.self(x, y)
}

const (
	sigmoid = ActivationFn{
		self: def_sigmoid
		deriv: def_sigmoidp
		rep: 'sigmoid'
	}
	tanh = ActivationFn{
		self: def_tanh
		deriv: def_tanhp
		rep: 'tanh'
	}
	relu = ActivationFn{
		self: def_relu
		deriv: def_relup
		rep: 'relu'
	}

	logloss = LossFn{
		self: def_logloss
		rep: 'logloss'
	}
)

/*
fn squaredef_mean(pred Vec, real Vec) !Vec {
	return randomdef_vec(pred.len)
}
*/
