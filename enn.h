/*
 * Copyright (c) 2019 Stepan Karpenko. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _EMBEDDED_NN_H_
#define _EMBEDDED_NN_H_

#include <stddef.h>


/* Base layer structure */
struct enn_layer {
	double *out;	/* Output vector */
	size_t no;	/* Number of outputs */
	/* Propagation function */
	void (*propagate)(struct enn_layer*, const double*);
};


/* Product layer */
struct enn_prod_layer {
	struct enn_layer base;	/* Base */
	size_t ni;		/* Number of inputs */
	double *weights;	/* Weights and biases */
};


/* Activation layer */
struct enn_act_layer {
	struct enn_layer base;	/* Base */
};


/* Neural Network */
struct enn_net {
	struct enn_layer **layers;	/* Layers */
	size_t nl;			/* Number of layers */
};


/*** Internally used functions ***/

/* Propagation function for product layer */
void enn_prod_propagate(struct enn_layer *layer, const double *in);
/* Propagation function for logistic activation layer */
void enn_logact_propagate(struct enn_layer *layer, const double *in);


/*** Public interface ***/

/* Returns a pointer to base layer structure */
#define ENNL(_b) &(_b).base


/*
 * Define product layer
 *
 * Arguments:
 *   _name - layer object name;
 *   _ni   - number of inputs;
 *   _no   - number of outputs.
 */
#define ENN_PROD_LAYER(_name, _ni, _no)			\
	static double _name##_weights[((_ni)+1)*(_no)];	\
	static double _name##_output[(_no)];		\
	struct enn_prod_layer _name = {			\
		.base = {				\
			.out = _name##_output,		\
			.no = (_no),			\
			.propagate = enn_prod_propagate	\
		},					\
		.ni = (_ni),				\
		.weights = _name##_weights		\
	}


/*
 * Define product layer using existing weights location
 *
 * Arguments:
 *   _name    - layer object name;
 *   _ni      - number of inputs;
 *   _no      - number of outputs;
 *   _weights - pointer to an array of weights.
 */
#define ENN_PROD_LAYER_PTR(_name, _ni, _no, _weights)	\
	static double _name##_output[(_no)];		\
	struct enn_prod_layer _name = {			\
		.base = {				\
			.out = _name##_output,		\
			.no = (_no),			\
			.propagate = enn_prod_propagate	\
		},					\
		.ni = (_ni),				\
		.weights = (_weights)			\
	}


/*
 * Define activation layer
 *
 * Arguments:
 *   _name - layer object name;
 *   _actf - activation function;
 *   _n    - number of inputs / outputs.
 */
#define ENN_ACT_LAYER(_name, _actf, _n)		\
	static double _name##_output[(_n)];	\
	struct enn_act_layer _name = {		\
		.base = {			\
			.out = _name##_output,	\
			.no = (_n),		\
			.propagate = (_actf)	\
		}				\
	}


/*
 * Define logistic activation layer
 *
 * Arguments:
 *   _name - layer object name;
 *   _n    - number of inputs / outputs.
 */
#define ENN_LOGACT_LAYER(_name, _n)	\
	ENN_ACT_LAYER(_name, enn_logact_propagate, _n)


/*
 * Define neural network
 *
 * Arguments:
 *   _name - layer object name;
 *   ...   - list of layers, for ex., ENNBP(prod1), ENNBP(act1), etc.
 */
#define ENN_NET(_name, ...)						\
	static struct enn_layer *_name##_layers[] = { __VA_ARGS__ };	\
	struct enn_net _name = {					\
		.layers = _name##_layers,				\
		.nl = sizeof(_name##_layers)/sizeof(_name##_layers[0])	\
	}


/* Get NN output vector */
static inline
const double *enn_get_output(const struct enn_net *nn)
{
	return nn->layers[nn->nl-1]->out;
}


/* Run forward propagation */
void enn_propagate(struct enn_net *nn, const double *input);


#endif /* _EMBEDDED_NN_H_ */
