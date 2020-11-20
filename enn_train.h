/*
 * Copyright (c) 2019-2020 Stepan Karpenko. All rights reserved.
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

#ifndef _EMBEDDED_NN_TRAIN_H_
#define _EMBEDDED_NN_TRAIN_H_

#include "enn.h"


struct enn_mlp_trainer;


/* Training layer for MLP */
struct enn_mlp_train_layer {
	struct enn_prod_layer *pl;	/* Pointer to product layer */
	struct enn_act_layer *al;	/* Pointer to activation layer */

	/* Derivative of activation function */
	double (*deriv)(struct enn_mlp_trainer*, double);
	/* Backpropagation step */
	void (*backprop)(struct enn_mlp_trainer*, struct enn_mlp_train_layer*,
		struct enn_mlp_train_layer*);
	/* Weights randomization */
	void (*rand_weights)(struct enn_mlp_trainer*, struct enn_mlp_train_layer*);
	/* Wights and biases reset */
	void (*reset_diffs)(struct enn_mlp_trainer*, struct enn_mlp_train_layer*);
	/* Wights adjustment after backpropagation */
	void (*adjust_weights)(struct enn_mlp_trainer*, const double*,
		struct enn_mlp_train_layer*);

	double *deltas;		/* Deltas */
	double *weights_diff;	/* Weights difference */
};


/* MLP trainer */
struct enn_mlp_trainer {
	struct enn_mlp_train_layer **layers;	/* Training layers */
	size_t nl;				/* Number of layers */
	double alpha;				/* Momentum factor */
	double eta;				/* Learning rate */
	double lambda;				/* Weight decay */
	/* Loss function */
	double (*loss)(struct enn_mlp_trainer*, double, double);
	const double *target;			/* Target vector */
};


/*** Internally used functions ***/

/* MLP: backpropagation step */
void enn_mlpl_backprop(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer, struct enn_mlp_train_layer *next);
/* MLP: weights and biases randomization */
void enn_mlpl_rand_weights(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer);
/* MLP: weight differences reset */
void enn_mlpl_reset_diffs(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer);
/* MLP: weights adjustment */
void enn_mlpl_adjust_weights(struct enn_mlp_trainer *mlp_train, const double *in,
	struct enn_mlp_train_layer *layer);


/*** Public interface ***/

/* Returns a pointer to a specified object */
#define ENNP(_b) &(_b)

/*
 * Define MLP training layer
 *
 * Arguments:
 *   _name  - training layer object name;
 *   _pl    - a pointer to product layer;
 *   _al    - a pointer to activation layer;
 *   _ni    - number of inputs;
 *   _no    - number of outputs;
 *   _deriv - derivative of activation function.
 */
#define ENN_MLP_TRAIN_LAYER(_name, _pl, _al, _ni, _no, _deriv)	\
	static double _name##_deltas[(_no)];			\
	static double _name##_weights_diff[((_ni)+1)*(_no)];	\
	struct enn_mlp_train_layer _name = {			\
		.pl = (_pl),					\
		.al = (_al),					\
		.deriv = (_deriv),				\
		.backprop = enn_mlpl_backprop,			\
		.rand_weights = enn_mlpl_rand_weights,		\
		.reset_diffs = enn_mlpl_reset_diffs,		\
		.adjust_weights = enn_mlpl_adjust_weights,	\
		.deltas = _name##_deltas,			\
		.weights_diff = _name##_weights_diff,		\
	}


/*
 * Define MLP trainer
 *
 * Arguments:
 *   _name - trainer object name;
 *   _mf   - momentum factor;
 *   _lr   - learning rate;
 *   _wd   - weight decay;
 *   _loss - loss function.
 */
#define ENN_MLP_TRAINER(_name, _mf, _lr, _wd, _loss, ...)				\
	static struct enn_mlp_train_layer *_name##_train_layers[] = { __VA_ARGS__ };	\
	struct enn_mlp_trainer _name = {						\
		.layers = _name##_train_layers,						\
		.nl = sizeof(_name##_train_layers) / sizeof(_name##_train_layers[0]),	\
		.alpha = (_mf),								\
		.eta = (_lr),								\
		.lambda = (_wd),							\
		.loss = (_loss)								\
	}


/* Set momentum factor */
static inline
void enn_mlp_set_alpha(struct enn_mlp_trainer *trainer, double alpha)
{
	trainer->alpha = alpha;
}


/* Get momentum factor value */
static inline
double enn_mlp_get_alpha(struct enn_mlp_trainer *trainer)
{
	return trainer->alpha;
}


/* Set learning rate */
static inline
void enn_mlp_set_eta(struct enn_mlp_trainer *trainer, double eta)
{
	trainer->eta = eta;
}


/* Get learning rate value */
static inline
double enn_mlp_get_eta(struct enn_mlp_trainer *trainer)
{
	return trainer->eta;
}


/* Set weight decay */
static inline
void enn_mlp_set_lambda(struct enn_mlp_trainer *trainer, double lambda)
{
	trainer->lambda = lambda;
}


/* Get weight decay value */
static inline
double enn_mlp_get_lambda(struct enn_mlp_trainer *trainer)
{
	return trainer->lambda;
}


/* Derivative of logistic activation function */
double enn_mlp_logact_deriv(struct enn_mlp_trainer *mlp_train, double in);


/* Loss function */
double enn_mlp_loss(struct enn_mlp_trainer *mlp_train, double out, double target);


/* Reset all differences */
void enn_mlp_reset_diffs(struct enn_mlp_trainer *mt);


/* Randomize weights */
void enn_mlp_rand_weights(struct enn_mlp_trainer *mt);


/*
 * Run backpropagation
 *
 * Arguments:
 *   mt     - pointer to MLP trainer;
 *   in     - input vector;
 *   target - expected output vector.
 */
void enn_mlp_backprop(struct enn_mlp_trainer *mt, const double *in, const double *target);


#endif /* _EMBEDDED_NN_TRAIN_H_ */
