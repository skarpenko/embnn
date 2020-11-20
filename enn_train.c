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

#include <stdlib.h>
#include <math.h>
#include "enn_train.h"


static inline
void enn_mlpl_backprop_out(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer)
{
	size_t n;

	/* Compute loss and deltas */
	for(n = 0; n < layer->al->base.no; ++n) {
		double diff = mlp_train->loss(mlp_train, layer->al->base.out[n],
			mlp_train->target[n]);
		layer->deltas[n] = layer->deriv(mlp_train, layer->pl->base.out[n]) * diff;
	}
}


static inline
void enn_mlpl_backprop_hid(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer, struct enn_mlp_train_layer *next)
{
	size_t n;
	size_t sn;

	/* Loop for all neurons in current layer */
	for(n = 0; n < layer->pl->base.no; ++n) {
		layer->deltas[n] = 0.0;
		/* Loop for all neurons in next layer */
		for(sn = 0; sn < next->pl->base.no; ++sn) {
			/* Weight of output for current layer neuron */
			double w = next->pl->weights[sn * (next->pl->ni+1) + 1/*Bias*/ + n];
			layer->deltas[n] += w * next->deltas[sn];
		}
		layer->deltas[n] *= layer->deriv(mlp_train, layer->pl->base.out[n]);
	}
}


void enn_mlpl_backprop(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer, struct enn_mlp_train_layer *next)
{
	if(!next)	/* For output layer */
		enn_mlpl_backprop_out(mlp_train, layer);
	else		/* For hidden layer */
		enn_mlpl_backprop_hid(mlp_train, layer, next);
}


void enn_mlpl_rand_weights(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer)
{
	size_t w, n = (layer->pl->ni + 1) * layer->pl->base.no;

	(void)mlp_train;

	for(w = 0; w < n; ++w)
		layer->pl->weights[w] = (double)rand() / RAND_MAX;

}


void enn_mlpl_reset_diffs(struct enn_mlp_trainer *mlp_train,
	struct enn_mlp_train_layer *layer)
{
	size_t w, n = (layer->pl->ni + 1) * layer->pl->base.no;

	(void)mlp_train;

	for(w = 0; w < n; ++w)
		layer->weights_diff[w] = 0.0;
}


void enn_mlpl_adjust_weights(struct enn_mlp_trainer *mlp_train, const double *in,
	struct enn_mlp_train_layer *layer)
{
	size_t n;
	size_t i;

	/* Adjust weights */
	for(n = 0; n < layer->pl->base.no; ++n) {
		double *bias = &layer->pl->weights[n * (layer->pl->ni + 1)];
		double *weights = &layer->pl->weights[n * (layer->pl->ni + 1) + 1];
		double *bias_diff = &layer->weights_diff[n * (layer->pl->ni + 1)];
		double *weights_diff = &layer->weights_diff[n * (layer->pl->ni + 1) + 1];

		bias_diff[0] = mlp_train->eta * layer->deltas[n] -
				/* Weight decay term */
				mlp_train->lambda * bias[0] +
				/* Momentum term */
				mlp_train->alpha * bias_diff[0];

		bias[0] += bias_diff[0];

		for(i=0; i < layer->pl->ni; ++i) {
			weights_diff[i] =
				/* Standard backpropagation */
				mlp_train->eta * layer->deltas[n] * in[i] -
				/* Weight decay term */
				mlp_train->lambda * weights[i] +
				/* Momentum term */
				mlp_train->alpha * weights_diff[i];
			weights[i] += weights_diff[i];
		}
	}
}


double enn_mlp_logact_deriv(struct enn_mlp_trainer *mlp_train, double in)
{
	double log_norm;

	(void)mlp_train;

	log_norm = (1.0 / (1.0 + exp (-in)));

	return (log_norm * (1.0 - log_norm));
}


double enn_mlp_loss(struct enn_mlp_trainer *mlp_train, double out, double target)
{
	(void)mlp_train;
	return target - out;
}


void enn_mlp_reset_diffs(struct enn_mlp_trainer *mt)
{
	size_t l;

	for(l = 0; l < mt->nl; ++l)
		mt->layers[l]->reset_diffs(mt, mt->layers[l]);
}


void enn_mlp_rand_weights(struct enn_mlp_trainer *mt)
{
	size_t l;

	for(l = 0; l < mt->nl; ++l)
		mt->layers[l]->rand_weights(mt, mt->layers[l]);
}


void enn_mlp_backprop(struct enn_mlp_trainer *mt, const double *in,
	const double *target)
{
	size_t l;
	struct enn_mlp_train_layer *next = NULL;

	mt->target = target;

	/* Backpropagate */
	for(l = mt->nl - 1; l != (size_t)-1 ; --l) {
		mt->layers[l]->backprop(mt, mt->layers[l], next);
		next = mt->layers[l];
	}

	/* Adjust weights */
	for(l = 0; l < mt->nl; ++l) {
		mt->layers[l]->adjust_weights(mt,
			l ? mt->layers[l-1]->al->base.out : in, mt->layers[l]);
	}
}
