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

#include <math.h>
#include "enn.h"


void enn_prod_propagate(struct enn_layer *layer, const double *in)
{
	struct enn_prod_layer *pl = (struct enn_prod_layer*)layer;
	size_t n;
	size_t i;

	for(n = 0; n < pl->base.no; ++n) {
		double *w = &pl->weights[n * (pl->ni + 1)];
		pl->base.out[n] = w[0];		/* Bias */
		++w;
		for(i = 0; i < pl->ni; ++i)
			pl->base.out[n] += w[i] * in[i];
	}
}


void enn_logact_propagate(struct enn_layer *layer, const double *in)
{
	size_t n;
	for(n = 0; n < layer->no; ++n)
		layer->out[n] = (1.0 / (1.0 + exp (-in[n])));
}


void enn_propagate(struct enn_net *nn, const double *input)
{
	size_t l;
	const double *in = input;
	for(l = 0; l < nn->nl; ++l) {
		nn->layers[l]->propagate(nn->layers[l], in);
		in = nn->layers[l]->out;
	}
}
