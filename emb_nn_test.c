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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "enn.h"


#define SKIP_TRAINING	0	/* Enable / disable training stage */


#if SKIP_TRAINING == 0
#  include "enn_train.h"
#endif


/** Training set for XOR function **/

/* Input training vectors */
const double in[4][2] = {
	{ 0.0, 0.0 },
	{ 1.0, 0.0 },
	{ 0.0, 1.0 },
	{ 1.0, 1.0 },
};

/* Expected output */
const double out[4] = {
	0.0,
	1.0,
	1.0,
	0.0
};

/* Trained weights (used when training is disabled) */
double weights_l1[] = { -10.585747, 6.942968, 6.942795, -4.106738, 8.767143, 8.766392 };
double weights_l2[] = { -8.959998, -19.033336, 18.471044 };


/* Define product layer with two inputs and outputs */
#if SKIP_TRAINING == 1
  ENN_PROD_LAYER_PTR(prod1, 2, 2, weights_l1);
#else
  ENN_PROD_LAYER(prod1, 2, 2);
#endif
/* Define logact layer with two inputs and outputs */
ENN_LOGACT_LAYER(act1, 2);
/* Define product layer with two inputs and one output */
#if SKIP_TRAINING == 1
  ENN_PROD_LAYER_PTR(prod2, 2, 1, weights_l2);
#else
  ENN_PROD_LAYER(prod2, 2, 1);
#endif
/* Define logact layer with one input and output */
ENN_LOGACT_LAYER(act2, 1);


/* Define neural network */
ENN_NET(mlp, ENNL(prod1), ENNL(act1), ENNL(prod2), ENNL(act2));


#if SKIP_TRAINING == 0
/* Define first training layer */
ENN_MLP_TRAIN_LAYER(tr_layer1, ENNP(prod1), ENNP(act1), 2, 2, enn_mlp_logact_deriv);
/* Define second training layer */
ENN_MLP_TRAIN_LAYER(tr_layer2, ENNP(prod2), ENNP(act2), 2, 1, enn_mlp_logact_deriv);
/* Define a trainer for Multilayer Perceptron (MLP) network */
ENN_MLP_TRAINER(mlp_trainer, 0.9, 0.5, 0.0, enn_mlp_loss, ENNP(tr_layer1), ENNP(tr_layer2));
#endif


/* Print weights of product layer */
void enn_print_weights(struct enn_prod_layer *pl);


/* MAIN */
int main()
{
#if SKIP_TRAINING == 0
	size_t epoch, nepochs = 1000000;
#endif
	size_t p, n = sizeof(out) / sizeof(out[0]);

	/* Random seed */
	srand(time(NULL));

	printf("EmbNN test\n---\n");

	printf("Initial weights:\n");
	printf("Layer 1:  "); enn_print_weights(&prod1);
	printf("Layer 2:  "); enn_print_weights(&prod2);
	printf("\n");

#if SKIP_TRAINING == 0
	/* Prepare for training */
	enn_mlp_rand_weights(&mlp_trainer);
	enn_mlp_reset_diffs(&mlp_trainer);

	printf("Randomized weights:\n");
	printf("Layer 1:  "); enn_print_weights(&prod1);
	printf("Layer 2:  "); enn_print_weights(&prod2);
	printf("\n");

	printf("Training ...\n");
	for(epoch = 0; epoch < nepochs; ++epoch) {
		for(p = 0; p < n; ++p) {
			/* Propagate */
			enn_propagate(&mlp, &in[p][0]);
			/* Backpropagate */
			enn_mlp_backprop(&mlp_trainer, &in[p][0], &out[p]);
		}
	}
	printf("\n");

	printf("Trained weights:\n");
	printf("Layer 1:  "); enn_print_weights(&prod1);
	printf("Layer 2:  "); enn_print_weights(&prod2);
	printf("\n");
#endif

	printf("Testing...\n");
	for(p = 0; p < n; ++p) {
		enn_propagate(&mlp, &in[p][0]);
		printf("%f  xor  %f  =  %f  (%f)\n", in[p][0], in[p][1],
			enn_get_output(&mlp)[0], out[p]);
	}

	printf("\nAll done.\n");

	return 0;
}


void enn_print_weights(struct enn_prod_layer *pl)
{
	size_t i;
	size_t n = (pl->ni + 1) * pl->base.no;

	printf("weights =");
	for(i = 1; i < n; ++i)
		printf(i == 1 ? " %f" : ", %f", pl->weights[i]);

	printf("   bias = %f\n", pl->weights[0]);
}
