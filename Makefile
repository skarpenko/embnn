# EmbNN Makefile


embnn_test: emb_nn_test.c enn.c enn_train.c enn.h enn_train.h
	gcc -O3 -Wall -pedantic -o $@ emb_nn_test.c enn.c enn_train.c -lm


.PHONY: clean
clean:
	rm -Rf embnn_test
