#ifndef _NEURO_H
#define _NEURO_H

#include <stdint.h>
#include <stdbool.h>

struct layer {
    double **weights;
    double *last_outputs;
    double *deltas;

    uint16_t inputs_n;
    uint16_t neurons_n;

    struct layer *prev;
    struct layer *next;
};

struct net {
    struct layer *input;
    struct layer *output;
};

struct net* neuro_init(uint16_t inputs_n, uint16_t layers_n, uint16_t *neurons_n);
void neuro_free(struct net* net);

void neuro_predict(struct net* net, double *inputs);
bool neuro_make_train_step(struct net* net, double *inputs, double *ref_out, double err_lim, double a);

#endif