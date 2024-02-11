#include <malloc.h>
#include <math.h>
#include <stdlib.h>

#include "neuro.h"

struct net* neuro_init(uint16_t inputs_n, uint16_t layers_n, uint16_t *neurons_n) {
    if (layers_n < 2) return NULL;

    struct layer* prev = NULL;
    struct layer* layer;
    struct net* net = (struct net*)malloc(sizeof(*net));

    for (uint16_t i = 0; i < layers_n; i++) {
        layer = (struct layer*)malloc(sizeof(*layer));
        if (i == 0) net->input = layer;

        layer->inputs_n = inputs_n;
        layer->neurons_n = neurons_n[i];
        inputs_n = neurons_n[i];

        layer->prev = prev;
        if (prev) prev->next = layer;
        prev = layer;

        layer->weights = (double**)malloc(layer->neurons_n * sizeof(double*));
        for (uint16_t j = 0; j < layer->neurons_n; j++) {
            layer->weights[j] = (double*)malloc(layer->inputs_n * sizeof(double));
            for (uint16_t k = 0; k < layer->inputs_n; k++) {
                layer->weights[j][k] =  (double)rand() / (double)RAND_MAX - 0.5;
            }
        }
        layer->last_outputs = (double*)malloc(layer->neurons_n * sizeof(double));
        layer->deltas = (double*)malloc(layer->neurons_n * sizeof(double));

        layer->last_output_int = (int8_t*)malloc(layer->neurons_n * sizeof(int8_t));
    }

    layer->next = NULL;
    net->output = layer;
    return net;
}

void neuro_free(struct net* net) {
    struct layer* layer = net->input;
    while (layer) {
        for (uint16_t i = 0; i < layer->neurons_n; i++) free(layer->weights[i]);
        free(layer->weights);
        free(layer->last_outputs);
        free(layer->deltas);
        free(layer->last_output_int);
        struct layer* next = layer->next;
        free(layer);
        layer = next;
    }
    free(net);
}

void neuro_predict(struct net* net, double *inputs) {
    struct layer* layer = net->input;
    double output_sum;

    while (layer) {
        for (uint16_t i = 0; i < layer->neurons_n; i++) {
            output_sum = 0;
            for (uint16_t j = 0; j < layer->inputs_n; j++) {
                output_sum += inputs[j] * layer->weights[i][j];
            }
            layer->last_outputs[i] = 1.0 / (1.0 + exp(-output_sum));
        }
        inputs = layer->last_outputs;
        layer = layer->next;
    }
}

void neuro_predict_test_int(struct net* net, int8_t *inputs) {
    struct layer* layer = net->input;
    int16_t output_sum;

    while (layer) {
        for (uint16_t i = 0; i < layer->neurons_n; i++) {
            output_sum = 0;
            for (uint16_t j = 0; j < layer->inputs_n; j++) {
                output_sum += inputs[j] * (int)(layer->weights[i][j] * 32.0);
            }
            layer->last_output_int[i] = (int)(1.0 / (1.0 + exp(-(double)output_sum / 1024.0)) * 32.0);
        }
        inputs = layer->last_output_int;
        layer = layer->next;
    }
}

static void update_deltas_output(struct layer* layer, double* ref_out) {
    for (uint16_t i = 0; i < layer->neurons_n; i++) {
        layer->deltas[i] = layer->last_outputs[i] 
                * (1.0 - layer->last_outputs[i]) * (ref_out[i] - layer->last_outputs[i]);
    }
}

static void update_deltas_non_output(struct layer* layer) {
    double w_sum;
    struct layer* next = layer->next;
    for (uint16_t i = 0; i < layer->neurons_n; i++) {
        w_sum = 0;
        for (uint16_t j = 0; j < next->neurons_n; j++) {
            w_sum += next->deltas[j] * next->weights[j][i];
        }
        layer->deltas[i] = layer->last_outputs[i] 
                * (1.0 - layer->last_outputs[i]) * w_sum;
    }
}

static void update_weights(struct layer* layer, double a, double* inputs) {
    for (uint16_t i = 0; i < layer->neurons_n; i++) {
        for (uint16_t j = 0; j < layer->inputs_n; j++) {
            layer->weights[i][j] += a * layer->deltas[i] * inputs[j];
        }
    }
}

bool neuro_make_train_step(struct net* net, double *inputs, double *ref_out, double err_lim, double a) {
    neuro_predict(net, inputs);

    double err = 0;
    for (uint16_t i = 0; i < net->output->neurons_n; i++) {
        err += fabs(ref_out[i] - net->output->last_outputs[i]);
    }
    if ((err / 2.0) <= err_lim) return true;

    update_deltas_output(net->output, ref_out);
    update_weights(net->output, a, net->output->prev->last_outputs);

    struct layer* layer = net->output->prev;
    while (layer) {
        update_deltas_non_output(layer);
        update_weights(layer, a, layer->prev != NULL ? layer->prev->last_outputs : inputs);
        layer = layer->prev;
    }
    return false;
}
