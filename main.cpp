#include <png++/png.hpp>
#include <time.h>
#include <math.h>

#include "neuro.h"

static void read_pic(std::string path, double* inputs) {
    png::image<png::gray_pixel> pic(path);
    for (uint32_t y = 0; y < 7; y++) {
        for (uint32_t x = 0; x < 7; x++) {
            inputs[y * 7 + x] = pic.get_pixel(x, y) > 0 ? 1.0 : 0.0;
        }
    }
}

int main(int argc, char* argv[]) {

    uint16_t inputs_n = atoi(argv[1]);
    uint16_t layers_n = atoi(argv[2]);

    uint16_t *neurons_n = (uint16_t*)malloc(layers_n * sizeof(neurons_n));
    for (uint16_t i = 0; i < layers_n; i++) neurons_n[i] = atoi(argv[3 + i]);

    struct net* net = neuro_init(inputs_n, layers_n, neurons_n);

    std::string path;
    std::string dirs[] = {"circle", "square", "triangle"};
    double refs[][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    
    double inputs[49];

    double err_limit = 0.1;
    double a = 0.2;
    bool flag = false;

    clock_t train_begin = clock();
    uint64_t epochs = 0;

    while (!flag) {
        flag = true;
        for (uint8_t i = 1; i < 6; i++) {
            for (uint8_t j = 0; j < 3; j++) {
                path = dirs[j] + "/" + std::to_string(i) + ".png";
                read_pic(path, inputs);
                if (!neuro_make_train_step(net, inputs, refs[j], err_limit, a)) flag = false;
            }
        }
        epochs++;
    }

    clock_t train_end = clock();

    neuro_predict(net, inputs);
    clock_t predict_end = clock();

    double train_time_ms = (double)(train_end - train_begin) / CLOCKS_PER_SEC * 1000.0;
    double inference_time_ms =  (double)(predict_end - train_end) / CLOCKS_PER_SEC * 1000.0;
    
    printf("Train epochs: %lu\nTrain time: %fms\nInference time: %fms\n\n", 
    epochs, train_time_ms, inference_time_ms);

    double err = 0;

    for (uint8_t i = 0; i < 3; i++) {
        for (uint8_t j = 6; j < 9; j++) {
            path = dirs[i] + "/" + std::to_string(j) + ".png";
            read_pic(path, inputs);
            neuro_predict(net, inputs);
            printf("Ref: %s; Neuro: cir - %f squ - %f tri - %f\n", dirs[i].c_str(), 
            net->output->last_outputs[0], net->output->last_outputs[1], net->output->last_outputs[2]);
            for (uint8_t k = 0; k < 3; k++) {
                err += fabs(refs[i][k] - net->output->last_outputs[k]);
            }
        }
        printf("\n\n");
    }

    printf("Total err: %f\n", err);

    neuro_free(net);
    free(neurons_n);
    return 0;
}