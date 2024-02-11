#include <png++/png.hpp>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>

#include "neuro.h"

static void read_pic(std::string path, double* inputs) {
    png::image<png::gray_pixel> pic(path);
    for (uint32_t y = 0; y < 7; y++) {
        for (uint32_t x = 0; x < 7; x++) {
            inputs[y * 7 + x] = pic.get_pixel(x, y) > 0 ? 1.0 : 0.0;
        }
    }
}

static void read_pic_int(std::string path, int8_t* inputs) {
    png::image<png::gray_pixel> pic(path);
    for (uint32_t y = 0; y < 7; y++) {
        for (uint32_t x = 0; x < 7; x++) {
            inputs[y * 7 + x] = pic.get_pixel(x, y) > 0 ? 32 : 0;
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
    int8_t inputs_int[49];

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

    // struct layer* layer = net->input;
    // while (layer) {
    //     for (uint16_t i = 0; i < layer->neurons_n; i++) {
    //         for (uint16_t j = 0; j < layer->inputs_n; j++) {
    //             printf("%f ", layer->weights[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     layer = layer->next;
    // }

    struct layer* layer = net->input;
    double min_w = __DBL_MAX__;
    double max_w = __DBL_MIN__;
    
    while (layer) {
        for (uint16_t i = 0; i < layer->neurons_n; i++) {
            for (uint16_t j = 0; j < layer->inputs_n; j++) {
                if (layer->weights[i][j] > max_w) max_w = layer->weights[i][j];
                if (layer->weights[i][j] < min_w) min_w = layer->weights[i][j];
            }
        }
        layer = layer->next;
    }

    printf("min w: %f; max w: %f\n\n", min_w, max_w);

    std::ofstream wfile;
    wfile.open("weights.txt");

    layer = net->input;
    while (layer) {
	for (uint16_t i = 0; i < layer->inputs_n; ++i) {
		for (uint16_t j = 0; j < layer->neurons_n; ++j) {
			wfile << (int)(layer->weights[j][i] * 32.0) << "\n";
		}
	}
	layer = layer->next;
    }
    wfile.close();

    std::ifstream ifile;
    ifile.open("weights.txt");
    std::string file_input;

//     while (std::getline(ifile, file_input)) {
// 	std::cout << std::stoi(file_input) << "\n";
//     }

    path = "/home/merlin/git/ovs1/circle/6.png";
    read_pic_int(path, inputs_int);
    for (uint8_t i = 0; i < 49; i++) std::cout << (int)inputs_int[i] << " ";
    std::cout << std::endl;

    for (uint8_t i = 0; i < 3; i++) {
        for (uint8_t j = 6; j < 9; j++) {
            path = dirs[i] + "/" + std::to_string(j) + ".png";
            read_pic(path, inputs);
            neuro_predict(net, inputs);
            read_pic_int(path, inputs_int);
            neuro_predict_test_int(net, inputs_int);
            printf("Ref: %s; Neuro: cir - %f squ - %f tri - %f; cir - %f squ - %f tri - %f\n", dirs[i].c_str(), 
            net->output->last_outputs[0], net->output->last_outputs[1], net->output->last_outputs[2], 
            net->output->last_output_int[0] / 32.0, net->output->last_output_int[1] / 32.0, net->output->last_output_int[2] / 32.0);
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