#ifndef FSAF_LAYER_H
#define FSAF_LAYER_H

//#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_fsaf_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_fsaf_layer(const layer l, network_state state);
void backward_fsaf_layer(const layer l, network_state state);
void resize_fsaf_layer(layer *l, int w, int h);
int fsaf_num_detections(layer l, float thresh);
int get_fsaf_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
void correct_fsaf_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_fsaf_layer_gpu(const layer l, network_state state);
void backward_fsaf_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
