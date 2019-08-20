#include "fsaf_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_fsaf_layer(int batch, int w, int h, int classes, int max_boxes)
//w,h: feature map width and height
// total: total number of anchors
// num: number of anchors on this level
// mask: anchor mask
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = FSAF;

    l.n = 1;    // anchor free
    // l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = (classes + 4);    // no background class
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)calloc(1, sizeof(float));
    // l.biases = (float*)calloc(total * 2, sizeof(float));
    // if(mask) l.mask = mask;
    // else{
    //     l.mask = (int*)calloc(n, sizeof(int));
    //     for(i = 0; i < n; ++i){
    //         l.mask[i] = i;
    //     }
    // }
    // l.bias_updates = (float*)calloc(n * 2, sizeof(float));
    l.outputs = h*w*(classes + 4);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
    l.delta = (float*)calloc(batch * l.outputs, sizeof(float));
    l.output = (float*)calloc(batch * l.outputs, sizeof(float));
    // for(i = 0; i < total*2; ++i){
    //     l.biases[i] = .5;
    // }

    l.forward = forward_fsaf_layer;
    l.backward = backward_fsaf_layer;
#ifdef GPU
    l.forward_gpu = forward_fsaf_layer_gpu;
    l.backward_gpu = backward_fsaf_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)calloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)calloc(batch * l.outputs, sizeof(float));
    }
#endif

    fprintf(stderr, "fsaf\n");
    srand(time(0));

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    if (!l->output_pinned) l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        cudaFreeHost(l->output);
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)realloc(l->output, l->batch * l->outputs * sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        cudaFreeHost(l->delta);
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)realloc(l->delta, l->batch * l->outputs * sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

void set_area_map_box(float *area_map, int index, float new_area, int stride)
{
    area_map[index + 0*stride] = new_area;
    area_map[index + 1*stride] = new_area;
    area_map[index + 2*stride] = new_area;
    area_map[index + 3*stride] = new_area;
}

box get_fsaf_box(float *x, int index, int i, int j, int lw, int lh, int w, int h, int stride)
// params: l.output, box_index, 
// i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h
// pred: [dis_top, dis_left, dis_bot, dis_right] in image pixel
// return: bbox relative to image size
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
                            // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
                            // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    b.x = i / lw + (x[index + 3*stride] - x[index + 1*stride]) / (2 * w);
    b.y = j / lh + (x[index + 2*stride] - x[index + 0*stride]) / (2 * h);
    b.w = (x[index + 3*stride] + x[index + 1*stride]) / w;
    b.h = (x[index + 2*stride] + x[index + 0*stride]) / h;
    return b;
}

ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss)
//inputs: truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, 
//        state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss
{
    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE)    // old loss
    {
        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2 * n]);
        float th = log(truth.h*h / biases[2 * n + 1]);

        delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
        delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
        delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
        delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    }
    else {
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        delta[index + 0 * stride] = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        delta[index + 1 * stride] = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        delta[index + 2 * stride] = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        delta[index + 3 * stride] = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        delta[index + 2 * stride] *= exp(x[index + 2 * stride]);
        delta[index + 3 * stride] *= exp(x[index + 3 * stride]);

        // normalize iou weight
        delta[index + 0 * stride] *= iou_normalizer;
        delta[index + 1 * stride] *= iou_normalizer;
        delta[index + 2 * stride] *= iou_normalizer;
        delta[index + 3 * stride] *= iou_normalizer;
    }

    return all_ious;
}

ious delta_fsaf_box(box truth, float *x, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss)
//inputs: truth, l.output, box_index, i, j, l.w, l.h, state.net.w, 
//        state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss
{
    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in relative coordinates
    box pred = get_fsaf_box(x, index, i, j, lw, lh, w, h, stride);
    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    
    // IoU loss or GIoU loss
    // https://github.com/generalized-iou/g-darknet
    // https://arxiv.org/abs/1902.09630v2
    // https://giou.stanford.edu/
    all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

    // jacobian^t (transpose)
    delta[index + 0 * stride] = -all_ious.dx_iou.dt;
    delta[index + 1 * stride] = -all_ious.dx_iou.dl;
    delta[index + 2 * stride] = all_ious.dx_iou.db;
    delta[index + 3 * stride] = all_ious.dx_iou.dr;

    // predict exponential, apply gradient of e^delta_t ONLY for w,h
    delta[index + 2 * stride] *= exp(x[index + 2 * stride]);
    delta[index + 3 * stride] *= exp(x[index + 3 * stride]);

    // normalize iou weight
    delta[index + 0 * stride] *= iou_normalizer;
    delta[index + 1 * stride] *= iou_normalizer;
    delta[index + 2 * stride] *= iou_normalizer;
    delta[index + 3 * stride] *= iou_normalizer;
    

    return all_ious;
}

void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss)
{
    int n;
    if (delta[index + stride*class_id]){
        delta[index + stride*class_id] = 1 - output[index + stride*class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = ((n == class_id) ? 1 : 0) - output[index + stride*n];
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes) + entry*l.w*l.h + loc;
}

static box float_to_box_stride(float *f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

void forward_fsaf_layer(const layer l, network_state state)
{
    int i, j, b, t, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);        // x,y,
            scal_add_cpu(4 * l.w*l.h, 4.0, 0.0, l.output + index, 1);    // S=4.0
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!state.train) return;
    //float avg_iou = 0;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;

    for (b = 0; b < l.batch; ++b) {
        for (t = 0; t < l.max_boxes; ++t) {
            // gt bbox in relative size
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
                continue;
            }
            int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
            if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file
            if (l.map) class_id = l.map[class_id];
                int ll=0, skip = 0;
                for (ll=0;ll<l.num_ignore_label;ll++)
                    if (class_id == l.ignore_label[ll]) {
                    //printf("ignoring image %d bbox %d with label %d\n",b,t,class);
                    skip=1;}
                if (skip) continue;

            if (!truth.x) break;  // continue;

            float* area_map = (float *)calloc(l.batch * l.outputs, sizeof(float));
            memset(area_map, 10.0, l.outputs * l.batch * sizeof(float));

            box truth_effective = truth;
            truth_effective.w *= 0.2;
            truth_effective.h *= 0.2;
            box truth_ignore = truth;
            truth_ignore.w *= 0.5;
            truth_ignore.h *= 0.5;

            int truth_ignore_left = (truth_ignore.x - 0.5 * truth_ignore.w) * l.w;
            int truth_ignore_right = (truth_ignore.x + 0.5 * truth_ignore.w) * l.w;
            truth_ignore_right = truth_ignore_right > truth_ignore_left ? truth_ignore_right : (truth_ignore_left + 1);

            int truth_ignore_top = (truth_ignore.y - 0.5 * truth_ignore.h) * l.h;
            int truth_ignore_bottom = (truth_ignore.y + 0.5 * truth_ignore.h) * l.h;
            truth_ignore_bottom = truth_ignore_bottom > truth_ignore_top ? truth_ignore_bottom : (truth_ignore_top + 1);
            
            int truth_effective_left = (truth_effective.x - 0.5 * truth_effective.w) * l.w;
            int truth_effective_right = (truth_effective.x + 0.5 * truth_effective.w) * l.w;
            truth_effective_right = truth_effective_right > truth_effective_left ? truth_effective_right : (truth_effective_left + 1);

            int truth_effective_top = (truth_effective.y - 0.5 * truth_effective.h) * l.h;
            int truth_effective_bottom = (truth_effective.y + 0.5 * truth_effective.h) * l.h;
            truth_effective_bottom = truth_effective_bottom > truth_effective_top ? truth_effective_bottom : (truth_effective_top + 1);
            
            float truth_area = truth.w * truth.h;

            for (j = truth_ignore_top; j < truth_ignore_bottom; ++j) {
                for (i = truth_ignore_left; i < truth_ignore_right; ++i){
                    int box_index = entry_index(l, b, j*l.w + i, 0);
                    // small instances have priority
                    if (area_map[box_index] > truth_area){
                        // effective region
                        if ( j >= truth_effective_top && j < truth_effective_bottom && 
                            i >= truth_effective_left && i < truth_effective_right){
                            set_area_map_box(area_map, box_index, truth_area, l.w*l.h);
                            ious all_ious = delta_fsaf_box(truth, l.output, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss);
                        }
                        else    /* ignore region */
                        {
                            set_area_map_box(area_map, box_index, truth_area, l.w*l.h);
                            l.delta[box_index + 0*l.w*l.h] = 0;
                            l.delta[box_index + 1*l.w*l.h] = 0;
                            l.delta[box_index + 2*l.w*l.h] = 0;
                            l.delta[box_index + 3*l.w*l.h] = 0;
                        }
                    }
                    
                }
            }


            
            

            // match anchors to gts
            // float best_iou = 0;
            // int best_n = 0;
            // i = (truth.x * l.w);
            // j = (truth.y * l.h);
            // box truth_shift = truth;
            // truth_shift.x = truth_shift.y = 0;
            // for (n = 0; n < l.total; ++n) {
            //     box pred = { 0 };
            //     pred.w = l.biases[2 * n] / state.net.w;
            //     pred.h = l.biases[2 * n + 1] / state.net.h;
            //     float iou = box_iou(pred, truth_shift);
            //     if (iou > best_iou) {
            //         best_iou = iou;
            //         best_n = n;
            //     }
            //}

            //int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class_id = l.map[class_id];
                int ll=0, skip = 0;
                for (ll=0;ll<l.num_ignore_label;ll++)
                    if (class_id == l.ignore_label[ll]) {
                    //printf("ignoring image %d bbox %d with label %d\n",b,t,class);
                    skip=1;}
                if (skip) continue;

                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss);

                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);

                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss);

                ++count;
                ++class_count;
                //if(iou > .5) recall += 1;
                //if(iou > .75) recall75 += 1;
                //avg_iou += iou;
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;
            }
        }
    }
    //*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);

    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE) {
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    }
    else {
        // Always compute classification loss both for iou + cls loss and for logging with mse loss
        // TODO: remove IOU loss fields before computing MSE on class
        //   probably split into two arrays
        int stride = l.w*l.h;
        float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
        memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
        for (b = 0; b < l.batch; ++b) {
            for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w; ++i) {
                    for (n = 0; n < l.n; ++n) {
                        int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                        no_iou_loss_delta[index + 0 * stride] = 0;
                        no_iou_loss_delta[index + 1 * stride] = 0;
                        no_iou_loss_delta[index + 2 * stride] = 0;
                        no_iou_loss_delta[index + 3 * stride] = 0;
                    }
                }
            }
        }
        float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
        free(no_iou_loss_delta);

        if (l.iou_loss == GIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
        }
        else {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
        }
        *(l.cost) = avg_iou_loss + classification_loss;
    }
    printf("v3 (%s loss, Normalizer: (iou: %f, cls: %f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d\n", (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.cls_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);
}

void backward_fsaf_layer(const layer l, network_state state)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void correct_fsaf_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int fsaf_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_fsaf_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);    // x,y
            scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale x,y
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}
#endif
