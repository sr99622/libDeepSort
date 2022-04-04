#ifndef FEATUREMODEL_H
#define FEATUREMODEL_H

#include <iostream>
#include <tensorflow/c/c_api.h>

#include <opencv2/opencv.hpp>
#include "model.h"

#define num_channels 3
#define feature_size 128
#define crop_width 64
#define crop_height 128

class FeatureModel
{

public:
    FeatureModel();
    ~FeatureModel();
    bool load(const char* saved_model_dir, double pct_gpu_mem = 0.2);
    void run(cv::Mat& image, DETECTIONS& detections);
    void clear();

    TF_SessionOptions* CreateSessionOptions(double perecentage);
    static void NoOpDeallocator(void *data, size_t a, void *b) { }

    cv::Mat getCrop(const cv::Mat& image, DETECTION_ROW& row) const;


    bool initialized = false;
    const char *tags = "serve";
    TF_Graph *Graph = nullptr;
    TF_Status *Status = nullptr;
    TF_SessionOptions *SessionOpts = nullptr;
    TF_Buffer *RunOpts = nullptr;
    TF_Session *Session = nullptr;
    int ntags = 1;

    TF_Output *Input = nullptr;
    TF_Output *Output = nullptr;
    TF_Tensor **InputValues = nullptr;
    TF_Tensor **OutputValues = nullptr;
    int NumInputs = 1;
    int NumOutputs = 1;

};

#endif // FEATUREMODEL_H
