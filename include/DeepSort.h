#ifndef DEEPSORT_H
#define DEEPSORT_H

#include "featuremodel.h"
#include "tracker.h"

class DeepSort
{
public:
    DeepSort(const char* saved_model_dir, float gpu_pct = 0.3f);

    void sort
    (
        cv::Mat& image,
        DETECTIONS& detections,
        std::vector<RESULT_DATA>* result
    );

    FeatureModel* model;
    tracker* mytracker;

};

#endif // DEEPSORT_H
