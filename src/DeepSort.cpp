#include "DeepSort.h"

#define args_nn_budget 100
#define args_max_cosine_distance 0.2
#define args_min_confidence 0.3
#define args_nms_max_overlap 1.0

DeepSort::DeepSort(const char* saved_model_dir, float gpu_pct)
{
    model = new FeatureModel();
    model->load(saved_model_dir, gpu_pct);
    mytracker = new tracker((float)args_max_cosine_distance, args_nn_budget);

}

void DeepSort::sort(cv::Mat& image, DETECTIONS& detections, std::vector<RESULT_DATA>* result)
{
    model->run(image, detections);

    ModelDetection::getInstance()->dataMoreConf((float)args_min_confidence, detections);
    ModelDetection::getInstance()->dataPreprocessing(args_nms_max_overlap, detections);

    mytracker->predict();
    mytracker->update(detections);

    for (Track& track : mytracker->tracks) {
        if (!track.is_confirmed() || track.time_since_update > 1)
            continue;
        result->push_back(std::make_pair(track.track_id, track.to_tlwh()));
    }
}
