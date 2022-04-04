#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"
#include <map>

enum DETECTBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H };

/**
 * Each rect's data structure.
 * tlwh: topleft point & (w,h)
 * confidence: detection confidence.
 * feature: the rect's 128d feature.
 */
class DETECTION_ROW {
public:
    DETECTBOX tlwh; //np.float
    float confidence; //float
    FEATURE feature; //np.float32
    DETECTBOX to_xyah() const;
    DETECTBOX to_tlbr() const;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;

class ModelDetection
{

public:
    static ModelDetection* getInstance();
    void dataMoreConf(float min_confidence, DETECTIONS& d);
    void dataPreprocessing(float max_bbox_overlap, DETECTIONS& d);

private:
    ModelDetection();
    static ModelDetection* instance;
    using AREAPAIR = std::pair<int, double>;
    struct cmp {
        bool operator()(const AREAPAIR a, const AREAPAIR b) {
            return a.second < b.second;
        }
    };
    //std::map<int, DETECTIONS> data;
    void _Qsort(DETECTIONS d, std::vector<int>& a, int low, int high);

};

#endif // MODEL_H
