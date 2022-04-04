#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <opencv2/opencv.hpp>

#include "DeepSort.h"

std::vector<std::string> image_filenames;
std::map<int, DETECTIONS> file_dets;
const char* image_pathname = "C:/Users/sr996/source/repos/deep_sort_v2/MOT16/test/MOT16-06/img1";
const char* detections_filename = "C:/Users/sr996/source/repos/deep_sort_v2/MOT16/test/MOT16-06/det/det.txt";
const char* saved_model_dir = "C:/Users/sr996/source/repos/deep_sort_v2/saved_model";


void loadFilenames(const char* dir)
{
    std::string path = dir;
    for (const auto& entry : std::filesystem::directory_iterator(path))
        image_filenames.push_back(entry.path().filename().string());
}

void loadDetections(const char* filename)
{
    std::ifstream ifs(filename, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << "error loadDetections could not open file" << std::endl;
        return;
    }

    for (std::string line; std::getline(ifs, line);) {
        std::istringstream ss(std::move(line));
        std::vector<std::string> row;
        for (std::string value; std::getline(ss, value, ',');) {
            row.push_back(std::move(value));
        }

        int index = (int)std::stof(row[0]);

        DETECTION_ROW tmpRow;
        float x = std::stof(row[2]);
        float y = std::stof(row[3]);
        float w = std::stof(row[4]);
        float h = std::stof(row[5]);
        tmpRow.tlwh = DETECTBOX(x, y, w, h);
        tmpRow.confidence = std::stof(row[6]);
        file_dets[index-1].push_back(tmpRow);

    }
}

int main(int argc, char *argv[])
{
    DeepSort sorter(saved_model_dir);

    loadFilenames(image_pathname);
    loadDetections(detections_filename);

    int current_index = -1;
    bool running = true;
        while (running) {
        current_index++;
        if (current_index < image_filenames.size()) {
            try {
                std::cout << image_filenames[current_index] << std::endl;

                std::stringstream str;
                str << image_pathname << "/" << image_filenames[current_index];
                cv::Mat image = cv::imread(str.str());

                DETECTIONS& detections = file_dets[current_index];

                std::vector<RESULT_DATA> result;
                sorter.sort(image, detections, &result);
                for (int i = 0; i < result.size(); i++) {
                    DETECTBOX box = result[i].second;
                    cv::Rect rect = cv::Rect(box[0], box[1], box[2], box[3]);
                    cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1);
                    cv::putText(image, std::to_string(result[i].first), cv::Point(rect.x, rect.y),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }

                cv::imshow("image", image);
                cv::waitKey(50);
            }
            catch (const std::exception& e) {
                std::cout << "ERROR: " << e.what() << std::endl;
            }

        }
        else {
            running = false;
        }
    }




    return 0;
}
