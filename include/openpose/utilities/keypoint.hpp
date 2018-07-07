#ifndef OPENPOSE_UTILITIES_KEYPOINT_HPP
#define OPENPOSE_UTILITIES_KEYPOINT_HPP

#include <vector>
#include <openpose/core/array.hpp>
#include <openpose/core/rectangle.hpp>
#include <opencv2/opencv.hpp>

extern int n, draw_suffix;
extern cv::Point head_top[];
extern cv::Point head_bottom[];
extern cv::Point rWest[];
extern cv::Point lWest[];
extern cv::Point elbow[];
extern cv::Point wrist[];
extern cv::Point foot[];
extern cv::Point RShoulder[];
extern cv::Point RElbow[];
extern cv::Point RWrist[];
extern cv::Point LShoulder[];
extern cv::Point LElbow[];
extern cv::Point LWrist[];
extern cv::Point RKnee[];
extern cv::Point RAnkle[];
extern cv::Point LKnee[];
extern cv::Point LAnkle[];
extern cv::Point drawLine[];

namespace op
{
    float getDistance(const float* keypointPtr, const int elementA, const int elementB);

    void scaleKeypoints(Array<float>& keypoints, const float scale);

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY);

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX, const float offsetY);

    void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints, const std::vector<unsigned int>& pairs,
                            const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                            const float threshold);

    Rectangle<float> getKeypointsRectangle(const float* keypointPtr, const int numberKeypoints, const float threshold);

    float getKeypointsArea(const float* keypointPtr, const int numberKeypoints, const float threshold);

    int getBiggestPerson(const Array<float>& keypoints, const float threshold);
}

#endif // OPENPOSE_UTILITIES_KEYPOINT_HPP
