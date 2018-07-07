#include <limits> // std::numeric_limits
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp> // cv::line, cv::circle
#include <opencv2/highgui/highgui.hpp> // cv::line, cv::circle
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>

#define MAX_SUFFIX 100

//int head_x;
//int head_y;
cv::Point head_top[64], head_bottom[64];
cv::Point rWest[64], lWest[64];
cv::Point foot[64];
cv::Point RShoulder[64];
cv::Point RElbow[64];
cv::Point RWrist[64];
cv::Point LShoulder[64];
cv::Point LElbow[64];
cv::Point LWrist[64];
cv::Point RKnee[64];
cv::Point RAnkle[64];
cv::Point LKnee[64];
cv::Point LAnkle[64];
cv::Point drawLine[128];

int n, draw_suffix = 0;
int zone_left = -1, zone_right = -1;

namespace op
{
    const std::string errorMessage = "The Array<float> is not a RGB image. This function is only for array of dimension: [sizeA x sizeB x 3].";

    float getDistance(const float* keypointPtr, const int elementA, const int elementB)
    {
        try
        {
            const auto pixelX = keypointPtr[elementA*3] - keypointPtr[elementB*3];
            const auto pixelY = keypointPtr[elementA*3+1] - keypointPtr[elementB*3+1];
            return std::sqrt(pixelX*pixelX+pixelY*pixelY);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.f;
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scale)
    {
        try
        {
            scaleKeypoints(keypoints, scale, scale);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keypoints.empty() && keypoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keypoints.getSize(0);
                const auto numberParts = keypoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = 3*(person*numberParts + part);
                        keypoints[finalIndex] *= scaleX;
                        keypoints[finalIndex+1] *= scaleY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX, const float offsetY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keypoints.empty() && keypoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keypoints.getSize(0);
                const auto numberParts = keypoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = keypoints.getSize(2)*(person*numberParts + part);
                        keypoints[finalIndex] = keypoints[finalIndex] * scaleX + offsetX;
                        keypoints[finalIndex+1] = keypoints[finalIndex+1] * scaleY + offsetY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints, const std::vector<unsigned int>& pairs,
                            const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                            const float threshold)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Array<float> --> cv::Mat
                auto frame = frameArray.getCvMat();

                // Security check
                if (frame.dims != 3 || frame.size[0] != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);

                // Get frame channels
                const auto width = frame.size[2];
                const auto height = frame.size[1];
                const auto area = width * height;
                cv::Mat frameB{height, width, CV_32FC1, &frame.data[0]};
                cv::Mat frameG{height, width, CV_32FC1, &frame.data[area * sizeof(float) / sizeof(uchar)]};
                cv::Mat frameR{height, width, CV_32FC1, &frame.data[2 * area * sizeof(float) / sizeof(uchar)]};

                // Parameters
                const auto lineType = 8;
                const auto shift = 0;
                const auto thresholdRectangle = 0.1f;
				const auto numberColors = colors.size();
                const auto numberKeypoints = keypoints.getSize(1);
                const auto areaKeypoints = numberKeypoints * keypoints.getSize(2);

                // Keypoints
                for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
                {
					/*’Ç‰Á*/
					n = person;
                    const auto personRectangle = getKeypointsRectangle(&keypoints[person*areaKeypoints], keypoints.getSize(1), thresholdRectangle);
                    const auto ratioAreas = fastMin(1.f, fastMax(personRectangle.width/(float)width, personRectangle.height/(float)height));
                    // Size-dependent variables
					const auto thicknessCircle = -1;//fastMax(intRound(std::sqrt(area)*thicknessCircleRatio * ratioAreas), 2);
					const auto thicknessLine = 20;//intRound(thicknessCircle * thicknessLineRatioWRTCircle);
					const auto radius = 30;//thicknessCircle / 2;

                    // Draw lines
                    for (auto pair = 0 ; pair < pairs.size() ; pair+=2)
                    {
						const auto index1 = (person * keypoints.getSize(1) + pairs[pair]) * keypoints.getSize(2);
						const auto index2 = (person * keypoints.getSize(1) + pairs[pair + 1]) * keypoints.getSize(2);
						if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
						{
							const auto colorIndex = pair / 2 * 3;
							//const cv::Scalar color{ colors[colorIndex % numberColors],
							//						colors[(colorIndex + 1) % numberColors],
							//						colors[(colorIndex + 2) % numberColors] };
							const cv::Point keypoint1{ intRound(keypoints[index1]), intRound(keypoints[index1 + 1]) };
							const cv::Point keypoint2{ intRound(keypoints[index2]), intRound(keypoints[index2 + 1]) };
						}
                    }

                    // Draw circles
					for (auto part = 0; part < keypoints.getSize(1); part++)
					{
						const auto faceIndex = (person * keypoints.getSize(1) + part) * keypoints.getSize(2);
						if (keypoints[faceIndex + 2] > threshold)
						{
							const auto colorIndex = part * 3;
							//const cv::Scalar color{ colors[colorIndex % numberColors],
							//	   colors[(colorIndex + 1) % numberColors],
							//	   colors[(colorIndex + 2) % numberColors] };
							const cv::Scalar color{255,0,0};
							const cv::Point center{ intRound(keypoints[faceIndex]), intRound(keypoints[faceIndex + 1]) };
							//if (LWrist[n].y < RWrist[n].y)  //¶Žè‚ðã‚°‚é‚ÆÁ‚¦‚é
							//	draw_suffix = 0;
							//for (int i = 1; i < draw_suffix; i++) {
							//	cv::line(frameR, drawLine[i-1], drawLine[i], color[0], thicknessLine, CV_AA, shift);
							//	cv::line(frameG, drawLine[i-1], drawLine[i], color[1], thicknessLine, CV_AA, shift);
							//	cv::line(frameB, drawLine[i-1], drawLine[i], color[2], thicknessLine, CV_AA, shift);
							//}
							if (part == 0)
								head_top[n] = center;
							else if (part == 1) 
								head_bottom[n] = center;
							else if (part == 2)
								RShoulder[n] = center;
							//else if (part == 3)
							//	RElbow[n] = center;
							else if (part == 4){
								RWrist[n] = center;
								//if(draw_suffix >= MAX_SUFFIX)
								//	draw_suffix = 0;
								//drawLine[draw_suffix].x = center.x;
								//drawLine[draw_suffix++].y = center.y - 90;
							}
							else if (part == 5)
								LShoulder[n] = center;
							//else if (part == 6)
							//	LElbow[n] = center;
							else if (part == 7)
								LWrist[n] = center;
							//else if (part == 8)
							//	rWest[n] = center;
							//else if (part == 9)
							//	RKnee[n] = center;
							//else if (part == 10)
							//	RAnkle[n] = center;
							else if (part == 11)
								lWest[n] = center;
							else if (part == 12)
								LKnee[n] = center;
							else if (part == 13)
								LAnkle[n] = center;
							//if(zone_left == -1 && zone_right == -1){
							//	zone_left = LShoulder[n].x-500;
							//	zone_right = LShoulder[n].x-300;
							//}
							int zone_top = (LShoulder[n].y + lWest[n].y) / 2;
							cv::rectangle(frameR, cv::Point(650 ,zone_top), cv::Point(800 ,LKnee[n].y), color[0], 5,8);
							cv::rectangle(frameG, cv::Point(650 ,zone_top), cv::Point(800 ,LKnee[n].y), color[0], 5,8);
							cv::rectangle(frameB, cv::Point(650 ,zone_top), cv::Point(800 ,LKnee[n].y), color[0], 5,8);
						}
					}
                }
            }
        }

        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Rectangle<float> getKeypointsRectangle(const float* keypointPtr, const int numberKeypoints, const float threshold)
    {
        try
        {
            if (numberKeypoints < 1)
                error("Number body parts must be > 0", __LINE__, __FUNCTION__, __FILE__);

            float minX = std::numeric_limits<float>::max();
            float maxX = 0.f;
            float minY = minX;
            float maxY = maxX;
            for (auto part = 0 ; part < numberKeypoints ; part++)
            {
                const auto score = keypointPtr[3*part + 2];
                if (score > threshold)
                {
                    const auto x = keypointPtr[3*part];
                    const auto y = keypointPtr[3*part + 1];
                    // Set X
                    if (maxX < x)
                        maxX = x;
                    if (minX > x)
                        minX = x;
                    // Set Y
                    if (maxY < y)
                        maxY = y;
                    if (minY > y)
                        minY = y;
                }
            }
            if (maxX >= minX && maxY >= minY)
                return Rectangle<float>{minX, minY, maxX-minX, maxY-minY};
            else
                return Rectangle<float>{};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<float>{};
        }
    }

    float getKeypointsArea(const float* keypointPtr, const int numberKeypoints, const float threshold)
    {
        try
        {
            return getKeypointsRectangle(keypointPtr, numberKeypoints, threshold).area();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    int getBiggestPerson(const Array<float>& keypoints, const float threshold)
    {
        try
        {
            if (!keypoints.empty())
            {
                const auto numberPeople = keypoints.getSize(0);
                const auto numberKeypoints = keypoints.getSize(1);
                const auto area = numberKeypoints * keypoints.getSize(2);
                auto biggestPoseIndex = -1;
                auto biggestArea = -1.f;
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    const auto newPersonArea = getKeypointsArea(&keypoints[person*area], numberKeypoints, threshold);
                    if (newPersonArea > biggestArea)
                    {
                        biggestArea = newPersonArea;
                        biggestPoseIndex = person;
                    }
                }
                return biggestPoseIndex;
            }
            else
                return -1;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }
}
