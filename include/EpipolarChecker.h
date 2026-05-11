#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace ORB_SLAM3 {

class EpipolarChecker {
public:
    explicit EpipolarChecker(float sampson_threshold = 2.0f);

    // Compute fundamental matrix from relative pose and intrinsics
    // R: 3x3 rotation (CV_32F), t: 3x1 translation (CV_32F)
    // K: 3x3 camera intrinsics (CV_32F)
    cv::Mat ComputeFundamentalMatrix(
        const cv::Mat& R,
        const cv::Mat& t,
        const cv::Mat& K
    );

    // Check all keypoint pairs — returns true = static, false = dynamic
    std::vector<bool> CheckKeypoints(
        const std::vector<cv::KeyPoint>& pts1,
        const std::vector<cv::KeyPoint>& pts2,
        const cv::Mat& F
    );

    // Check single point pair
    bool IsStaticPoint(
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const cv::Mat& F
    );

    float ComputeSampsonDistance(
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const cv::Mat& F
    );

    void SetThreshold(float threshold) { mThreshold = threshold; }
    float GetThreshold() const { return mThreshold; }

private:
    float mThreshold;
};

} // namespace ORB_SLAM3