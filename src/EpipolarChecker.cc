#include "EpipolarChecker.h"
#include <cmath>

namespace ORB_SLAM3 {

EpipolarChecker::EpipolarChecker(float sampson_threshold)
    : mThreshold(sampson_threshold) {}

cv::Mat EpipolarChecker::ComputeFundamentalMatrix(
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Mat& K)
{
    // Skew-symmetric matrix [t]x
    // Bug fix: was missing << operator
    cv::Mat tx = (cv::Mat_<float>(3,3) <<
         0,               -t.at<float>(2),  t.at<float>(1),
         t.at<float>(2),   0,              -t.at<float>(0),
        -t.at<float>(1),   t.at<float>(0),  0);

    // F = K^-T * [t]x * R * K^-1
    cv::Mat Kinv = K.inv();
    cv::Mat F = Kinv.t() * tx * R * Kinv;
    return F;
}

float EpipolarChecker::ComputeSampsonDistance(
    const cv::Point2f& p1,
    const cv::Point2f& p2,
    const cv::Mat& F)
{
    // Homogeneous coordinates
    cv::Mat pt1 = (cv::Mat_<float>(3,1) << p1.x, p1.y, 1.0f);
    cv::Mat pt2 = (cv::Mat_<float>(3,1) << p2.x, p2.y, 1.0f);

    cv::Mat Fpt1  = F * pt1;          // 3x1
    cv::Mat pt2tF = pt2.t() * F;      // 1x3  (== F^T * pt2)^T

    // Sampson distance = (x2^T F x1)^2 / ((Fx1)_0^2 + (Fx1)_1^2 + (F^Tx2)_0^2 + (F^Tx2)_1^2)
    // Bug fix: cv::Mat cannot be cast to float directly — must use .at<float>(0,0)
    cv::Mat num_mat = pt2.t() * F * pt1;   // 1x1 matrix
    float num = num_mat.at<float>(0, 0);

    float den = Fpt1.at<float>(0)  * Fpt1.at<float>(0)
              + Fpt1.at<float>(1)  * Fpt1.at<float>(1)
              + pt2tF.at<float>(0) * pt2tF.at<float>(0)
              + pt2tF.at<float>(1) * pt2tF.at<float>(1);

    if (den < 1e-8f) return 999.0f;   // degenerate case
    return (num * num) / den;
}

bool EpipolarChecker::IsStaticPoint(
    const cv::Point2f& p1,
    const cv::Point2f& p2,
    const cv::Mat& F)
{
    try {
        float dist = ComputeSampsonDistance(p1, p2, F);
        return dist < mThreshold;
    } catch (...) {
        return true;  // fail safe — keep point if error
    }
}

std::vector<bool> EpipolarChecker::CheckKeypoints(
    const std::vector<cv::KeyPoint>& pts1,
    const std::vector<cv::KeyPoint>& pts2,
    const cv::Mat& F)
{
    // Fail safe — if inputs invalid, keep all points
    if (pts1.size() != pts2.size() || F.empty() || pts1.empty())
        return std::vector<bool>(pts1.size(), true);

    std::vector<bool> results(pts1.size());
    for (size_t i = 0; i < pts1.size(); i++) {
        results[i] = IsStaticPoint(pts1[i].pt, pts2[i].pt, F);
    }
    return results;
}

} // namespace ORB_SLAM3