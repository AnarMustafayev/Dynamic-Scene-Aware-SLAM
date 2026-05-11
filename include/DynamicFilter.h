#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>

namespace ORB_SLAM3 {

class Frame;
class MapPoint;

/**
 * DynamicFilter  —  cascade dynamic-point rejection pipeline
 * ════════════════════════════════════════════════════════════
 *
 * Per-match decision tree (evaluated independently for EACH match):
 *
 *  Stage 1 │ Epipolar  (Sampson distance vs chi² threshold)
 *          │   PASS  (dist < 3.84) → point is DEFINITELY static
 *          │            → fast-path KEEP, skip stages 2 & 3
 *          │   FAIL  (dist ≥ 3.84) → geometrically suspicious → go to Stage 2
 *          │   SKIP  (F empty)     → no velocity / pure-rotation → go to Stage 2
 *          │
 *  Stage 2 │ YOLO semantic mask
 *          │   IN mask  → point is on a detected human/vehicle → REJECT
 *          │   NOT in mask / no mask → go to Stage 3
 *          │
 *  Stage 3 │ Optical-flow mask  (Step 5 — pass cv::Mat() to skip)
 *          │   IN mask  → flow confirms dynamic motion → REJECT
 *          │   NOT in mask / no mask → KEEP (ambiguous, conservative)
 *
 * Ambiguous points (failed epipolar but not confirmed by YOLO or flow) are
 * KEPT intentionally — better to keep a suspicious point than lose tracking
 * on a dynamic-dominant frame.
 */
class DynamicFilter {
public:
    /**
     * Per-stage diagnostics returned from FilterFrame().
     * Useful for thesis logging and performance analysis.
     */
    struct FilterStats {
        int n_epipolar_failed = 0;  ///< failed Sampson test (proceeded to Stage 2)
        int n_yolo            = 0;  ///< rejected by YOLO semantic mask
        int n_flow            = 0;  ///< rejected by optical-flow mask
        int n_ambiguous       = 0;  ///< failed epipolar, not confirmed → kept
        int n_kept            = 0;  ///< surviving matches after all stages
    };

    /**
     * @param epipolar_threshold  Sampson chi² threshold.
     *                            Below → static fast-path keep.
     *                            Above → suspicious, proceed to Stage 2.
     *                            3.84 = 95% confidence interval (1 DOF).
     */
    explicit DynamicFilter(float epipolar_threshold = 3.84f);

    /**
     * Cascade filter.  Evaluates each match through the decision tree
     * and nulls out dynamic MapPoints in current_frame in-place.
     *
     * @param current_frame  Frame under evaluation (mvpMapPoints modified).
     * @param last_frame     Previous frame for epipolar correspondence lookup.
     * @param F              Fundamental matrix CV_32F 3×3.
     *                       Pass empty Mat to skip Stage 1.
     * @param yolo_mask      Binary mask CV_8U (1=dynamic) from YOLODetector.
     *                       Pass empty Mat to skip Stage 2.
     * @param flow_mask      Binary mask CV_8U (1=dynamic) from FlowChecker.
     *                       Pass empty Mat to skip Stage 3.
     * @param nmatches       Decremented for each rejected match.
     * @return               Per-stage statistics.
     */
    FilterStats FilterFrame(
        Frame&         current_frame,
        const Frame&   last_frame,
        const cv::Mat& F,
        const cv::Mat& yolo_mask,
        const cv::Mat& flow_mask,
        int&           nmatches
    );

    float GetEpipolarThreshold() const { return mEpipolarThreshold; }

private:
    /** Resize mask to (w × h) with INTER_NEAREST; returns original if already correct. */
    static cv::Mat ResizeMaskIfNeeded(const cv::Mat& mask, int w, int h);

    /**
     * Sampson distance:  (x2ᵀ F x1)² / (Fx1[0]²+Fx1[1]²+Fᵀx2[0]²+Fᵀx2[1]²)
     * @param p1  keypoint in last frame
     * @param p2  keypoint in current frame
     */
    static float SampsonDistance(cv::Point2f p1, cv::Point2f p2,
                                 const cv::Mat& F);

    float mEpipolarThreshold;  ///< chi² threshold (3.84 → 95% CI, 1 DOF)
};

} // namespace ORB_SLAM3
