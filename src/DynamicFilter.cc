#include "DynamicFilter.h"
#include "Frame.h"
#include "MapPoint.h"
#include "System.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>

namespace ORB_SLAM3 {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

DynamicFilter::DynamicFilter(float epipolar_threshold)
    : mEpipolarThreshold(epipolar_threshold)
{}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat DynamicFilter::ResizeMaskIfNeeded(const cv::Mat& mask, int w, int h)
{
    if (mask.empty())                        return mask;
    if (mask.cols == w && mask.rows == h)    return mask;
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
    return resized;
}

float DynamicFilter::SampsonDistance(cv::Point2f p1, cv::Point2f p2,
                                     const cv::Mat& F)
{
    // Homogeneous coordinates
    cv::Mat x1 = (cv::Mat_<float>(3,1) << p1.x, p1.y, 1.0f);
    cv::Mat x2 = (cv::Mat_<float>(3,1) << p2.x, p2.y, 1.0f);

    cv::Mat Fx1  = F * x1;
    cv::Mat Ftx2 = F.t() * x2;
    float   num  = static_cast<float>(x2.dot(Fx1));   // x2ᵀ F x1

    float denom = Fx1.at<float>(0)  * Fx1.at<float>(0)
                + Fx1.at<float>(1)  * Fx1.at<float>(1)
                + Ftx2.at<float>(0) * Ftx2.at<float>(0)
                + Ftx2.at<float>(1) * Ftx2.at<float>(1);

    return (denom > 1e-9f) ? (num * num / denom) : 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterFrame  —  cascade decision tree, per-match
// ─────────────────────────────────────────────────────────────────────────────

DynamicFilter::FilterStats DynamicFilter::FilterFrame(
    Frame&         current_frame,
    const Frame&   last_frame,
    const cv::Mat& F,
    const cv::Mat& yolo_mask,
    const cv::Mat& flow_mask,
    int&           nmatches)
{
    FilterStats stats;

    const int W = current_frame.mnMaxX;
    const int H = current_frame.mnMaxY;

    // ── Pre-scale masks to frame resolution (done once, not per-keypoint) ────
    const cv::Mat yolo_m = ResizeMaskIfNeeded(yolo_mask, W, H);
    const cv::Mat flow_m = ResizeMaskIfNeeded(flow_mask, W, H);

    // ── Stage 1 setup: reverse lookup  MapPoint* → index in last_frame ───────
    std::unordered_map<MapPoint*, int> mpToLastIdx;
    if (!F.empty())
    {
        mpToLastIdx.reserve(static_cast<size_t>(last_frame.N));
        for (int j = 0; j < last_frame.N; j++)
            if (last_frame.mvpMapPoints[j])
                mpToLastIdx[last_frame.mvpMapPoints[j]] = j;
    }

    // ── Per-match cascade ─────────────────────────────────────────────────────
    for (int i = 0; i < current_frame.N; i++)
    {
        MapPoint* pMP = current_frame.mvpMapPoints[i];
        if (!pMP) continue;

        const cv::Point2f& kp = current_frame.mvKeys[i].pt;
        const int px = std::min((int)kp.x, W - 1);
        const int py = std::min((int)kp.y, H - 1);

        // ── Stage 1: Epipolar  ────────────────────────────────────────────────
        //
        // Sampson distance < threshold  →  point is DEFINITELY static
        //   Fast-path KEEP: skip stages 2 & 3 entirely.
        //
        // Sampson distance ≥ threshold  →  geometrically suspicious
        //   Proceed to Stage 2 for secondary confirmation.
        //
        // F is empty  →  epipolar skipped (no velocity / pure rotation)
        //   Proceed directly to Stage 2.
        bool epipolar_failed = false;

        if (!F.empty())
        {
            auto it = mpToLastIdx.find(pMP);
            if (it != mpToLastIdx.end())
            {
                float sd = SampsonDistance(
                    last_frame.mvKeys[it->second].pt, kp, F);

                if (sd < mEpipolarThreshold)
                {
                    // ✓ PASS: geometrically consistent with motion model
                    //   This is the hot path — most static points exit here.
                    continue;   // KEEP — skip stages 2 & 3
                }

                epipolar_failed = true;
                stats.n_epipolar_failed++;
            }
            // If MapPoint not found in last_frame → falls through to Stage 2
        }

        // Reach here only if:
        //   (a) F is empty — epipolar skipped entirely
        //   (b) epipolar ran and FAILED for this point

        // ── Stage 2: YOLO semantic mask  ─────────────────────────────────────
        //
        // Epipolar already flagged this point as suspicious.
        // YOLO provides semantic confirmation: is it actually on a human/vehicle?
        //   YES (in mask)  → confirmed dynamic → REJECT immediately, skip Stage 3
        //   NO / no mask   → not confirmed semantically → proceed to Stage 3
        if (!yolo_m.empty())
        {
            if (yolo_m.at<uchar>(py, px) > 0)
            {
                // Epipolar failed + YOLO confirms → definitely dynamic
                current_frame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                nmatches--;
                stats.n_yolo++;
                continue;   // REJECT — skip Stage 3
            }
        }

        // ── Stage 3: Optical-flow mask  (Step 5 — placeholder) ───────────────
        //
        // Final arbiter: dense optical-flow consistency.
        // If a point's observed flow disagrees with the rigid-body prediction,
        // it is moving → REJECT.
        //   IN mask  → flow confirms dynamic motion → REJECT
        //   no mask  → inconclusive → KEEP (conservative)
        if (!flow_m.empty())
        {
            if (flow_m.at<uchar>(py, px) > 0)
            {
                // Epipolar failed + flow confirms → dynamic
                current_frame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                nmatches--;
                stats.n_flow++;
                continue;   // REJECT
            }
        }

        // ── Ambiguous: epipolar failed but no secondary filter confirmed ───────
        //
        // Conservative decision: KEEP.
        // On dynamic-dominant frames, keeping ambiguous points is safer than
        // aggressively removing them and losing all tracking matches.
        if (epipolar_failed)
            stats.n_ambiguous++;
        // else: F was empty, no secondary filters → unconditionally kept
    }

    stats.n_kept = nmatches;

    // ── Consolidated per-frame log ────────────────────────────────────────────
    const int total_rejected = stats.n_yolo + stats.n_flow;
    if (stats.n_epipolar_failed > 0 || total_rejected > 0)
        Verbose::PrintMess(
            "DynamicFilter: epi_failed=" + std::to_string(stats.n_epipolar_failed) +
            "  rejected=[YOLO=" + std::to_string(stats.n_yolo) +
            " Flow=" + std::to_string(stats.n_flow) + "]" +
            "  ambiguous=" + std::to_string(stats.n_ambiguous) +
            "  kept=" + std::to_string(stats.n_kept),
            Verbose::VERBOSITY_NORMAL);

    return stats;
}

} // namespace ORB_SLAM3
