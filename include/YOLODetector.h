#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <string>
#include <vector>
#include <memory>

namespace ORB_SLAM3 {

class YOLODetector {
public:
    // model_path  : full path to your exported yolo11n_aerial.onnx file
    // input_size  : must match imgsz used during export (320)
    // conf_thresh : confidence threshold for detections (0.0 - 1.0)
    // mask_thresh : threshold to binarize soft segmentation mask (0.0 - 1.0)
    YOLODetector(const std::string& model_path,
                 int   input_size   = 320,
                 float conf_thresh  = 0.40f,
                 float mask_thresh  = 0.50f);

    ~YOLODetector();

    // Start the async inference thread
    // Call this once after construction, before pushing frames
    void Start();

    // Stop the async inference thread cleanly
    // Called automatically in destructor
    void Stop();

    // Push a new frame for inference — NON-BLOCKING
    // Drops oldest frame if internal queue is full
    // Call this every tracking iteration from Tracking.cc
    void PushFrame(const cv::Mat& frame);

    // Get the latest binary dynamic mask — NON-BLOCKING
    // Returns empty Mat if no inference has completed yet
    // Caller must handle empty Mat gracefully (skip YOLO check)
    // Mask values: 1 = dynamic region, 0 = static region
    cv::Mat GetLatestMask();

    // Returns true if at least one inference has completed
    bool HasMask() const;

    // Runtime statistics — useful for thesis FPS reporting
    double GetAvgInferenceMs() const;
    int    GetFramesProcessed() const;

private:
    // ── Internal Types ────────────────────────────────────────

    // One decoded detection proposal
    struct Detection {
        cv::Rect             box;          // bounding box in original image coords
        float                confidence;   // class confidence score
        int                  class_id;     // 0=human, 1=vehicle
        std::vector<float>   mask_coeffs;  // 32 mask coefficients
    };

    // ── Internal Methods ──────────────────────────────────────

    // Main loop running inside mThread
    void InferenceLoop();

    // Resize + normalize + HWC→CHW layout → float32 vector
    std::vector<float> Preprocess(const cv::Mat& frame);

    // Full postprocessing pipeline → binary mask
    // box_data   : pointer to output0 tensor  shape [1, 38, 2100]
    // proto_data : pointer to output1 tensor  shape [1, 32, 80, 80]
    // orig_w/h   : original frame dimensions before resize
    cv::Mat PostprocessSegmentation(
        float* box_data,
        float* proto_data,
        int orig_w,
        int orig_h
    );

    // Parse output0 into Detection structs
    // Filters by confidence and dynamic class IDs only
    std::vector<Detection> DecodeProposals(
        float* box_data,
        int    num_proposals,
        int orig_w,
        int orig_h
    );

    // Remove duplicate detections using Non-Maximum Suppression
    std::vector<Detection> ApplyNMS(
        std::vector<Detection>& detections,
        float iou_threshold = 0.45f
    );

    // Combine 32 mask coefficients with 32 prototype masks (80x80 each)
    // Returns binary mask resized to orig_w x orig_h
    cv::Mat CombineMaskWithProtos(
        const std::vector<float>& coeffs,
        float*          proto_data,
        const cv::Rect& box_crop,
        int orig_w,
        int orig_h
    );

    // ── ONNX Runtime ──────────────────────────────────────────

    Ort::Env                          mEnv;
    std::unique_ptr<Ort::Session>     mpSession;
    Ort::SessionOptions               mSessionOptions;
    Ort::AllocatorWithDefaultOptions  mAllocator;

    // Tensor names — verified from your model:
    // input  : "images"
    // output0: "output0"  shape [1, 38, 2100]
    // output1: "output1"  shape [1, 32, 80, 80]
    const char* mInputName   = "images";
    const char* mOutputName0 = "output0";
    const char* mOutputName1 = "output1";

    // ── Model Constants ───────────────────────────────────────
    // These are fixed by how your model was exported.
    // DO NOT change unless you re-export with different settings.

    // Number of proposals from YOLO detection head
    // 320x320 input → 2100 proposals  (640x640 → 8400)
    static constexpr int NUM_PROPOSALS = 2100;

    // Number of mask prototype channels
    static constexpr int NUM_PROTOS    = 32;

    // Spatial size of prototype masks from output1
    // Your model: 80x80  (standard 640 model uses 160x160)
    static constexpr int PROTO_SIZE    = 80;

    // Number of object classes in your model
    // output0 row = 4 (box) + NUM_CLASSES + NUM_PROTOS = 38
    // 38 - 4 - 32 = 2 classes
    static constexpr int NUM_CLASSES   = 2;

    // Your trained classes:
    // 0: human   (pedestrians, people, cyclists)
    // 1: vehicle (cars, trucks, motorcycles, vans)
    // Both are dynamic — detect both
    const std::vector<int> DYNAMIC_CLASS_IDS = {0, 1};

    // ── Configuration ─────────────────────────────────────────

    std::string mModelPath;
    int         mInputSize;       // 320 — fixed at export time
    float       mConfThreshold;   // default 0.40
    float       mMaskThreshold;   // default 0.50

    // Process every Nth frame to save compute
    // YOLO async thread runs at ~10fps when mFrameSkip=3 and SLAM runs at 30fps
    int mFrameSkip  = 3;
    int mFrameCount = 0;

    // ── Threading ─────────────────────────────────────────────

    std::thread       mThread;
    std::atomic<bool> mRunning{false};

    // Input frame queue — max 2 frames buffered
    // Oldest is dropped if tracking pushes faster than YOLO processes
    std::queue<cv::Mat> mFrameQueue;
    std::mutex          mQueueMutex;
    static constexpr int MAX_QUEUE_SIZE = 2;

    // Output mask — written by inference thread, read by tracking thread
    cv::Mat           mLatestMask;
    mutable std::mutex mMaskMutex;
    std::atomic<bool>  mbHasMask{false};

    // ── Statistics ────────────────────────────────────────────

    double             mAvgInferenceMs{0.0};
    int                mFramesProcessed{0};
    mutable std::mutex mStatsMutex;
};

} // namespace ORB_SLAM3