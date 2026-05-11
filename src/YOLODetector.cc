#include "YOLODetector.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <cmath>

namespace ORB_SLAM3 {

// ─────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────

YOLODetector::YOLODetector(const std::string& model_path,
                             int input_size,
                             float conf_thresh,
                             float mask_thresh)
    : mEnv(ORT_LOGGING_LEVEL_WARNING, "YOLODetector"),
      mModelPath(model_path),
      mInputSize(input_size),
      mConfThreshold(conf_thresh),
      mMaskThreshold(mask_thresh)
{
    // Configure session for performance
    mSessionOptions.SetIntraOpNumThreads(2);
    mSessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Try to enable CUDA — falls back to CPU if not available
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        mSessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[YOLODetector] Using CUDA provider" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[YOLODetector] CUDA not available, using CPU: "
                  << e.what() << std::endl;
    }

    // Load ONNX model
    mpSession = std::make_unique<Ort::Session>(
        mEnv,
        mModelPath.c_str(),
        mSessionOptions
    );

    std::cout << "[YOLODetector] Model loaded: " << mModelPath << std::endl;
    std::cout << "[YOLODetector] Input size: "
              << mInputSize << "x" << mInputSize << std::endl;
}

YOLODetector::~YOLODetector() {
    Stop();
}

// ─────────────────────────────────────────────────────────────
// Thread Control
// ─────────────────────────────────────────────────────────────

void YOLODetector::Start() {
    mRunning = true;
    mThread = std::thread(&YOLODetector::InferenceLoop, this);
    std::cout << "[YOLODetector] Async thread started" << std::endl;
}

void YOLODetector::Stop() {
    mRunning = false;
    if (mThread.joinable())
        mThread.join();
}

// ─────────────────────────────────────────────────────────────
// Public Interface
// ─────────────────────────────────────────────────────────────

void YOLODetector::PushFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(mQueueMutex);

    // Drop oldest frame if queue is full — never block tracking thread
    if ((int)mFrameQueue.size() >= MAX_QUEUE_SIZE)
        mFrameQueue.pop();

    mFrameQueue.push(frame.clone());
}

cv::Mat YOLODetector::GetLatestMask() {
    std::lock_guard<std::mutex> lock(mMaskMutex);
    if (!mbHasMask)
        return cv::Mat();   // empty — DynamicFilter handles this
    return mLatestMask.clone();
}

bool YOLODetector::HasMask() const {
    return mbHasMask.load();
}

double YOLODetector::GetAvgInferenceMs() const {
    std::lock_guard<std::mutex> lock(mStatsMutex);
    return mAvgInferenceMs;
}

int YOLODetector::GetFramesProcessed() const {
    std::lock_guard<std::mutex> lock(mStatsMutex);
    return mFramesProcessed;
}

// ─────────────────────────────────────────────────────────────
// Async Inference Loop
// ─────────────────────────────────────────────────────────────

void YOLODetector::InferenceLoop() {
    while (mRunning) {
        cv::Mat frame;

        // Try to get a frame — release lock before any sleeping
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            if (!mFrameQueue.empty()) {
                frame = mFrameQueue.front();
                mFrameQueue.pop();
            }
        }   // ← mutex released here, before any blocking call

        if (frame.empty()) {
            // Nothing in queue — sleep WITHOUT holding the lock
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        mFrameCount++;

        // Skip frames to save compute
        if (mFrameCount % mFrameSkip != 0)
            continue;

        if (frame.empty())
            continue;

        try {
            auto t_start = std::chrono::high_resolution_clock::now();

            int orig_w = frame.cols;
            int orig_h = frame.rows;

            // ── Step 1: Preprocess ─────────────────────────
            std::vector<float> input_data = Preprocess(frame);

            // ── Step 2: Build input tensor ─────────────────
            std::array<int64_t, 4> input_shape{
                1, 3, mInputSize, mInputSize
            };

            Ort::MemoryInfo mem_info =
                Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtArenaAllocator,
                    OrtMemType::OrtMemTypeDefault
                );

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info,
                input_data.data(),
                input_data.size(),
                input_shape.data(),
                input_shape.size()
            );

            // ── Step 3: Run inference ──────────────────────
            std::vector<const char*> input_names  = {mInputName};
            std::vector<const char*> output_names = {mOutputName0,
                                                      mOutputName1};

            std::vector<Ort::Value> output_tensors = mpSession->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),  &input_tensor, 1,
                output_names.data(), output_names.size()
            );

            // ── Step 4: Get output pointers ────────────────
            float* box_data   = output_tensors[0].GetTensorMutableData<float>();
            float* proto_data = output_tensors[1].GetTensorMutableData<float>();

            // ── Step 5: Postprocess → binary mask ──────────
            cv::Mat mask = PostprocessSegmentation(
                box_data, proto_data, orig_w, orig_h
            );

            // ── Step 6: Write to shared buffer ────────────
            {
                std::lock_guard<std::mutex> lock(mMaskMutex);
                mLatestMask = mask;
                mbHasMask   = true;
            }

            // ── Step 7: Update statistics ──────────────────
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(
                t_end - t_start).count();

            {
                std::lock_guard<std::mutex> lock(mStatsMutex);
                mAvgInferenceMs = 0.9 * mAvgInferenceMs + 0.1 * ms;
                mFramesProcessed++;
            }

        } catch (const std::exception& e) {
            std::cerr << "[YOLODetector] Inference error: "
                      << e.what() << std::endl;
            // Continue — never crash ORB-SLAM3
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────

std::vector<float> YOLODetector::Preprocess(const cv::Mat& frame) {
    cv::Mat resized, rgb, float_img;

    // Resize to model input size
    cv::resize(frame, resized,
               cv::Size(mInputSize, mInputSize));

    // Convert to RGB if needed
    if (resized.channels() == 1)
        cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
    else if (resized.channels() == 4)
        cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
    else
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Normalize to [0, 1]
    rgb.convertTo(float_img, CV_32F, 1.0f / 255.0f);

    // HWC → CHW (channels first for ONNX)
    int sz = mInputSize * mInputSize;
    std::vector<float> data(3 * sz);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    memcpy(data.data(),           channels[0].data, sz * sizeof(float));
    memcpy(data.data() + sz,      channels[1].data, sz * sizeof(float));
    memcpy(data.data() + 2 * sz,  channels[2].data, sz * sizeof(float));

    return data;
}

// ─────────────────────────────────────────────────────────────
// Decode Proposals from output0
// output0 shape: [1, 116, 8400]
// 116 = 4 (xywh) + 80 (classes) + 32 (mask coeffs)
// ─────────────────────────────────────────────────────────────

std::vector<YOLODetector::Detection> YOLODetector::DecodeProposals(
    float* box_data,
    int num_proposals,
    int orig_w,
    int orig_h)
{
    std::vector<Detection> detections;

    // output0 is transposed: [116, 8400] after removing batch dim
    // box_data[feat * 8400 + prop_idx]

    float scale_x = (float)orig_w / mInputSize;
    float scale_y = (float)orig_h / mInputSize;

    for (int i = 0; i < num_proposals; i++) {
        // Box center coordinates and size (normalized to input size)
        float x_c = box_data[0 * num_proposals + i];
        float y_c = box_data[1 * num_proposals + i];
        float w   = box_data[2 * num_proposals + i];
        float h   = box_data[3 * num_proposals + i];

        // Find best class among dynamic classes only
        float best_conf = 0.0f;
        int   best_cls  = -1;

        for (int cls_id : DYNAMIC_CLASS_IDS) {
            float score = box_data[(4 + cls_id) * num_proposals + i];
            if (score > best_conf) {
                best_conf = score;
                best_cls  = cls_id;
            }
        }

        if (best_conf < mConfThreshold)
            continue;

        // Convert xywh → xyxy, scale to original image size
        int x1 = std::max(0, (int)((x_c - w / 2.0f) * scale_x));
        int y1 = std::max(0, (int)((y_c - h / 2.0f) * scale_y));
        int x2 = std::min(orig_w - 1, (int)((x_c + w / 2.0f) * scale_x));
        int y2 = std::min(orig_h - 1, (int)((y_c + h / 2.0f) * scale_y));

        if (x2 <= x1 || y2 <= y1)
            continue;

        // Extract 32 mask coefficients
        std::vector<float> coeffs(NUM_PROTOS);
        for (int k = 0; k < NUM_PROTOS; k++)
            coeffs[k] = box_data[(4 + NUM_CLASSES + k) * num_proposals + i];

        Detection det;
        det.box         = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        det.confidence  = best_conf;
        det.class_id    = best_cls;
        det.mask_coeffs = coeffs;

        detections.push_back(det);
    }

    return detections;
}

// ─────────────────────────────────────────────────────────────
// Non-Maximum Suppression
// ─────────────────────────────────────────────────────────────

std::vector<YOLODetector::Detection> YOLODetector::ApplyNMS(
    std::vector<Detection>& detections,
    float iou_threshold)
{
    if (detections.empty())
        return {};

    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    // Convert to format OpenCV NMS expects
    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    for (auto& d : detections) {
        boxes.push_back(d.box);
        scores.push_back(d.confidence);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores,
                      mConfThreshold, iou_threshold, indices);

    std::vector<Detection> result;
    for (int idx : indices)
        result.push_back(detections[idx]);

    return result;
}

// ─────────────────────────────────────────────────────────────
// Combine Mask Coefficients with Prototype Masks
// ─────────────────────────────────────────────────────────────

cv::Mat YOLODetector::CombineMaskWithProtos(
    const std::vector<float>& coeffs,
    float* proto_data,
    const cv::Rect& box_crop,
    int orig_w,
    int orig_h)
{
    // Weighted sum of prototype masks
    // result[y][x] = sigmoid( sum_k( coeffs[k] * proto[k][y][x] ) )
    int proto_area = PROTO_SIZE * PROTO_SIZE;
    cv::Mat combined = cv::Mat::zeros(PROTO_SIZE, PROTO_SIZE, CV_32F);

    for (int k = 0; k < NUM_PROTOS; k++) {
        cv::Mat proto_k(PROTO_SIZE, PROTO_SIZE, CV_32F,
                        proto_data + k * proto_area);
        combined += coeffs[k] * proto_k;
    }

    // Sigmoid activation
    cv::Mat sigmoid_mask(PROTO_SIZE, PROTO_SIZE, CV_32F);
    for (int y = 0; y < PROTO_SIZE; y++) {
        for (int x = 0; x < PROTO_SIZE; x++) {
            float val = combined.at<float>(y, x);
            sigmoid_mask.at<float>(y, x) = 1.0f / (1.0f + std::exp(-val));
        }
    }

    // Resize to original frame resolution
    cv::Mat resized;
    cv::resize(sigmoid_mask, resized,
               cv::Size(orig_w, orig_h),
               0, 0, cv::INTER_LINEAR);

    // Crop to bounding box (zero out everything outside box)
    cv::Mat cropped = cv::Mat::zeros(orig_h, orig_w, CV_32F);
    int sx = std::max(0, box_crop.x);
    int sy = std::max(0, box_crop.y);
    int sw = std::min(box_crop.width,  orig_w - sx);
    int sh = std::min(box_crop.height, orig_h - sy);
    if (sw > 0 && sh > 0) {
        cv::Rect safe_box(sx, sy, sw, sh);
        resized(safe_box).copyTo(cropped(safe_box));
    }

    // Threshold to binary mask
    cv::Mat binary;
    cv::threshold(cropped, binary,
                  mMaskThreshold, 1.0f, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);

    return binary;
}

// ─────────────────────────────────────────────────────────────
// Full Postprocessing Pipeline
// ─────────────────────────────────────────────────────────────

cv::Mat YOLODetector::PostprocessSegmentation(
    float* box_data,
    float* proto_data,
    int orig_w,
    int orig_h)
{
    cv::Mat final_mask = cv::Mat::zeros(orig_h, orig_w, CV_8U);

    // Step 1: Decode all proposals
    std::vector<Detection> detections =
        DecodeProposals(box_data, NUM_PROPOSALS, orig_w, orig_h);

    if (detections.empty())
        return final_mask;

    // Step 2: NMS to remove duplicates
    std::vector<Detection> kept = ApplyNMS(detections);

    // Step 3: Generate pixel mask for each detection
    for (auto& det : kept) {
        cv::Mat obj_mask = CombineMaskWithProtos(
            det.mask_coeffs,
            proto_data,
            det.box,
            orig_w, orig_h
        );

        // Merge into final mask using OR
        cv::bitwise_or(final_mask, obj_mask, final_mask);
    }

    // Step 4: Dilate slightly to cover keypoints near object edges
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::dilate(final_mask, final_mask, kernel, cv::Point(-1,-1), 1);

    return final_mask;
}

} // namespace ORB_SLAM3