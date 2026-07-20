// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Defs.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace TITAN_API_NAMESPACE
{

/**
 * SMPLX static model data structure for array-based initialization.
 */
struct SMPLXStaticData
{
    std::vector<float> vertices;           // NumVerts * 3, column-major (NumVerts, 3)
    std::vector<uint32_t> faces;           // NumFaces * 3
    std::vector<float> jointRegressor;     // NumJoints * NumVerts, column-major (NumJoints, NumVerts)
    std::vector<float> weights;            // NumVerts * NumJoints, column-major (NumVerts, NumJoints)
    std::vector<float> blendShapes;        // NumVerts*3 * (NumShapes + 9*(NumJoints-1)), column-major
    std::vector<int> kintreeParents;       // NumJoints parent indices
    std::vector<int> extraJointsIdxs;      // 21 extra joint indices
    std::vector<int> landmarkFacesIdx;     // 51 landmark face indices
    std::vector<float> landmarkBaryCoords; // 51*3 barycentric coords

    // GMM Prior data (optional - used for pose regularization if provided)
    int gmmNumGaussians = 0;               // Number of Gaussian components
    int gmmDim = 0;                        // Dimensionality of GMM
    std::vector<float> gmmMeans;           // [M*D] GMM means
    std::vector<float> gmmPrecisions;      // [M*D*D] GMM precisions
    std::vector<float> gmmLogNllWeights;   // [M] Log of NLL weights (for numerical stability)
};

/**
 * Per-frame keypoint data.
 */
struct FrameKeypointData
{
    int frameIndex = -1;
    int personId = -1;
    std::vector<float> keypointPositions2D; // Nx2 keypoint positions
    std::vector<float> keypointWeights;     // N weights
    std::vector<int> keypointIds;           // N keypoint IDs

    // Bounding box [minX, minY, maxX, maxY] (optional - will be calculated from keypoints if empty)
    std::vector<float> bbox4;          // 4 values: [minX, minY, maxX, maxY]

    // Initial SMPLX parameters (required for optimization)
    std::vector<float> smplxPose165;   // 165 pose parameters (55 joints * 3 axis-angle)
    std::vector<float> smplxBetas10;   // 10 shape parameters
    std::vector<float> smplxTrans3;    // 3D translation

    // Camera parameters for this frame
    std::vector<float> cameraRcw9;     // 3x3 rotation matrix (row-major)
    std::vector<float> cameraTcw3;     // 3D translation vector

    // Contact predictions (CHMR static confidence logits for floor contact)
    std::vector<float> staticConfLogits;  // Contact confidence logits (typically 4-6 values)

    // Frame validity flag (true = valid frame to include in optimization, false = skip)
    bool valid = true;
};

/**
 * Camera parameters for a frame.
 */
struct CameraParameters
{
    // only intrinsics are needed; extrinsics are per frame
    std::vector<float> intrinsics;  // 3x3 camera intrinsics (row-major)
};

/**
 * Optimization configuration/weights.
 */
struct OptimizationConfig
{
    // Loss weights
    float keypointPositionWeight = 0.5f;
    float keypointEdgeWeight = 5.0f;
    float smoothnessWeight = 0.1f;
    float smoothnessJoint3DWeight = 0.01f;
    float priorWeight = 2.0f;
    float gmmWeight = 0.0005f;
    float contactWeight = 11.11f;
    float contactHeightWeight = 10.0f;
    float belowFloorWeight = 100.0f;

    // Sequence parameters
    float fps = 30.0f;  // Frames per second for smoothness loss

    // Solver parameters
    int maxIterations = 1500;
    bool useSparseJacobian = true;
    int numThreads = -1;  // -1 for auto-detect

    // LBFGS convergence criterion: stop when |delta_residual| < threshold
    // Set to <= 0 to disable (default: 1e-4)
    float lbfgsAbsDeltaStoppingCriterion = 1e-4f;

    // Variables to optimize (if empty, all variables are optimized)
    // Supported values: "transl", "cam_height_offset", "cam_init_r6d", "cam_scale", "smplx_pose_r6d_lower_body"
    std::vector<std::string> optimizationVariables;
};

/**
 * Optimization result for a single frame.
 */
struct FrameOptimizationResult
{
    int frameIndex = -1;
    int personId = -1;

    // Optimized parameters
    std::vector<float> translationDelta;  // 3D translation delta
    std::vector<float> lowerBodyR6D;      // Lower body 6D rotations

    // Objective values
    float initialObjective = 0;
    float finalObjective = 0;
    int iterations = 0;
    bool success = false;
};

/**
 * Full sequence optimization results.
 */
struct SequenceOptimizationResult
{
    std::vector<FrameOptimizationResult> frameResults;

    // Global parameters (shared across sequence)
    float cameraHeightOffset = 0;
    std::vector<float> cameraInitR6D;  // 6D rotation
    float cameraScale = 0;

    // Statistics
    float totalOptimizationTimeSeconds = 0;
    int totalIterations = 0;
    bool success = false;
};

/**
 * MetaHuman Body Tracker API
 *
 * Provides a clean interface for body tracking optimization that can be
 * called from Unreal Engine. Uses PIMPL pattern to hide implementation
 * details and ensure binary compatibility.
 *
 * Typical workflow:
 * 1. Create API instance
 * 2. InitializeModel() with SMPLX static data
 * 3. SetOptimizationConfig() to configure solver (optional)
 * 4. SetFrameData() with all frame keypoints, camera parameters, and sequence info
 * 5. OptimizeSequence() to run optimization
 * 6. GetResults() to retrieve optimized parameters
 */
class TITAN_API MetaHumanBodyTrackerAPI final
{
public:
    MetaHumanBodyTrackerAPI();
    ~MetaHumanBodyTrackerAPI();

    // Disable copy and move
    MetaHumanBodyTrackerAPI(MetaHumanBodyTrackerAPI&&) = delete;
    MetaHumanBodyTrackerAPI(const MetaHumanBodyTrackerAPI&) = delete;
    MetaHumanBodyTrackerAPI& operator=(MetaHumanBodyTrackerAPI&&) = delete;
    MetaHumanBodyTrackerAPI& operator=(const MetaHumanBodyTrackerAPI&) = delete;

    /**
     * Initialize the SMPLX model from raw arrays.
     * Must be called before any other operations.
     *
     * IMPORTANT: Arrays must be in COLUMN-MAJOR format (Unreal Engine / Eigen default).
     * - vertices: (NumVerts, 3) column-major: [x0,x1,...,xN, y0,y1,...,yN, z0,z1,...,zN]
     * - weights: (NumVerts, NumJoints) column-major
     * - jointRegressor: (NumJoints, NumVerts) column-major
     * - blendShapes: (NumVerts*3, NumShapes) column-major
     *
     * @param data SMPLX static model data (vertices, faces, blend shapes, etc.)
     * @return true on success, false on failure
     */
    bool InitializeModel(const SMPLXStaticData& data);

    /**
     * Set optimization configuration and weights.
     * Optional - defaults will be used if not called.
     *
     * @param config Optimization configuration
     * @return true on success, false on failure
     */
    bool SetOptimizationConfig(const OptimizationConfig& config);

    /**
     * Set frame data for the entire sequence.
     * Includes all frame keypoints, camera parameters, and sequence configuration.
     * Must be called before optimizing.
     *
     * @param frameData Vector of keypoint data for all frames in the sequence
     * @param camera Camera parameters (intrinsics only) - required
     * @param startFrame Starting frame index (default: 0)
     * @return true on success, false on failure
     */
    bool SetFrameData(
        const std::vector<FrameKeypointData>& frameData,
        const CameraParameters& camera,
        int startFrame = 0);

    /**
     * Run optimization on the loaded sequence.
     * Must call InitializeModel() and SetFrameData() before this.
     *
     * @return true on success, false on failure
     */
    bool OptimizeSequence();

    /**
     * Get optimization results after OptimizeSequence() completes.
     *
     * @param outResults Output results structure
     * @return true on success, false on failure
     */
    bool GetResults(SequenceOptimizationResult& outResults);

    /**
     * Check if model is initialized and ready for optimization.
     *
     * @return true if initialized, false otherwise
     */
    bool IsInitialized() const;

private:
    struct Private;
    Private* m{};
};

}  // namespace TITAN_API_NAMESPACE
