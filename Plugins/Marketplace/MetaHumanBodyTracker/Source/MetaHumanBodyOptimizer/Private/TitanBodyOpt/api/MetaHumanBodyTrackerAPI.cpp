// Copyright Epic Games, Inc. All Rights Reserved.

#include "MetaHumanBodyTrackerAPI.h"

#include "Common.h"

#include <bodypostprocessing/SmplxForwardLite.h>
#include <bodypostprocessing/Stage1Objective.h>
#include <bodypostprocessing/Stage1NlsLbfgs.h>
#include <bodypostprocessing/OptimizerDataTypes.h>
#include <carbon/utils/TaskThreadPool.h>

#include <chrono>
#include <limits>
#include <memory>


using namespace TITAN_NAMESPACE;

namespace TITAN_API_NAMESPACE
{


struct MetaHumanBodyTrackerAPI::Private
{
    // SMPLX model
    bodyopt::SmplxForwardLite smplx;
    bool modelInitialized = false;

    // Configuration
    OptimizationConfig config;
    bodyopt::ObjectiveConstants objectiveConstants;

    // Frame data
    std::vector<FrameKeypointData> frameData;
    CameraParameters camera;
    int startFrame = 0;

    // Results
    SequenceOptimizationResult lastResults;

    // Thread pool
    std::shared_ptr<TITAN_NAMESPACE::TaskThreadPool> threadPool;

    bool InitializeModel(const SMPLXStaticData& data)
    {
        try
        {
            // Call SmplxForwardLite::LoadFromArrays with column-major arrays from UE
            const bool success = smplx.LoadFromArrays(
                data.vertices,
                data.faces,
                data.jointRegressor,
                data.weights,
                data.blendShapes,
                data.kintreeParents,
                data.extraJointsIdxs,
                data.landmarkFacesIdx,
                data.landmarkBaryCoords);

            if (!success)
            {
                return false;
            }

            // Store GMM prior data (optional - will be used if valid)
            objectiveConstants.gmmNumGaussians = data.gmmNumGaussians;
            objectiveConstants.gmmDim = data.gmmDim;
            objectiveConstants.gmmMeans = data.gmmMeans;
            objectiveConstants.gmmPrecisions = data.gmmPrecisions;
            objectiveConstants.gmmLogNllWeights = data.gmmLogNllWeights;

            modelInitialized = true;
            return true;
        }
        catch (const std::exception&)
        {
            modelInitialized = false;
            throw;
        }
    }

    void InitializeThreadPool()
    {
        if (config.numThreads == 0)
        {
            threadPool = nullptr;
            return;
        }

        int numThreads = config.numThreads;
        if (numThreads < 0)
        {
            numThreads = static_cast<int>(TITAN_NAMESPACE::TaskThreadPool::MaxNumThreads());
        }

        threadPool = std::make_shared<TITAN_NAMESPACE::TaskThreadPool>(numThreads);
    }

    bodyopt::OptimizerDataInfo BuildOptimizerDataInfo() const
    {
        bodyopt::OptimizerDataInfo info;
        info.sequence.sequenceLength = static_cast<int>(frameData.size());
        info.sequence.startFrame = startFrame;
        info.sequence.fps = config.fps;
        info.sequence.numPeople = frameData.empty() ? 0 : 1;  // Single person for now

        // Add Stage1 info
        bodyopt::StageInfo stage1Info;
        stage1Info.name = "stage1";
        stage1Info.numIters = config.maxIterations;
        stage1Info.lossKpPosW = config.keypointPositionWeight;
        stage1Info.lossKpEdgeW = config.keypointEdgeWeight;
        stage1Info.lossSmoothW = config.smoothnessWeight;
        stage1Info.lossSmoothJ3dW = config.smoothnessJoint3DWeight;
        stage1Info.lossPriorW = config.priorWeight;
        stage1Info.lossGmmPriorW = config.gmmWeight;
        stage1Info.lossContactVelW = config.contactWeight;
        stage1Info.lossContactHeightW = config.contactHeightWeight;
        stage1Info.lossBelowFloorW = config.belowFloorWeight;
        // Use optimizationVariables from config, or default to optimizing all variables if empty
        stage1Info.optVars = config.optimizationVariables.empty()
            ? std::vector<std::string>{"transl_delta_seq", "cam_height_offset", "cam_init_r6d", "cam_scale", "lower_body_r6d_seq"}
            : config.optimizationVariables;
        info.stages.push_back(stage1Info);

        return info;
    }

    void BuildObjectiveConstants()
    {
        objectiveConstants.kpConf = 0.7f;
        objectiveConstants.gmofSigma = 100.0f;

        // Camera intrinsics (now always provided since camera is required)
        // Assume intrinsics are [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        if (camera.intrinsics.size() >= 9)
        {
            objectiveConstants.imgFocal = camera.intrinsics[0];  // fx
            objectiveConstants.imgCenterX = camera.intrinsics[2];  // cx
            objectiveConstants.imgCenterY = camera.intrinsics[5];  // cy
        }

        // Joint mapping and skeleton edges
        // Maps from 2D keypoint indices (25 COCO keypoints) to SMPL-X joint indices
        objectiveConstants.jointMapping25 = {
            55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65
        };

        // Position keypoint indices (subset used for position loss)
        objectiveConstants.jointIdxsPos = {
            0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18
        };

        // Skeleton edges (parent-child pairs in 2D keypoint space) NB this is a subset of the full OpenPose Skeleton
        const std::vector<std::pair<int, int>> edges = {
            {14, 13}, {11, 10}, {5, 2}, {5, 6}, {2, 3}, {6, 7}, {3, 4},
            {16, 15}, {0, 16}, {0, 15}, {16, 18}, {15, 17}, {18, 5}, {17, 2}
        };

        objectiveConstants.skeletonEdgesFlat.clear();
        objectiveConstants.skeletonEdgesFlat.reserve(edges.size() * 2);
        for (const auto& edge : edges)
        {
            objectiveConstants.skeletonEdgesFlat.push_back(edge.first);
            objectiveConstants.skeletonEdgesFlat.push_back(edge.second);
        }

        // Contact joint IDs (SMPL-X joint indices for feet: left_ankle=7, left_foot=10, right_ankle=8, right_foot=11)
        objectiveConstants.contactJointIds = {7, 10, 8, 11};

        // Contact height vertex IDs (SMPLX mesh vertices for floor contact)
        objectiveConstants.contactHeightLeftToeVids = {5790, 5843};
        objectiveConstants.contactHeightRightToeVids = {8487, 8471};
        objectiveConstants.contactHeightLeftFootVids = {8921};
        objectiveConstants.contactHeightRightFootVids = {8714};

        // GMM prior will be set and validated in InitializeModel
    }

    bool ConvertFrameDataToSequenceCases(std::vector<bodyopt::SequenceFrameCase>& cases, std::string& errorMsg) const
    {
        cases.clear();
        cases.reserve(frameData.size());

        for (size_t frameIdx = 0; frameIdx < frameData.size(); ++frameIdx)
        {
            const auto& frame = frameData[frameIdx];
            bodyopt::SequenceFrameCase frameCase;
            frameCase.frameIndex = frame.frameIndex;
            frameCase.personId = frame.personId;

            // Convert keypoints to internal format
            // API provides Nx2 positions + N weights + N IDs
            // Internal format expects 25x3 (x, y, conf)
            frameCase.keypoints2d25x3.resize(75, 0.0f);  // 25 keypoints * 3 (x,y,conf)

            const size_t numKps = frame.keypointIds.size();
            for (size_t i = 0; i < numKps; ++i)
            {
                const int kpId = frame.keypointIds[i];
                if (kpId >= 0 && kpId < 25)
                {
                    const size_t baseIdx = static_cast<size_t>(kpId) * 3;
                    frameCase.keypoints2d25x3[baseIdx + 0] = frame.keypointPositions2D[i * 2 + 0];  // x
                    frameCase.keypoints2d25x3[baseIdx + 1] = frame.keypointPositions2D[i * 2 + 1];  // y
                    frameCase.keypoints2d25x3[baseIdx + 2] = frame.keypointWeights[i];              // conf
                }
            }

            // Validate and use SMPLX parameters from tracking data (REQUIRED)
            if (frame.smplxPose165.size() != 165) {
                errorMsg = "Frame " + std::to_string(frameIdx) + ": smplxPose165 must have exactly 165 elements (got " +
                           std::to_string(frame.smplxPose165.size()) + ")";
                return false;
            }
            if (frame.smplxBetas10.size() < 10) {
                errorMsg = "Frame " + std::to_string(frameIdx) + ": smplxBetas10 must have at least 10 elements (got " +
                           std::to_string(frame.smplxBetas10.size()) + ")";
                return false;
            }
            if (frame.smplxTrans3.size() != 3) {
                errorMsg = "Frame " + std::to_string(frameIdx) + ": smplxTrans3 must have exactly 3 elements (got " +
                           std::to_string(frame.smplxTrans3.size()) + ")";
                return false;
            }

            frameCase.smplxPose165 = frame.smplxPose165;
            frameCase.smplxBetas10 = frame.smplxBetas10;

			// TODO: Bring this back in once we get rid of the padding logic
            // Padding beyond 10 betas is required: SmplxForwardLite expects betas to match
            // the full blend shape basis dimension (m_numBetas, typically 400). Extra entries
            // are zero-filled by the caller and don't affect results, but the array must be
            // the correct size to avoid out-of-bounds access in the shape blend loop.

            // if (frameCase.smplxBetas10.size() > 10) {
            //     frameCase.smplxBetas10.resize(10);
            // }

            frameCase.smplxTrans3 = frame.smplxTrans3;

            // Validate and use camera parameters from tracking data (REQUIRED)
            if (frame.cameraRcw9.size() != 9) {
                errorMsg = "Frame " + std::to_string(frameIdx) + ": cameraRcw9 must have exactly 9 elements (got " +
                           std::to_string(frame.cameraRcw9.size()) + ")";
                return false;
            }
            if (frame.cameraTcw3.size() != 3) {
                errorMsg = "Frame " + std::to_string(frameIdx) + ": cameraTcw3 must have exactly 3 elements (got " +
                           std::to_string(frame.cameraTcw3.size()) + ")";
                return false;
            }

            frameCase.cameraRcw9 = frame.cameraRcw9;
            frameCase.cameraTcw3 = frame.cameraTcw3;

            // Load static confidence logits (contact predictions)
            frameCase.staticConfLogits = frame.staticConfLogits;

            // Use bbox from input if provided, otherwise calculate from keypoints
            if (frame.bbox4.size() == 4) {
                // Use provided bbox
                frameCase.bbox4 = frame.bbox4;
            } else {
                // Calculate bbox from keypoints with confidence > 0
                float minX = std::numeric_limits<float>::max();
                float maxX = -std::numeric_limits<float>::max();
                float minY = std::numeric_limits<float>::max();
                float maxY = -std::numeric_limits<float>::max();
                bool hasValidKp = false;

                for (size_t i = 0; i < 25; ++i) {
                    const size_t baseIdx = i * 3;
                    const float x = frameCase.keypoints2d25x3[baseIdx + 0];
                    const float y = frameCase.keypoints2d25x3[baseIdx + 1];
                    const float conf = frameCase.keypoints2d25x3[baseIdx + 2];

                    if (conf > 0.0f) {
                        minX = std::min(minX, x);
                        maxX = std::max(maxX, x);
                        minY = std::min(minY, y);
                        maxY = std::max(maxY, y);
                        hasValidKp = true;
                    }
                }

                // If no valid keypoints, use a default bbox (should not happen in practice)
                if (!hasValidKp) {
                    frameCase.bbox4 = {0.0f, 0.0f, 1920.0f, 1080.0f};  // Default to image size
                } else {
                    frameCase.bbox4 = {minX, minY, maxX, maxY};
                }
            }

            frameCase.valid = frame.valid;
            cases.push_back(std::move(frameCase));
        }

        return true;
    }

    bool OptimizeSequence()
    {
        if (!modelInitialized)
        {
            return false;
        }

        if (frameData.empty())
        {
            return false;
        }

        try
        {
            // Initialize thread pool
            InitializeThreadPool();

            // Build configuration
            BuildObjectiveConstants();
            const bodyopt::OptimizerDataInfo info = BuildOptimizerDataInfo();
            bodyopt::Stage1Config stage1Cfg = bodyopt::BuildDefaultStage1Config(info, objectiveConstants);

            // Pass through LBFGS convergence criterion from API config
            stage1Cfg.lbfgsAbsDeltaStoppingCriterion = config.lbfgsAbsDeltaStoppingCriterion;

            // Build Stage1Data
            bodyopt::Stage1Data stage1Data;
            std::string errorMsg;
            if (!ConvertFrameDataToSequenceCases(stage1Data.seqCases, errorMsg)) {
                LOG_ERROR("Failed to convert frame data: {}", errorMsg);
                return false;
            }

            // Build initial parameters
            const bodyopt::Stage1Params p0 = bodyopt::BuildStage1InitialParams(stage1Cfg, stage1Data);
            const std::vector<float> x0 = bodyopt::PackStage1Params(stage1Cfg, p0);

            // Validate betas consistency and set evaluation constants
            if (stage1Data.seqCases.empty()) {
                LOG_ERROR("No sequence data available");
                return false;
            }

            // Check if betas are consistent across all frames
            const std::vector<float>& firstBetas = stage1Data.seqCases[0].smplxBetas10;
            bool betasConsistent = true;
            constexpr float betasTolerance = 1e-3f;

            for (const auto& frameCase : stage1Data.seqCases)
            {
                if (frameCase.smplxBetas10.size() != firstBetas.size())
                {
                    LOG_ERROR("Betas size mismatch across frames");
                    return false;
                }
                for (size_t i = 0; i < firstBetas.size(); ++i)
                {
                    const float beta = frameCase.smplxBetas10[i];
                    if (std::abs(beta - firstBetas[i]) > betasTolerance)
                    {
                        betasConsistent = false;
                    }
                }
            }

            // Compute betas to use (average if inconsistent, otherwise use first frame)
            std::vector<float> betasToUse;
            if (!betasConsistent)
            {
                LOG_WARNING("Betas vary across sequence (At least one beta is > {} from its first value) - using average betas", betasTolerance);
                betasToUse.assign(firstBetas.size(), 0.0f);
                for (const auto& frameCase : stage1Data.seqCases)
                {
                    for (size_t i = 0; i < betasToUse.size(); ++i)
                    {
                        betasToUse[i] += frameCase.smplxBetas10[i];
                    }
                }
                for (float& b : betasToUse)
                {
                    b /= static_cast<float>(stage1Data.seqCases.size());
                }
            }
            else
            {
                betasToUse = firstBetas;
            }

            // Determine selected joints and vertices for optimization
            // Build selectedJointIds from jointMapping25 and contactJointIds (matches Stage1Objective.cpp logic)
            std::vector<int> selectedJointIds;
            std::unordered_map<int, int> selectedJointSlot;
            selectedJointIds.reserve(stage1Cfg.jointMapping25.size() + stage1Cfg.contactJointIds.size());

            for (int jidx : stage1Cfg.jointMapping25) {
                if (jidx < 0) {
                    continue;
                }
                if (selectedJointSlot.find(jidx) == selectedJointSlot.end()) {
                    selectedJointSlot[jidx] = static_cast<int>(selectedJointIds.size());
                    selectedJointIds.push_back(jidx);
                }
            }
            for (int jidx : stage1Cfg.contactJointIds) {
                if (jidx < 0) {
                    continue;
                }
                if (selectedJointSlot.find(jidx) == selectedJointSlot.end()) {
                    selectedJointSlot[jidx] = static_cast<int>(selectedJointIds.size());
                    selectedJointIds.push_back(jidx);
                }
            }

            // Build selectedVertexIds from contact height vertex IDs (matches Stage1Objective.cpp logic)
            std::vector<int> selectedVertexIds;
            selectedVertexIds.reserve(
                stage1Cfg.contactHeightLeftToeVids.size() +
                stage1Cfg.contactHeightRightToeVids.size() +
                stage1Cfg.contactHeightLeftFootVids.size() +
                stage1Cfg.contactHeightRightFootVids.size());

            auto addUniqueVid = [&](int vid) {
                if (vid < 0) {
                    return;
                }
                if (std::find(selectedVertexIds.begin(), selectedVertexIds.end(), vid) == selectedVertexIds.end()) {
                    selectedVertexIds.push_back(vid);
                }
            };
            for (int vid : stage1Cfg.contactHeightLeftToeVids) { addUniqueVid(vid); }
            for (int vid : stage1Cfg.contactHeightRightToeVids) { addUniqueVid(vid); }
            for (int vid : stage1Cfg.contactHeightLeftFootVids) { addUniqueVid(vid); }
            for (int vid : stage1Cfg.contactHeightRightFootVids) { addUniqueVid(vid); }

            // Set evaluation constants (this pre-computes shaped vertices, dependencies, etc.)
            smplx.SetSmplxEvaluationConstants(
                betasToUse,
                stage1Cfg.lowerBodyJointIds,
                selectedJointIds,
                selectedVertexIds
            );

            // Run optimization
            const auto startTime = std::chrono::high_resolution_clock::now();

            const bodyopt::Stage1NlsLbfgsResult result = bodyopt::SolveStage1WithNlsLbfgs(
                smplx,
                stage1Cfg,
                stage1Data,
                x0,
                config.maxIterations,
                /*jacobianEps=*/1e-4f,
                threadPool);

            const auto endTime = std::chrono::high_resolution_clock::now();
            const auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            // Unpack results
            const bodyopt::Stage1Params resultParams = bodyopt::UnpackStage1Params(stage1Cfg, result.x);

            // Convert to API result format
            lastResults.success = result.success;
            lastResults.totalIterations = result.iterations;
            lastResults.totalOptimizationTimeSeconds = static_cast<float>(durationMs.count()) / 1000.0f;

            // Global parameters
            lastResults.cameraHeightOffset = resultParams.camHeightOffset;
            lastResults.cameraInitR6D = resultParams.camInitR6d;
            lastResults.cameraScale = resultParams.camScale;

            // Per-frame results
            lastResults.frameResults.clear();
            lastResults.frameResults.reserve(stage1Data.seqCases.size());

            const int seqLen = stage1Cfg.seqLen;
            const int numLowerBodyJoints = static_cast<int>(stage1Cfg.lowerBodyJointIds.size());

            for (int i = 0; i < seqLen; ++i)
            {
                FrameOptimizationResult frameResult;
                frameResult.frameIndex = stage1Data.seqCases[i].frameIndex;
                frameResult.personId = stage1Data.seqCases[i].personId;

                // Translation delta for this frame
                frameResult.translationDelta.resize(3);
                for (int j = 0; j < 3; ++j)
                {
                    frameResult.translationDelta[j] = resultParams.translDeltaSeq[i * 3 + j];
                }

                // Lower body R6D for this frame
                const int r6dSize = numLowerBodyJoints * 6;
                frameResult.lowerBodyR6D.resize(r6dSize);
                for (int j = 0; j < r6dSize; ++j)
                {
                    frameResult.lowerBodyR6D[j] = resultParams.lowerBodyR6dSeq[i * r6dSize + j];
                }

                frameResult.initialObjective = result.fInitial;
                frameResult.finalObjective = result.fFinal;
                frameResult.iterations = result.iterations;
                frameResult.success = result.success;

                lastResults.frameResults.push_back(std::move(frameResult));
            }

            return result.success;
        }
        catch (const std::exception&)
        {
            lastResults.success = false;
            throw;
        }
    }
};

// Constructor / Destructor
MetaHumanBodyTrackerAPI::MetaHumanBodyTrackerAPI()
    : m(new Private{})
{
}

MetaHumanBodyTrackerAPI::~MetaHumanBodyTrackerAPI()
{
    delete m;
}

// API Methods
bool MetaHumanBodyTrackerAPI::InitializeModel(const SMPLXStaticData& data)
{
    try
    {
        TITAN_RESET_ERROR;
        TITAN_CHECK_OR_RETURN(!data.vertices.empty(), false, "SMPLX vertices array is empty");
        TITAN_CHECK_OR_RETURN(!data.faces.empty(), false, "SMPLX faces array is empty");
        TITAN_CHECK_OR_RETURN(!data.jointRegressor.empty(), false, "SMPLX joint regressor array is empty");
        TITAN_CHECK_OR_RETURN(!data.weights.empty(), false, "SMPLX weights array is empty");
        TITAN_CHECK_OR_RETURN(!data.blendShapes.empty(), false, "SMPLX blend shapes array is empty");

        const bool success = m->InitializeModel(data);
        TITAN_CHECK_OR_RETURN(success, false, "Failed to initialize SMPLX model");
        return true;
    }
    catch (const std::exception& e)
    {
        TITAN_HANDLE_EXCEPTION("Failed to initialize model: {}", e.what());
    }
}

bool MetaHumanBodyTrackerAPI::SetOptimizationConfig(const OptimizationConfig& config)
{
    try
    {
        TITAN_RESET_ERROR;
        TITAN_CHECK_OR_RETURN(config.maxIterations > 0, false, "Max iterations must be > 0");
        m->config = config;
        return true;
    }
    catch (const std::exception& e)
    {
        TITAN_HANDLE_EXCEPTION("Failed to set optimization config: {}", e.what());
    }
}

bool MetaHumanBodyTrackerAPI::SetFrameData(
    const std::vector<FrameKeypointData>& frameData,
    const CameraParameters& camera,
    int startFrame)
{
    try
    {
        TITAN_RESET_ERROR;
        TITAN_CHECK_OR_RETURN(!frameData.empty(), false, "Frame data vector is empty");
        TITAN_CHECK_OR_RETURN(camera.intrinsics.size() == 9, false, "Camera intrinsics must be 3x3 (9 elements)");
        TITAN_CHECK_OR_RETURN(startFrame >= 0, false, "Start frame must be >= 0");

        // Validate all frame data
        for (size_t i = 0; i < frameData.size(); ++i)
        {
            const auto& frame = frameData[i];
            TITAN_CHECK_OR_RETURN(!frame.keypointPositions2D.empty(), false,
                                  "Frame {} keypoint positions are empty", i);
            TITAN_CHECK_OR_RETURN(frame.keypointPositions2D.size() == frame.keypointWeights.size() * 2,
                                  false, "Frame {} keypoint positions and weights size mismatch", i);
            TITAN_CHECK_OR_RETURN(frame.keypointWeights.size() == frame.keypointIds.size(),
                                  false, "Frame {} keypoint weights and IDs size mismatch", i);
        }

        m->frameData = frameData;
        m->camera = camera;
        m->startFrame = startFrame;

        // Clear previous results when setting new frame data
        m->lastResults = SequenceOptimizationResult{};

        return true;
    }
    catch (const std::exception& e)
    {
        TITAN_HANDLE_EXCEPTION("Failed to set frame data: {}", e.what());
    }
}

bool MetaHumanBodyTrackerAPI::OptimizeSequence()
{
    try
    {
        TITAN_RESET_ERROR;
        TITAN_CHECK_OR_RETURN(m->modelInitialized, false, "Model not initialized. Call InitializeModel() first.");
        TITAN_CHECK_OR_RETURN(!m->frameData.empty(), false, "No frame data set. Call SetFrameData() first.");

        const bool success = m->OptimizeSequence();
        TITAN_CHECK_OR_RETURN(success, false, "Optimization failed");
        return true;
    }
    catch (const std::exception& e)
    {
        TITAN_HANDLE_EXCEPTION("Failed to optimize sequence: {}", e.what());
    }
}

bool MetaHumanBodyTrackerAPI::GetResults(SequenceOptimizationResult& outResults)
{
    try
    {
        TITAN_RESET_ERROR;
        TITAN_CHECK_OR_RETURN(m->lastResults.success, false, "No successful optimization results available");

        outResults = m->lastResults;
        return true;
    }
    catch (const std::exception& e)
    {
        TITAN_HANDLE_EXCEPTION("Failed to get results: {}", e.what());
    }
}

bool MetaHumanBodyTrackerAPI::IsInitialized() const
{
    return m->modelInitialized;
}

}  // namespace TITAN_API_NAMESPACE
