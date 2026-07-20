// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/common/Defs.h>
#include <carbon/utils/TaskThreadPool.h>

#include <vector>
#include <memory>

#include "bodypostprocessing/Stage1Types.h"
#include "bodypostprocessing/SmplxForwardLite.h"
#include "bodypostprocessing/OptimizerDataTypes.h"

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace bodyopt
{


struct Stage1Weights
{
    float lossKpPosW = 0.5f;
    float lossKpEdgeW = 5.0f;
    float lossSmoothW = 0.1f;
    float lossSmoothJ3dW = 0.01f;
    float lossPriorW = 2.0f;
    float lossGmmPriorW = 0.0005f;
    float lossContactVelW = 11.11f;
    float lossContactHeightW = 10.0f;
    float lossBelowFloorW = 100.0f;
};

struct Stage1Config
{
    int seqLen = 0;
    std::vector<int> lowerBodyJointIds; // joint ids, not flattened component indices
    std::vector<int> jointMapping25;
    std::vector<int> jointIdxsPos;
    std::vector<int> skeletonEdgesFlat;
    std::vector<int> contactJointIds;
    std::vector<int> contactHeightLeftToeVids;
    std::vector<int> contactHeightRightToeVids;
    std::vector<int> contactHeightLeftFootVids;
    std::vector<int> contactHeightRightFootVids;
    int gmmNumGaussians = 0;
    int gmmDim = 0;
    std::vector<float> gmmMeans; // [M*D]
    std::vector<float> gmmPrecisions; // [M*D*D]
    std::vector<float> gmmLogNllWeights; // [M]
    float kpConf = 0.7f;
    float gmofSigma = 100.0f;
    float imgFocal = 0.0f;
    float imgCenterX = 0.0f;
    float imgCenterY = 0.0f;
    float fps = 30.0f;
    // LBFGS convergence criterion: stop when |delta_residual| < threshold
    // If <= 0, this criterion is disabled (default: 1e-5)
    float lbfgsAbsDeltaStoppingCriterion = 1e-5f;
    Stage1Weights weights;
    // Variables to optimize (if empty, optimize all)
    std::vector<std::string> optVars;
};

struct Stage1Data
{
    std::vector<SequenceFrameCase> seqCases; // length seqLen
};

struct Stage1Params
{
    std::vector<float> translDeltaSeq; // [S,3]
    float camHeightOffset = 0.0f; // scalar
    std::vector<float> camInitR6d; // [6]
    float camScale = 1.0f; // scalar
    std::vector<float> lowerBodyR6dSeq; // [S, numLower*6]
};

struct Stage1ObjectiveEval
{
    float objective = 0.0f;
    std::vector<float> gradient; // same layout as packed Stage-1 params
};

struct Stage1ObjectiveTimingBreakdown
{
    int calls = 0;
    double totalMs = 0.0;
    double unpackMs = 0.0;
    double setupMs = 0.0;
    double forwardJacMs = 0.0;
    double projLossGradMs = 0.0;
    double camR6dMs = 0.0;
    double smoothMs = 0.0;
    double finalizeMs = 0.0;
};

Stage1Config BuildDefaultStage1Config(
    const OptimizerDataInfo& info,
    const ObjectiveConstants& constants);

size_t Stage1ParameterCount(const Stage1Config& cfg);

Stage1Params UnpackStage1Params(const Stage1Config& cfg, const std::vector<float>& x);
std::vector<float> PackStage1Params(const Stage1Config& cfg, const Stage1Params& p);
Stage1Params BuildStage1InitialParams(const Stage1Config& cfg, const Stage1Data& data);

float EvaluateStage1Objective(
    const SmplxForwardLite& smplx,
    const Stage1Config& cfg,
    const Stage1Data& data,
    const std::vector<float>& x,
    std::shared_ptr<TITAN_NAMESPACE::TaskThreadPool> threadPool = nullptr);

Stage1ObjectiveEval EvaluateStage1ObjectiveAndGradient(
    const SmplxForwardLite& smplx,
    const Stage1Config& cfg,
    const Stage1Data& data,
    const std::vector<float>& x,
    std::shared_ptr<TaskThreadPool> threadPool = nullptr);
} // namespace bodyopt


CARBON_NAMESPACE_END(TITAN_NAMESPACE)
