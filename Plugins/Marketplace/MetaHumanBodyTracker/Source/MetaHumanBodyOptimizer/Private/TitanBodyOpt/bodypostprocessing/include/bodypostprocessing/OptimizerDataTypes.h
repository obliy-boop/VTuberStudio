// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/common/Defs.h>

#include <string>
#include <vector>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace bodyopt
{

struct StageInfo
{
    std::string name;
    int numIters = 0;
    float lr = 0.0f;
    float lossKpPosW = 0.0f;
    float lossKpEdgeW = 0.0f;
    float lossSmoothW = 0.0f;
    float lossSmoothGoW = 0.0f;
    float lossSmoothJ3dW = 0.0f;
    float lossPriorW = 0.0f;
    float lossGmmPriorW = 0.0f;
    float lossContactVelW = 0.0f;
    float lossContactHeightW = 0.0f;
    float lossBelowFloorW = 0.0f;
    std::vector<std::string> optVars;
};

struct SequenceInfo
{
    int sequenceLength = 0;
    int startFrame = 0;
    float fps = 0.0f;
    int numPeople = 0;
};

struct OptimizerDataInfo
{
    std::string folder;
    std::string metaPath;
    std::string framePattern;
    int numFrameFilesOnDisk = 0;
    SequenceInfo sequence;
    std::vector<StageInfo> stages;
    bool hasDebugSequenceLength = false;
    int debugSequenceLength = 0;
};

struct ObjectiveConstants
{
    float kpConf = 0.7f;
    float gmofSigma = 100.0f;
    float imgFocal = 0.0f;
    float imgCenterX = 0.0f;
    float imgCenterY = 0.0f;
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
};
} // namespace bodyopt

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
