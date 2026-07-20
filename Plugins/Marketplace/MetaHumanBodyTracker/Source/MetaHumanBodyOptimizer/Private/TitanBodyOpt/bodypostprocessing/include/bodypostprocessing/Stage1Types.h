// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/common/Defs.h>

#include <vector>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace bodyopt
{

// Frame-level case data for sequence optimization
struct SequenceFrameCase
{
    int frameIndex = 0;
    int personId = -1;
    std::vector<float> smplxPose165;
    std::vector<float> smplxBetas10;
    std::vector<float> smplxTrans3;
    std::vector<float> cameraRcw9;
    std::vector<float> cameraTcw3;
    std::vector<float> bbox4;
    std::vector<float> keypoints2d25x3;
    std::vector<float> staticConfLogits; // Contact predictions (CHMR output)
    bool valid = false;
};

} // namespace bodyopt

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
