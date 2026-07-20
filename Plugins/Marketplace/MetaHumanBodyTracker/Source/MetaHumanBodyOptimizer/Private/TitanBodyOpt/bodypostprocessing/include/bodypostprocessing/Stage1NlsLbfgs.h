// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/common/Defs.h>

#include <vector>
#include <memory>

#include "bodypostprocessing/SmplxForwardLite.h"
#include "bodypostprocessing/Stage1Objective.h"

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace bodyopt
{

struct Stage1NlsLbfgsResult
{
    std::vector<float> x;
    float fInitial = 0.0f;
    float fFinal = 0.0f;
    int iterations = 0;
    bool success = false;
};

// Solves Stage-1 objective with Titan nls Context + extNls LBFGS.
Stage1NlsLbfgsResult SolveStage1WithNlsLbfgs(
    const SmplxForwardLite& smplx,
    const Stage1Config& cfg,
    const Stage1Data& data,
    const std::vector<float>& x0,
    int iterations,
    float jacobianEps = 1e-4f,
    std::shared_ptr<TaskThreadPool> threadPool = nullptr);
} // namespace bodyopt

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
