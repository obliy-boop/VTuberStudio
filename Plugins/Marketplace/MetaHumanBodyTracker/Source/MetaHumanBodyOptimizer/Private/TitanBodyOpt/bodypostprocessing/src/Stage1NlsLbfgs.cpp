// Copyright Epic Games, Inc. All Rights Reserved.

#include "bodypostprocessing/Stage1NlsLbfgs.h"

#include <carbon/common/Defs.h>
#include <nls/Context.h>
#include <nls/DiffData.h>
#include <nls/Jacobian.h>
#include <nls/VectorVariable.h>
#include <nls/solver/LBFGSSolver.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace bodyopt
{

Stage1NlsLbfgsResult SolveStage1WithNlsLbfgs(
    const SmplxForwardLite& smplx,
    const Stage1Config& cfg,
    const Stage1Data& data,
    const std::vector<float>& x0,
    int iterations,
    float jacobianEps,
    std::shared_ptr<TITAN_NAMESPACE::TaskThreadPool> threadPool)
{
    if (x0.empty())
    {
        throw std::runtime_error("SolveStage1WithNlsLbfgs: x0 is empty");
    }
    if (iterations <= 0)
    {
        throw std::runtime_error("SolveStage1WithNlsLbfgs: iterations must be > 0");
    }
    (void)jacobianEps;

    using namespace TITAN_NAMESPACE;

    Stage1NlsLbfgsResult out;
    out.fInitial = EvaluateStage1Objective(smplx, cfg, data, x0, threadPool);
    out.iterations = iterations;

    VectorVariable<float> xVar(static_cast<int>(x0.size()));
    xVar.Set(Eigen::Map<const Vector<float>>(x0.data(), static_cast<int>(x0.size())));

    std::function<DiffData<float>(Context<float>*)> evaluationFunction =
        [&](Context<float>* context) -> DiffData<float>
    {
        DiffData<float> xDiff = xVar.Evaluate(context);
        const Vector<float>& xVec = xDiff.Value();
        std::vector<float> x(static_cast<size_t>(xVec.size()));
        for (int i = 0; i < xVec.size(); ++i)
        {
            x[static_cast<size_t>(i)] = xVec[i];
        }

        const Stage1ObjectiveEval eval = EvaluateStage1ObjectiveAndGradient(smplx, cfg, data, x, threadPool);
        const float f = eval.objective;
        const float clampThreshold = 1e-12f;
        const float safeF = std::max(clampThreshold, f);
        const float r = std::sqrt(safeF);
        Vector<float> residual(1);
        residual[0] = r;

        JacobianConstPtr<float> jacobian;
        if (context)
        {
            Eigen::Matrix<float, -1, -1, Eigen::RowMajor> J(1, static_cast<int>(eval.gradient.size()));

            // When f is clamped (f <= clampThreshold), r is constant, so dr/dx = 0
            // Otherwise, r = sqrt(f), so dr/df = 1/(2*sqrt(f)), and dr/dx = dr/df * df/dx
            if (f <= clampThreshold)
            {
                // Residual is constant in clamped regime, derivative is zero
                J.setZero();
            }
            else
            {
                // Chain rule: dr/dx = (1/(2*sqrt(f))) * df/dx
                const float denom = 2.0f * r;
                for (int c = 0; c < J.cols(); ++c)
                {
                    J(0, c) = eval.gradient[static_cast<size_t>(c)] / denom;
                }
            }

            jacobian = std::make_shared<DenseJacobian<float>>(
                std::make_shared<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(J),
                0);
        }
        return DiffData<float>(std::move(residual), std::move(jacobian));
    };

    Context<float> context;
    LBFGSSolver<float> solver;

    // Configure delta residual convergence criterion from config
    if (cfg.lbfgsAbsDeltaStoppingCriterion > 0.0f) {
        solver.SetAbsDeltaStoppingCriterion(cfg.lbfgsAbsDeltaStoppingCriterion);
        solver.SetAbsDeltaStoppingWindow(1);
    }

    out.success = solver.Solve(evaluationFunction, context, iterations);

    const Vector<float>& xOptVec = xVar.Value();
    out.x.resize(static_cast<size_t>(xOptVec.size()));
    for (int i = 0; i < xOptVec.size(); ++i)
    {
        out.x[static_cast<size_t>(i)] = xOptVec[i];
    }
    out.fFinal = EvaluateStage1Objective(smplx, cfg, data, out.x);
    return out;
}
} // namespace bodyopt

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
