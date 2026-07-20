// Copyright Epic Games, Inc. All Rights Reserved.

#include <nls/solver/LBFGSSolver.h>
#include <nls/Context.h>
#include <nls/Variable.h>

#include <chrono>
#include <cmath>
#include <limits>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

template <class T>
bool LBFGSSolver<T>::Solve(const std::function<DiffData<T>(Context<T>*)>& evaluationFunction,
		                    Context<T>& context,
		                    int iterations) const
{
    // Validate configuration
    if (historySize <= 0) {
        CARBON_CRITICAL("LBFGSSolver: historySize must be positive, got {}", historySize);
    }
    if (iterations <= 0) {
        return true; // No iterations requested
    }

    auto WrapIndex = [](int value, int mod) -> int
    {
        // Defensive check - should never trigger due to validation above
        if (mod <= 0) {
            CARBON_CRITICAL("WrapIndex: invalid mod value {}", mod);
        }
        const int r = value % mod;
        return (r < 0) ? (r + mod) : r;
    };

    // gradient lambda
    auto getEnergyAndGradient = [&]()
    {
        DiffData<T> diffData = evaluationFunction(&context);
        if (!diffData.HasJacobian())
        {
            CARBON_CRITICAL("data is missing the jacobian");
        }
        const Vector<T> fx = diffData.Value();
        const T residualError = diffData.Value().squaredNorm();

        // the gradient can be calculated as 2 Jt r
        Vector<T> g = Vector<T>::Zero(diffData.Jacobian().Cols());
        diffData.Jacobian().AddJtx(g, fx, T(2));

        return std::make_pair(residualError, g);
    };

    // Use member variables for stopping criteria (configured via setter methods)

    // storage for other variables
    Vector<T> x;
    Vector<T> g;
    Vector<T> newG;
    Vector<T> dx;
    Vector<T> prevX;
    Vector<T> prevG;
    int currentHistorySize = -1;
    int currentIndex = -2;
    T residualError = T(0.0);
    T prevResidualError = std::numeric_limits<T>::infinity();
    int smallDeltaCount = 0;
    T alpha = T(1.0);

    for (int iter = 0; iter < iterations; ++iter) {

        // perform LBFGS update
        currentHistorySize = std::min(currentHistorySize + 1, historySize);
        currentIndex = WrapIndex(currentIndex + 1, historySize);

        std::tie(residualError, g) = getEnergyAndGradient();
        //LOG_INFO("Iteration {}, residual error = {}", iter, residualError);
        x = context.Value();

        // check for termination
        if (residualError < residualErrorStoppingCriterion)
        {
            return true;
        }
        if (absDeltaStoppingCriterion > T(0) && std::isfinite(prevResidualError))
        {
            const T absDelta = std::abs(prevResidualError - residualError);
            if (absDelta < absDeltaStoppingCriterion) {
                ++smallDeltaCount;
                if (smallDeltaCount >= absDeltaStoppingWindow) {
                    return true;
                }
            } else {
                smallDeltaCount = 0;
            }
        }
        prevResidualError = residualError;

        if (currentHistorySize == 0)
        {
            // first iteration, just do gradient descent
            dx = -g;

            alpha = std::min(T(0.1), T(0.1) / dx.cwiseAbs().sum());
        }
        else
        {
            // later iteration, update search direction using lbfgs updat
            s[currentIndex] = x - prevX;
            y[currentIndex] = g - prevG;
            const auto ys = y[currentIndex].dot(s[currentIndex]);

            // check if current update is in a numerically stable direction
            if (ys > T(1e-7))
            {
                p[currentIndex] = T(1.0) / ys;

                const auto yk = ys / y[currentIndex].squaredNorm();

                dx = g;
                // Backward loop: iterate from newest to oldest
                for (int i = 0; i < currentHistorySize; ++i)
                {
                    const auto index = WrapIndex(currentIndex - i, historySize);
                    a[index] = s[index].dot(dx) * p[index];
                    dx -= y[index] * a[index];
                }

                dx *= yk;

                // Forward loop: iterate from oldest to newest
                for (int i = 0; i < currentHistorySize; ++i)
                {
                    const auto index = WrapIndex(currentIndex - currentHistorySize + 1 + i, historySize);
                    const auto b = y[index].dot(dx) * p[index];
                    dx += s[index] * (a[index] - b);
                }

                dx = -dx;

                alpha = T(1.0);
            }
            else
            {
                // no, reset and clear history
                currentHistorySize = 0;
                currentIndex = -1;
                dx = -g;
                alpha = std::min(T(0.1), T(0.1) / dx.cwiseAbs().sum());
            }
        }

        prevX = x;
        prevG = g;

        auto gtd = g.dot(dx);

        if (gtd >= T(0.0)) {
            // hessian approximation failed, reset
            currentHistorySize = 0;
            currentIndex = -1;
            dx = -g;
            alpha = std::min(T(0.1), T(0.1) / dx.cwiseAbs().sum());
            gtd = g.dot(dx);
        }

        if (-gtd / residualError < predictionReductionStoppingCriterion) {
            // almost no further change in residual possible, so stop the optimization
            return true;
        }

        context.Update(alpha * dx);

        // get gradient and value at estimated step
        T newResidualError;
        std::tie(newResidualError, newG) = getEnergyAndGradient();
        auto newGtd = newG.dot(dx);

        // calculate cubic interpolation udpate
        //self._cubic_interpolate(0, f, gtd.item(), alpha, f_new, gtd_new.item())
        const auto d1 = gtd + newGtd - T(3.0) * (residualError - newResidualError) / (0.0 - alpha);
        const auto d2_square = std::max(T(0.0), T(d1 * d1 - gtd * newGtd));
        const auto d2 = std::sqrt(d2_square);
        const auto step = alpha - alpha * ((newGtd + d2 - d1) / (newGtd - gtd + T(2.0) * d2));
        const auto alpha_new = std::min(std::max(T(step), T(0.0)), T(100.0));

        if (alpha_new == T(0.0))
        {
            // almost no further change in residual possible, so stop the optimization
            return true;
        }

        context.Update((alpha_new - alpha) * dx);

        //T res = evaluationFunction(nullptr).Value().squaredNorm();
    }

    return true;
}

template class LBFGSSolver<float>;
template class LBFGSSolver<double>;

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
