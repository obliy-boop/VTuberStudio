// Copyright Epic Games, Inc. All Rights Reserved.

#include "bodypostprocessing/Stage1Objective.h"

#include <carbon/common/Defs.h>
#include <carbon/utils/TaskThreadPool.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace bodyopt
{


namespace {

inline float Dot3(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline std::array<float, 3> Cross3(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

inline std::array<float, 3> Normalize3(const std::array<float, 3>& v) {
    const float n = std::sqrt(std::max(1e-20f, Dot3(v, v)));
    return {v[0] / n, v[1] / n, v[2] / n};
}

// Squared-velocity scaling factor for contact velocity loss.
// Matches the Python formulation: ((j[t] - j[t-1]) * fps)² = displacement² * fps².
inline float ContactVelocityFpsScale(float fps) {
    return fps * fps;
}

inline void AxisAngleToRotation3x3(float x, float y, float z, float r[9]) {
    const float theta = std::sqrt(x * x + y * y + z * z);
    if (theta <= 1e-8f) {
        r[0] = 1.0f; r[1] = 0.0f; r[2] = 0.0f;
        r[3] = 0.0f; r[4] = 1.0f; r[5] = 0.0f;
        r[6] = 0.0f; r[7] = 0.0f; r[8] = 1.0f;
        return;
    }
    const float inv = 1.0f / theta;
    const float ax = x * inv;
    const float ay = y * inv;
    const float az = z * inv;
    const float c = std::cos(theta);
    const float s = std::sin(theta);
    const float t = 1.0f - c;

    r[0] = c + ax * ax * t;
    r[1] = ax * ay * t - az * s;
    r[2] = ax * az * t + ay * s;
    r[3] = ay * ax * t + az * s;
    r[4] = c + ay * ay * t;
    r[5] = ay * az * t - ax * s;
    r[6] = az * ax * t - ay * s;
    r[7] = az * ay * t + ax * s;
    r[8] = c + az * az * t;
}

inline void Rotation6dToMatrix(const float r6d[6], float R[9]) {
    const std::array<float, 3> a1{r6d[0], r6d[1], r6d[2]};
    const std::array<float, 3> a2{r6d[3], r6d[4], r6d[5]};
    const std::array<float, 3> b1 = Normalize3(a1);
    const float proj = Dot3(b1, a2);
    const std::array<float, 3> a2o{
        a2[0] - proj * b1[0],
        a2[1] - proj * b1[1],
        a2[2] - proj * b1[2],
    };
    const std::array<float, 3> b2 = Normalize3(a2o);
    const std::array<float, 3> b3 = Cross3(b1, b2);
    // Columns are b1,b2,b3 (pytorch3d convention)
    R[0] = b1[0]; R[1] = b2[0]; R[2] = b3[0];
    R[3] = b1[1]; R[4] = b2[1]; R[5] = b3[1];
    R[6] = b1[2]; R[7] = b2[2]; R[8] = b3[2];
}

inline void Rotation6dToMatrixAndJacobians(const float r6d[6], float R[9], float dR_dr6d[6][9]) {
    const std::array<float, 3> a1{r6d[0], r6d[1], r6d[2]};
    const std::array<float, 3> a2{r6d[3], r6d[4], r6d[5]};

    const float n1 = std::sqrt(std::max(1e-20f, Dot3(a1, a1)));
    const std::array<float, 3> b1{a1[0] / n1, a1[1] / n1, a1[2] / n1};
    const float proj = Dot3(b1, a2);
    const std::array<float, 3> u2{
        a2[0] - proj * b1[0],
        a2[1] - proj * b1[1],
        a2[2] - proj * b1[2],
    };
    const float n2 = std::sqrt(std::max(1e-20f, Dot3(u2, u2)));
    const std::array<float, 3> b2{u2[0] / n2, u2[1] / n2, u2[2] / n2};
    const std::array<float, 3> b3 = Cross3(b1, b2);

    R[0] = b1[0]; R[1] = b2[0]; R[2] = b3[0];
    R[3] = b1[1]; R[4] = b2[1]; R[5] = b3[1];
    R[6] = b1[2]; R[7] = b2[2]; R[8] = b3[2];

    for (int i = 0; i < 6; ++i) {
        const std::array<float, 3> da1{
            (i == 0) ? 1.0f : 0.0f,
            (i == 1) ? 1.0f : 0.0f,
            (i == 2) ? 1.0f : 0.0f,
        };
        const std::array<float, 3> da2{
            (i == 3) ? 1.0f : 0.0f,
            (i == 4) ? 1.0f : 0.0f,
            (i == 5) ? 1.0f : 0.0f,
        };

        const float b1DotDa1 = Dot3(b1, da1);
        const std::array<float, 3> db1{
            (da1[0] - b1[0] * b1DotDa1) / n1,
            (da1[1] - b1[1] * b1DotDa1) / n1,
            (da1[2] - b1[2] * b1DotDa1) / n1,
        };

        const float dproj = Dot3(db1, a2) + Dot3(b1, da2);
        const std::array<float, 3> du2{
            da2[0] - dproj * b1[0] - proj * db1[0],
            da2[1] - dproj * b1[1] - proj * db1[1],
            da2[2] - dproj * b1[2] - proj * db1[2],
        };

        const float b2DotDu2 = Dot3(b2, du2);
        const std::array<float, 3> db2{
            (du2[0] - b2[0] * b2DotDu2) / n2,
            (du2[1] - b2[1] * b2DotDu2) / n2,
            (du2[2] - b2[2] * b2DotDu2) / n2,
        };

        const std::array<float, 3> db3{
            db1[1] * b2[2] + b1[1] * db2[2] - db1[2] * b2[1] - b1[2] * db2[1],
            db1[2] * b2[0] + b1[2] * db2[0] - db1[0] * b2[2] - b1[0] * db2[2],
            db1[0] * b2[1] + b1[0] * db2[1] - db1[1] * b2[0] - b1[1] * db2[0],
        };

        dR_dr6d[i][0] = db1[0]; dR_dr6d[i][1] = db2[0]; dR_dr6d[i][2] = db3[0];
        dR_dr6d[i][3] = db1[1]; dR_dr6d[i][4] = db2[1]; dR_dr6d[i][5] = db3[1];
        dR_dr6d[i][6] = db1[2]; dR_dr6d[i][7] = db2[2]; dR_dr6d[i][8] = db3[2];
    }
}

inline void MatrixToRotation6d(const float R[9], float r6d[6]) {
    // flatten first two columns
    r6d[0] = R[0]; r6d[1] = R[3]; r6d[2] = R[6];
    r6d[3] = R[1]; r6d[4] = R[4]; r6d[5] = R[7];
}

inline void MatMul3x3(const float A[9], const float B[9], float C[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float v = 0.0f;
            for (int k = 0; k < 3; ++k) {
                v += A[r * 3 + k] * B[k * 3 + c];
            }
            C[r * 3 + c] = v;
        }
    }
}

inline std::array<float, 3> MatVec3(const float A[9], const std::array<float, 3>& x) {
    return {
        A[0] * x[0] + A[1] * x[1] + A[2] * x[2],
        A[3] * x[0] + A[4] * x[1] + A[5] * x[2],
        A[6] * x[0] + A[7] * x[1] + A[8] * x[2],
    };
}

inline void RotationMatrixToAxisAngle(const float R[9], float aa[3]) {
    const float trace = R[0] + R[4] + R[8];
    float cosTheta = 0.5f * (trace - 1.0f);
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));
    const float theta = std::acos(cosTheta);
    if (theta < 1e-8f) {
        aa[0] = aa[1] = aa[2] = 0.0f;
        return;
    }
    const float sinTheta = std::sin(theta);
    if (std::abs(sinTheta) < 1e-8f) {
        aa[0] = aa[1] = aa[2] = 0.0f;
        return;
    }
    const float k = theta / (2.0f * sinTheta);
    aa[0] = k * (R[7] - R[5]);
    aa[1] = k * (R[2] - R[6]);
    aa[2] = k * (R[3] - R[1]);
}

inline float Gmof(float x, float sigma) {
    const float x2 = x * x;
    const float s2 = sigma * sigma;
    return (s2 * x2) / (s2 + x2);
}

}  // namespace

Stage1Config BuildDefaultStage1Config(
    const OptimizerDataInfo& info,
    const ObjectiveConstants& constants) {
    Stage1Config cfg;
    cfg.seqLen = info.sequence.sequenceLength;
    cfg.lowerBodyJointIds = {1, 4, 7, 2, 5, 8};
    cfg.jointMapping25 = constants.jointMapping25;
    cfg.jointIdxsPos = constants.jointIdxsPos;
    cfg.skeletonEdgesFlat = constants.skeletonEdgesFlat;
    cfg.contactJointIds = constants.contactJointIds;
    cfg.contactHeightLeftToeVids = constants.contactHeightLeftToeVids;
    cfg.contactHeightRightToeVids = constants.contactHeightRightToeVids;
    cfg.contactHeightLeftFootVids = constants.contactHeightLeftFootVids;
    cfg.contactHeightRightFootVids = constants.contactHeightRightFootVids;
    cfg.gmmNumGaussians = constants.gmmNumGaussians;
    cfg.gmmDim = constants.gmmDim;
    cfg.gmmMeans = constants.gmmMeans;
    cfg.gmmPrecisions = constants.gmmPrecisions;
    cfg.gmmLogNllWeights = constants.gmmLogNllWeights;
    cfg.kpConf = constants.kpConf;
    cfg.gmofSigma = constants.gmofSigma;
    cfg.imgFocal = constants.imgFocal;
    cfg.imgCenterX = constants.imgCenterX;
    cfg.imgCenterY = constants.imgCenterY;
    cfg.fps = info.sequence.fps;
    if (!info.stages.empty()) {
        cfg.weights.lossKpPosW = info.stages[0].lossKpPosW;
        cfg.weights.lossKpEdgeW = info.stages[0].lossKpEdgeW;
        cfg.weights.lossSmoothW = info.stages[0].lossSmoothW;
        cfg.weights.lossSmoothJ3dW = info.stages[0].lossSmoothJ3dW;
        cfg.weights.lossPriorW = info.stages[0].lossPriorW;
        cfg.weights.lossGmmPriorW = info.stages[0].lossGmmPriorW;
        cfg.weights.lossContactVelW = info.stages[0].lossContactVelW;
        cfg.weights.lossContactHeightW = info.stages[0].lossContactHeightW;
        cfg.weights.lossBelowFloorW = info.stages[0].lossBelowFloorW;
        cfg.optVars = info.stages[0].optVars;
    }
    return cfg;
}

// Helper function to check if a variable should be optimized
static bool ShouldOptimizeVar(const Stage1Config& cfg, const std::string& varName) {
    if (cfg.optVars.empty()) {
        return true;  // If optVars is empty, optimize all variables
    }
    for (const auto& v : cfg.optVars) {
        if (v.find(varName) != std::string::npos) {
            return true;
        }
    }
    return false;
}

size_t Stage1ParameterCount(const Stage1Config& cfg) {
    const size_t S = static_cast<size_t>(cfg.seqLen);
    const size_t lowerDim = static_cast<size_t>(cfg.lowerBodyJointIds.size() * 6);
    return S * 3 + 1 + 6 + 1 + S * lowerDim;
}

Stage1Params UnpackStage1Params(const Stage1Config& cfg, const std::vector<float>& x) {
    const size_t expected = Stage1ParameterCount(cfg);
    if (x.size() != expected) {
        throw std::runtime_error("Stage1 parameter size mismatch");
    }
    Stage1Params p;
    const size_t S = static_cast<size_t>(cfg.seqLen);
    const size_t lowerDim = static_cast<size_t>(cfg.lowerBodyJointIds.size() * 6);
    size_t off = 0;
    p.translDeltaSeq.assign(x.begin() + static_cast<std::ptrdiff_t>(off), x.begin() + static_cast<std::ptrdiff_t>(off + S * 3));
    off += S * 3;
    p.camHeightOffset = x[off++];
    p.camInitR6d.assign(x.begin() + static_cast<std::ptrdiff_t>(off), x.begin() + static_cast<std::ptrdiff_t>(off + 6));
    off += 6;
    p.camScale = x[off++];
    p.lowerBodyR6dSeq.assign(x.begin() + static_cast<std::ptrdiff_t>(off), x.begin() + static_cast<std::ptrdiff_t>(off + S * lowerDim));
    return p;
}

std::vector<float> PackStage1Params(const Stage1Config& cfg, const Stage1Params& p) {
    std::vector<float> x;
    x.reserve(Stage1ParameterCount(cfg));
    x.insert(x.end(), p.translDeltaSeq.begin(), p.translDeltaSeq.end());
    x.push_back(p.camHeightOffset);
    x.insert(x.end(), p.camInitR6d.begin(), p.camInitR6d.end());
    x.push_back(p.camScale);
    x.insert(x.end(), p.lowerBodyR6dSeq.begin(), p.lowerBodyR6dSeq.end());
    if (x.size() != Stage1ParameterCount(cfg)) {
        throw std::runtime_error("PackStage1Params produced wrong size");
    }
    return x;
}

Stage1Params BuildStage1InitialParams(const Stage1Config& cfg, const Stage1Data& data) {
    if (static_cast<int>(data.seqCases.size()) != cfg.seqLen) {
        throw std::runtime_error("BuildStage1InitialParams: seq size mismatch");
    }
    Stage1Params p;
    const size_t S = static_cast<size_t>(cfg.seqLen);
    const size_t lowerJoints = cfg.lowerBodyJointIds.size();
    p.translDeltaSeq.assign(S * 3, 0.0f);
    p.camHeightOffset = 0.0f;
    p.camInitR6d = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    p.camScale = 1.0f;
    p.lowerBodyR6dSeq.assign(S * lowerJoints * 6, 0.0f);

    for (size_t t = 0; t < S; ++t) {
        const auto& c = data.seqCases[t];
        if (!c.valid || c.smplxPose165.size() != 165) {
            continue;
        }
        for (size_t j = 0; j < lowerJoints; ++j) {
            const int jid = cfg.lowerBodyJointIds[j];
            const int aaOff = jid * 3;

            // Validate bounds to prevent out-of-range access
            if (jid < 0 || aaOff + 2 >= static_cast<int>(c.smplxPose165.size())) {
                throw std::runtime_error("Invalid joint ID " + std::to_string(jid) +
                    " results in out-of-bounds access to smplxPose165 (size: " +
                    std::to_string(c.smplxPose165.size()) + ")");
            }

            const float ax = c.smplxPose165[aaOff + 0];
            const float ay = c.smplxPose165[aaOff + 1];
            const float az = c.smplxPose165[aaOff + 2];
            float R[9];
            AxisAngleToRotation3x3(ax, ay, az, R);
            float r6d[6];
            MatrixToRotation6d(R, r6d);
            const size_t base = t * lowerJoints * 6 + j * 6;
            for (int k = 0; k < 6; ++k) {
                p.lowerBodyR6dSeq[base + static_cast<size_t>(k)] = r6d[k];
            }
        }
    }
    return p;
}

Stage1ObjectiveEval EvaluateStage1ObjectiveAndGradient(
    const SmplxForwardLite& smplx,
    const Stage1Config& cfg,
    const Stage1Data& data,
    const std::vector<float>& x,
    std::shared_ptr<TITAN_NAMESPACE::TaskThreadPool> threadPool) {


    if (cfg.contactJointIds.size() != 4)
    {
        throw std::runtime_error("There must be 4 contact joint ids");
    }
    const Stage1Params p = UnpackStage1Params(cfg, x);

    const size_t S = static_cast<size_t>(cfg.seqLen);
    if (data.seqCases.size() != S) {
        throw std::runtime_error("Stage1 objective seq size mismatch");
    }
    const size_t numLower = cfg.lowerBodyJointIds.size();
    const size_t lowerDim = numLower * 6;
    const int jacDimGlobal = static_cast<int>(lowerDim);
    const size_t translDim = S * 3;
    const size_t idxCamH = translDim;
    const size_t idxCamR6D = idxCamH + 1;
    const size_t idxCamScale = idxCamR6D + 6;
    const size_t totalDim = Stage1ParameterCount(cfg);
    Stage1ObjectiveEval out;
    out.gradient.assign(totalDim, 0.0f);

    float cam_init_R[9];
    Rotation6dToMatrix(p.camInitR6d.data(), cam_init_R);

    float lossKpPosTotal = 0.0f;
    float lossKpEdgeTotal = 0.0f;
    float lossContactHeightTotal = 0.0f;
    float lossBelowFloorTotal = 0.0f;
    float maskSum = 0.0f;
    std::array<float, 9> dL_dR_total{};
    dL_dR_total.fill(0.0f);

    std::vector<std::array<float, 3>> translTotal(S);
    std::vector<float> mask(S, 0.0f);
    std::vector<SmplxSelectedJointsLowerBodyJacobianOutput> frameLowerJacs(S);
    std::vector<char> frameValid(S, 0);

    std::vector<int> selectedJointIds;
    std::unordered_map<int, int> selectedJointSlot;
    selectedJointIds.reserve(cfg.jointMapping25.size() + cfg.contactJointIds.size());
    for (int jidx : cfg.jointMapping25) {
        if (jidx < 0) {
            continue;
        }
        if (selectedJointSlot.find(jidx) == selectedJointSlot.end()) {
            selectedJointSlot[jidx] = static_cast<int>(selectedJointIds.size());
            selectedJointIds.push_back(jidx);
        }
    }
    for (int jidx : cfg.contactJointIds) {
        if (jidx < 0) {
            continue;
        }
        if (selectedJointSlot.find(jidx) == selectedJointSlot.end()) {
            selectedJointSlot[jidx] = static_cast<int>(selectedJointIds.size());
            selectedJointIds.push_back(jidx);
        }
    }
    std::vector<int> selectedVertexIds;
    selectedVertexIds.reserve(
        cfg.contactHeightLeftToeVids.size() +
        cfg.contactHeightRightToeVids.size() +
        cfg.contactHeightLeftFootVids.size() +
        cfg.contactHeightRightFootVids.size());
    auto addUniqueVid = [&](int vid) {
        if (vid < 0) {
            return;
        }
        if (std::find(selectedVertexIds.begin(), selectedVertexIds.end(), vid) == selectedVertexIds.end()) {
            selectedVertexIds.push_back(vid);
        }
    };
    for (int vid : cfg.contactHeightLeftToeVids) { addUniqueVid(vid); }
    for (int vid : cfg.contactHeightRightToeVids) { addUniqueVid(vid); }
    for (int vid : cfg.contactHeightLeftFootVids) { addUniqueVid(vid); }
    for (int vid : cfg.contactHeightRightFootVids) { addUniqueVid(vid); }
    // Build per-frame validity/mask once so objective and gradient share identical normalization.
    for (size_t t = 0; t < S; ++t) {
        const SequenceFrameCase& c = data.seqCases[t];
        translTotal[t] = {
            p.translDeltaSeq[t * 3 + 0],
            p.translDeltaSeq[t * 3 + 1],
            p.translDeltaSeq[t * 3 + 2],
        };
        if (!c.valid) {
            continue;
        }
        mask[t] = 1.0f;
        maskSum += 1.0f;
        translTotal[t][0] += c.smplxTrans3[0];
        translTotal[t][1] += c.smplxTrans3[1];
        translTotal[t][2] += c.smplxTrans3[2];
    }
    const float maskSumSafe = std::max(1.0f, maskSum);
    const float validNormDenom = maskSumSafe;

    auto computeFrameForwardAndJacobian = [&](int start, int end) {
        for (int ti = start; ti < end; ++ti) {
            const size_t t = static_cast<size_t>(ti);
            const SequenceFrameCase& c = data.seqCases[t];
            if (mask[t] <= 0.0f) {
                continue;
            }
            SmplxForwardInput in;
            in.pose165 = c.smplxPose165;
            std::vector<float> frameR6D(lowerDim, 0.0f);
            for (size_t k = 0; k < lowerDim; ++k) {
                frameR6D[k] = p.lowerBodyR6dSeq[t * lowerDim + k];
            }
            frameLowerJacs[t] = smplx.ForwardSelectedJointsWithLowerBodyR6dJacobian(
                in, frameR6D);
            frameValid[t] = 1;
        }
    };

    // Note: Cache warmup no longer needed - SetSmplxEvaluationConstants has already
    // pre-computed all data (shaped vertices, dependencies, sparse matrices) before optimization


    if (threadPool && S > 1) {
        threadPool->AddTaskRangeAndWait(
            static_cast<int>(S),
            [&](int start, int end) { computeFrameForwardAndJacobian(start, end); });
    } else {
        computeFrameForwardAndJacobian(0, static_cast<int>(S));
    }



    for (size_t t = 0; t < S; ++t) {
        const SequenceFrameCase& c = data.seqCases[t];
        if (mask[t] <= 0.0f || frameValid[t] == 0) {
            continue;
        }
        const auto& selectedXyz = frameLowerJacs[t].selectedJointsXyz;
        const auto& selectedJac = frameLowerJacs[t].selectedJointsJacobian;
        const int jacDim = frameLowerJacs[t].lowerDim;
        const auto& selectedVxyz = frameLowerJacs[t].selectedVerticesXyz;
        const auto& selectedVjac = frameLowerJacs[t].selectedVerticesJacobian;
        std::unordered_map<int, int> selectedVidSlot;
        selectedVidSlot.reserve(frameLowerJacs[t].selectedVertexIds.size());
        for (size_t i = 0; i < frameLowerJacs[t].selectedVertexIds.size(); ++i) {
            selectedVidSlot[frameLowerJacs[t].selectedVertexIds[i]] = static_cast<int>(i);
        }

        float rcwMod[9];
        MatMul3x3(cam_init_R, c.cameraRcw9.data(), rcwMod);
        std::array<float, 3> tcw{
            c.cameraTcw3[0],
            c.cameraTcw3[1],
            c.cameraTcw3[2],
        };
        std::array<float, 3> tcwPre = MatVec3(cam_init_R, tcw);
        tcwPre[1] += p.camHeightOffset;
        std::array<float, 3> tcwMod = tcwPre;
        tcwMod[0] *= p.camScale;
        tcwMod[1] *= p.camScale;
        tcwMod[2] *= p.camScale;

        std::array<std::array<float, 2>, 25> pj{};
        std::array<std::array<float, 3>, 25> jcCache{};
        std::array<std::array<float, 3>, 25> jwCache{};
        std::array<int, 25> validJoint{};
        std::array<int, 25> jointSlotCache{};
        validJoint.fill(0);
        jointSlotCache.fill(-1);
        for (int k = 0; k < 25; ++k) {
            int jidx = -1;
            if (k < static_cast<int>(cfg.jointMapping25.size())) {
                jidx = cfg.jointMapping25[static_cast<size_t>(k)];
            }
            const auto sit = selectedJointSlot.find(jidx);
            if (jidx < 0 || sit == selectedJointSlot.end()) {
                continue;
            }
            const int jslot = sit->second;
            validJoint[k] = 1;
            jointSlotCache[k] = jslot;
            const std::array<float, 3> jw{
                selectedXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t][0],
                selectedXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t][1],
                selectedXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t][2],
            };
            jwCache[k] = jw;
            const std::array<float, 3> jc0 = MatVec3(rcwMod, jw);
            const std::array<float, 3> jc{jc0[0] + tcwMod[0], jc0[1] + tcwMod[1], jc0[2] + tcwMod[2]};
            jcCache[k] = jc;
            const float z = std::max(1e-6f, jc[2]);
            pj[k][0] = cfg.imgFocal * (jc[0] / z) + cfg.imgCenterX;
            pj[k][1] = cfg.imgFocal * (jc[1] / z) + cfg.imgCenterY;
        }

        auto addPointGrad = [&](int k, float dL_dpx, float dL_dpy) {
            if (k < 0 || k >= 25 || validJoint[k] == 0) {
                return;
            }
            const float jcx = jcCache[k][0];
            const float jcy = jcCache[k][1];
            const float jcz = std::max(1e-6f, jcCache[k][2]);
            const float invz = 1.0f / jcz;
            const float invz2 = invz * invz;
            const float dpxDjx = cfg.imgFocal * invz;
            const float dpxDjz = -cfg.imgFocal * jcx * invz2;
            const float dpyDjy = cfg.imgFocal * invz;
            const float dpyDjz = -cfg.imgFocal * jcy * invz2;
            const std::array<float, 3> dL_djc{
                dL_dpx * dpxDjx,
                dL_dpy * dpyDjy,
                dL_dpx * dpxDjz + dL_dpy * dpyDjz,
            };
            const std::array<float, 3> dL_djw{
                rcwMod[0] * dL_djc[0] + rcwMod[3] * dL_djc[1] + rcwMod[6] * dL_djc[2],
                rcwMod[1] * dL_djc[0] + rcwMod[4] * dL_djc[1] + rcwMod[7] * dL_djc[2],
                rcwMod[2] * dL_djc[0] + rcwMod[5] * dL_djc[1] + rcwMod[8] * dL_djc[2],
            };
            const size_t toff = t * 3;
            out.gradient[toff + 0] += dL_djw[0];
            out.gradient[toff + 1] += dL_djw[1];
            out.gradient[toff + 2] += dL_djw[2];
            out.gradient[idxCamH] += p.camScale * dL_djc[1];
            out.gradient[idxCamScale] += dL_djc[0] * tcwPre[0] + dL_djc[1] * tcwPre[1] + dL_djc[2] * tcwPre[2];

            const int jslot = jointSlotCache[k];
            if (jslot >= 0 && frameValid[t] != 0) {
                if (jacDim > 0) {
                    const size_t goff = (idxCamScale + 1) + t * static_cast<size_t>(jacDim);
                    const int row0 = jslot * 3 + 0;
                    const int row1 = jslot * 3 + 1;
                    const int row2 = jslot * 3 + 2;
                    for (int d = 0; d < jacDim; ++d) {
                        const float jx = selectedJac[static_cast<size_t>(row0) * static_cast<size_t>(jacDim) + static_cast<size_t>(d)];
                        const float jy = selectedJac[static_cast<size_t>(row1) * static_cast<size_t>(jacDim) + static_cast<size_t>(d)];
                        const float jz = selectedJac[static_cast<size_t>(row2) * static_cast<size_t>(jacDim) + static_cast<size_t>(d)];
                        out.gradient[goff + static_cast<size_t>(d)] +=
                            dL_djw[0] * jx + dL_djw[1] * jy + dL_djw[2] * jz;
                    }
                }
            }

            // Accumulate dL/dR through rcwMod = R * rcw and tcwPre = R * tcw.
            const auto& jw = jwCache[k];
            for (int a = 0; a < 3; ++a) {
                for (int r = 0; r < 3; ++r) {
                    float sumRcw = 0.0f;
                    for (int b = 0; b < 3; ++b) {
                        sumRcw += c.cameraRcw9[r * 3 + b] * jw[static_cast<size_t>(b)];
                    }
                    dL_dR_total[static_cast<size_t>(a) * 3 + static_cast<size_t>(r)] += dL_djc[static_cast<size_t>(a)] * sumRcw;
                }
            }
            const std::array<float, 3> dL_dtcw_pre{
                p.camScale * dL_djc[0],
                p.camScale * dL_djc[1],
                p.camScale * dL_djc[2],
            };
            for (int a = 0; a < 3; ++a) {
                for (int r = 0; r < 3; ++r) {
                    dL_dR_total[static_cast<size_t>(a) * 3 + static_cast<size_t>(r)] += dL_dtcw_pre[static_cast<size_t>(a)] * tcw[static_cast<size_t>(r)];
                }
            }
        };

        const float bboxH = std::max(1e-6f, c.bbox4[3] - c.bbox4[1]);
        float framePosSum = 0.0f;
        const float posDenom = std::max(1.0f, static_cast<float>(cfg.jointIdxsPos.size()));
        const float posWBase = (cfg.weights.lossKpPosW / validNormDenom) / posDenom;
        for (int kpIdx : cfg.jointIdxsPos) {
            if (kpIdx < 0 || kpIdx >= 25) {
                continue;
            }
            const float conf = c.keypoints2d25x3[static_cast<size_t>(kpIdx) * 3 + 2];
            const float w = (conf > cfg.kpConf) ? conf : 0.0f;
            const float gx = c.keypoints2d25x3[static_cast<size_t>(kpIdx) * 3 + 0];
            const float gy = c.keypoints2d25x3[static_cast<size_t>(kpIdx) * 3 + 1];
            const float dx = pj[static_cast<size_t>(kpIdx)][0] - gx;
            const float dy = pj[static_cast<size_t>(kpIdx)][1] - gy;
            const float ex = Gmof(dx, cfg.gmofSigma) / bboxH;
            const float ey = Gmof(dy, cfg.gmofSigma) / bboxH;
            framePosSum += w * (0.5f * (ex + ey));

            if (w > 0.0f) {
                const float s2 = cfg.gmofSigma * cfg.gmofSigma;

                // Guard against 0/0 in Geman-McClure derivative when sigma=0
                // When s2=0, the robustifier saturates to zero and derivative is zero
                float dgmofDx = 0.0f;
                float dgmofDy = 0.0f;

                if (s2 > 1e-12f) {
                    const float denom_x = s2 + dx * dx;
                    const float denom_y = s2 + dy * dy;
                    dgmofDx = (2.0f * s2 * s2 * dx) / (denom_x * denom_x);
                    dgmofDy = (2.0f * s2 * s2 * dy) / (denom_y * denom_y);
                }
                // else: s2 ≈ 0, derivative is zero (robustifier has no influence)

                const float dL_dpx = posWBase * w * 0.5f * (dgmofDx / bboxH);
                const float dL_dpy = posWBase * w * 0.5f * (dgmofDy / bboxH);
                addPointGrad(kpIdx, dL_dpx, dL_dpy);
            }
        }
        lossKpPosTotal += framePosSum / posDenom;

        float frameEdgeSum = 0.0f;
        const size_t edgeCount = cfg.skeletonEdgesFlat.size() / 2;
        const float edgeDenom = std::max(1.0f, static_cast<float>(edgeCount));
        const float edgeWBase = (cfg.weights.lossKpEdgeW / validNormDenom) / edgeDenom;

        for (size_t e = 0; e < edgeCount; ++e) {
            const int k0 = cfg.skeletonEdgesFlat[e * 2 + 0];
            const int k1 = cfg.skeletonEdgesFlat[e * 2 + 1];
            if (k0 < 0 || k0 >= 25 || k1 < 0 || k1 >= 25) {
                continue;
            }
            const float c0 = c.keypoints2d25x3[static_cast<size_t>(k0) * 3 + 2];
            const float c1 = c.keypoints2d25x3[static_cast<size_t>(k1) * 3 + 2];
            const float w0 = (c0 > cfg.kpConf) ? c0 : 0.0f;
            const float w1 = (c1 > cfg.kpConf) ? c1 : 0.0f;
            const float edgeConf = std::min(w0, w1);
            const float predEx = pj[static_cast<size_t>(k0)][0] - pj[static_cast<size_t>(k1)][0];
            const float predEy = pj[static_cast<size_t>(k0)][1] - pj[static_cast<size_t>(k1)][1];
            const float gtEx = c.keypoints2d25x3[static_cast<size_t>(k0) * 3 + 0] - c.keypoints2d25x3[static_cast<size_t>(k1) * 3 + 0];
            const float gtEy = c.keypoints2d25x3[static_cast<size_t>(k0) * 3 + 1] - c.keypoints2d25x3[static_cast<size_t>(k1) * 3 + 1];
            const float predN = std::sqrt(predEx * predEx + predEy * predEy) + 1e-8f;
            const float gtN = std::sqrt(gtEx * gtEx + gtEy * gtEy) + 1e-8f;
            const float cosv = (predEx * gtEx + predEy * gtEy) / (predN * gtN);
            const float clamped = std::max(-1.0f, std::min(1.0f, cosv));
            frameEdgeSum += (1.0f - clamped) * edgeConf;

            if (edgeConf > 0.0f) {
                const float uu = predEx * predEx + predEy * predEy + 1e-8f;
                const float vv = gtEx * gtEx + gtEy * gtEy + 1e-8f;
                const float invU = 1.0f / std::sqrt(uu);
                const float invV = 1.0f / std::sqrt(vv);
                const float dot = predEx * gtEx + predEy * gtEy;
                const float cosRaw = dot * invU * invV;
                const float dcosDuX = gtEx * invU * invV - cosRaw * predEx / uu;
                const float dcosDuY = gtEy * invU * invV - cosRaw * predEy / uu;
                const float dL_du_x = -edgeWBase * edgeConf * dcosDuX;
                const float dL_du_y = -edgeWBase * edgeConf * dcosDuY;
                addPointGrad(k0, dL_du_x, dL_du_y);
                addPointGrad(k1, -dL_du_x, -dL_du_y);
            }
        }
        lossKpEdgeTotal += frameEdgeSum / edgeDenom;

        // Contact height term with contact confidence weighting.
        if (cfg.weights.lossContactHeightW > 0.0f && !selectedVidSlot.empty()) {
            // Extract and apply sigmoid to contact predictions for this frame.
            std::array<float, 4> contactConf = {1.0f, 1.0f, 1.0f, 1.0f};  // Default if no predictions
            if (t < data.seqCases.size() && data.seqCases[t].staticConfLogits.size() >= 4) {
                for (int i = 0; i < 4; ++i) {
                    const float logit = data.seqCases[t].staticConfLogits[static_cast<size_t>(i)];
                    contactConf[static_cast<size_t>(i)] = 1.0f / (1.0f + std::exp(-logit));  // sigmoid
                }
            }

            const float groups = 4.0f;
            auto groupAbsMinY = [&](const std::vector<int>& vids, float& outAbs, int& outMinVid, float& outMinY) {
                outAbs = 0.0f;
                outMinVid = -1;
                outMinY = 0.0f;
                if (vids.empty()) {
                    return;
                }
                bool inited = false;
                float bestY = 0.0f;
                int bestVid = -1;
                for (int vid : vids) {
                    const auto itv = selectedVidSlot.find(vid);
                    if (itv == selectedVidSlot.end()) {
                        continue;
                    }
                    const int vslot = itv->second;
                    const float y = selectedVxyz[static_cast<size_t>(vslot) * 3 + 1] + translTotal[t][1];
                    if (!inited || y < bestY) {
                        inited = true;
                        bestY = y;
                        bestVid = vid;
                    }
                }
                if (!inited) {
                    return;
                }
                outMinY = bestY;
                outMinVid = bestVid;
                outAbs = std::abs(bestY);
            };
            float absLf = 0.0f, absLt = 0.0f, absRf = 0.0f, absRt = 0.0f;
            int vidLf = -1, vidLt = -1, vidRf = -1, vidRt = -1;
            float yLf = 0.0f, yLt = 0.0f, yRf = 0.0f, yRt = 0.0f;
            groupAbsMinY(cfg.contactHeightLeftFootVids, absLf, vidLf, yLf);
            groupAbsMinY(cfg.contactHeightLeftToeVids, absLt, vidLt, yLt);
            groupAbsMinY(cfg.contactHeightRightFootVids, absRf, vidRf, yRf);
            groupAbsMinY(cfg.contactHeightRightToeVids, absRt, vidRt, yRt);

            // Weight each foot group by its corresponding contact confidence.
            // Mapping: [L_Ankle→leftFoot, L_foot→leftToe, R_Ankle→rightFoot, R_foot→rightToe]
            const float weightedLf = absLf * contactConf[0];
            const float weightedLt = absLt * contactConf[1];
            const float weightedRf = absRf * contactConf[2];
            const float weightedRt = absRt * contactConf[3];
            const float frameContactH = (weightedLf + weightedLt + weightedRf + weightedRt) / groups;
            lossContactHeightTotal += frameContactH;

            const float baseCoeff = (cfg.weights.lossContactHeightW / validNormDenom) / groups;
            auto addVertexYGrad = [&](int vid, float yval, float conf) {
                if (vid < 0) {
                    return;
                }
                const auto itv = selectedVidSlot.find(vid);
                if (itv == selectedVidSlot.end()) {
                    return;
                }
                const int vslot = itv->second;
                const float sign = (yval > 0.0f) ? 1.0f : ((yval < 0.0f) ? -1.0f : 0.0f);
                const float dL_dy = baseCoeff * conf * sign;
                out.gradient[t * 3 + 1] += dL_dy;
                if (jacDim > 0) {
                    const size_t goff = (idxCamScale + 1) + t * static_cast<size_t>(jacDim);
                    const int row = vslot * 3 + 1;
                    for (int d = 0; d < jacDim; ++d) {
                        const float jy = selectedVjac[static_cast<size_t>(row) * static_cast<size_t>(jacDim) + static_cast<size_t>(d)];
                        out.gradient[goff + static_cast<size_t>(d)] += dL_dy * jy;
                    }
                }
            };
            addVertexYGrad(vidLf, yLf, contactConf[0]);
            addVertexYGrad(vidLt, yLt, contactConf[1]);
            addVertexYGrad(vidRf, yRf, contactConf[2]);
            addVertexYGrad(vidRt, yRt, contactConf[3]);
        }

        // Below-floor term over joints used by mapping + contact joints (same structural intent as Python).
        if (cfg.weights.lossBelowFloorW > 0.0f) {
            std::vector<int> floorJointIds = cfg.jointMapping25;
            floorJointIds.insert(floorJointIds.end(), cfg.contactJointIds.begin(), cfg.contactJointIds.end());
            const float denom = std::max(1.0f, static_cast<float>(floorJointIds.size()));
            float frameFloorSum = 0.0f;
            const float coeff = (cfg.weights.lossBelowFloorW / validNormDenom) / denom;
            for (int jid : floorJointIds) {
                const auto sit = selectedJointSlot.find(jid);
                if (jid < 0 || sit == selectedJointSlot.end()) {
                    continue;
                }
                const int jslot = sit->second;
                const float y = selectedXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t][1];
                const float relu = std::max(0.0f, -y);
                frameFloorSum += relu;
                if (y < 0.0f) {
                    const float dL_dy = -coeff;
                    out.gradient[t * 3 + 1] += dL_dy;
                    if (jacDim > 0) {
                        const size_t goff = (idxCamScale + 1) + t * static_cast<size_t>(jacDim);
                        const int row = jslot * 3 + 1;
                        for (int d = 0; d < jacDim; ++d) {
                            const float jy = selectedJac[static_cast<size_t>(row) * static_cast<size_t>(jacDim) + static_cast<size_t>(d)];
                            out.gradient[goff + static_cast<size_t>(d)] += dL_dy * jy;
                        }
                    }
                }
            }
            lossBelowFloorTotal += frameFloorSum / denom;
        }
    }


    // Analytic camInitR6d Jacobian via dL/dR and dR/dr6d chain rule.
    {
        float R_tmp[9];
        float dR_dr6d[6][9];
        Rotation6dToMatrixAndJacobians(p.camInitR6d.data(), R_tmp, dR_dr6d);
        for (int i = 0; i < 6; ++i) {
            float gi = 0.0f;
            for (int q = 0; q < 9; ++q) {
                gi += dL_dR_total[static_cast<size_t>(q)] * dR_dr6d[i][q];
            }
            out.gradient[idxCamR6D + static_cast<size_t>(i)] = gi;
        }
    }


    float lossKpPos = (maskSum > 0.0f) ? (lossKpPosTotal / validNormDenom) : 0.0f;
    float lossKpEdge = (maskSum > 0.0f) ? (lossKpEdgeTotal / validNormDenom) : 0.0f;
    float lossContactHeight = (maskSum > 0.0f) ? (lossContactHeightTotal / validNormDenom) : 0.0f;
    float lossBelowFloor = (maskSum > 0.0f) ? (lossBelowFloorTotal / validNormDenom) : 0.0f;
    float lossSmoothJ3D = 0.0f;
    float lossPrior = 0.0f;
    float lossGmmPrior = 0.0f;


    float velSum = 0.0f;
    float velDen = 0.0f;
    const float smoothW = cfg.weights.lossSmoothW;
    for (size_t t = 1; t < S; ++t) {
        const float vx = translTotal[t][0] - translTotal[t - 1][0];
        const float vy = translTotal[t][1] - translTotal[t - 1][1];
        const float vz = translTotal[t][2] - translTotal[t - 1][2];
        const float dx = vx * cfg.fps;
        const float dy = vy * cfg.fps;
        const float dz = vz * cfg.fps;
        const float val = (dx * dx + dy * dy + dz * dz) / 3.0f;
        velSum += val * mask[t];
        velDen += mask[t];
    }
    const float vel = (velDen > 0.0f) ? (velSum / velDen) : 0.0f;

    if (velDen > 0.0f) {
        const float c = smoothW / velDen;
        const float scale = (2.0f * cfg.fps * cfg.fps) / 3.0f;
        for (size_t t = 1; t < S; ++t) {
            if (mask[t] <= 0.0f) {
                continue;
            }
            const float vx = translTotal[t][0] - translTotal[t - 1][0];
            const float vy = translTotal[t][1] - translTotal[t - 1][1];
            const float vz = translTotal[t][2] - translTotal[t - 1][2];
            const float k = c * mask[t] * scale;
            out.gradient[t * 3 + 0] += k * vx;
            out.gradient[t * 3 + 1] += k * vy;
            out.gradient[t * 3 + 2] += k * vz;
            out.gradient[(t - 1) * 3 + 0] -= k * vx;
            out.gradient[(t - 1) * 3 + 1] -= k * vy;
            out.gradient[(t - 1) * 3 + 2] -= k * vz;
        }
    }

    float accSum = 0.0f;
    float accDen = 0.0f;
    for (size_t t = 1; t + 1 < S; ++t) {
        const float ax = (translTotal[t + 1][0] + translTotal[t - 1][0] - 2.0f * translTotal[t][0]) * cfg.fps;
        const float ay = (translTotal[t + 1][1] + translTotal[t - 1][1] - 2.0f * translTotal[t][1]) * cfg.fps;
        const float az = (translTotal[t + 1][2] + translTotal[t - 1][2] - 2.0f * translTotal[t][2]) * cfg.fps;
        const float val = (ax * ax + ay * ay + az * az);  // Sum over x,y,z to match Python .sum(-1)
        accSum += val * mask[t];
        accDen += mask[t];
    }
    const float acc = (accDen > 0.0f) ? (accSum / accDen) : 0.0f;
    const float lossSmooth = vel + 10.0f * acc;
    float lossContactVel = 0.0f;
    if (cfg.weights.lossContactVelW > 0.0f && S > 1 && !cfg.contactJointIds.empty()) {
        float velContactSum = 0.0f;
        float velContactDen = 0.0f;
        const float contacts = static_cast<float>(cfg.contactJointIds.size());
        const float fpsScale = ContactVelocityFpsScale(cfg.fps);
        for (size_t t = 1; t < S; ++t) {
            if (mask[t] <= 0.0f || frameValid[t] == 0 || frameValid[t - 1] == 0) {
                continue;
            }

            // Extract and apply sigmoid to contact predictions for frame t (current frame).
            std::array<float, 4> contactConf = {1.0f, 1.0f, 1.0f, 1.0f};  // Default if no predictions
            if (t < data.seqCases.size() && data.seqCases[t].staticConfLogits.size() >= 4) {
                for (int i = 0; i < 4; ++i) {
                    const float logit = data.seqCases[t].staticConfLogits[static_cast<size_t>(i)];
                    contactConf[static_cast<size_t>(i)] = 1.0f / (1.0f + std::exp(-logit));  // sigmoid
                }
            }

            float frameSum = 0.0f;
            if (cfg.contactJointIds.size() )
            for (size_t jidx = 0; jidx < cfg.contactJointIds.size(); ++jidx) {
                const int jid = cfg.contactJointIds[jidx];
                const auto sit = selectedJointSlot.find(jid);
                if (sit == selectedJointSlot.end()) {
                    continue;
                }
                const int jslot = sit->second;
                const std::array<float, 3> jCurr{
                    frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t][0],
                    frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t][1],
                    frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t][2],
                };
                const std::array<float, 3> jPrev{
                    frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t - 1][0],
                    frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t - 1][1],
                    frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t - 1][2],
                };
                const std::array<float, 3> d{
                    jCurr[0] - jPrev[0],
                    jCurr[1] - jPrev[1],
                    jCurr[2] - jPrev[2],
                };
                // Squared velocity: (displacement * fps)² = displacement² * fps²
                const float sqDist = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                frameSum += sqDist * fpsScale * contactConf[jidx];
            }
            velContactSum += (frameSum / std::max(1.0f, contacts)) * mask[t];
            velContactDen += mask[t];
        }
        lossContactVel = (velContactDen > 0.0f) ? (velContactSum / velContactDen) : 0.0f;

        if (velContactDen > 0.0f) {
            const float baseCoeff = cfg.weights.lossContactVelW / (velContactDen * std::max(1.0f, contacts)) * fpsScale;
            for (size_t t = 1; t < S; ++t) {
                if (mask[t] <= 0.0f || frameValid[t] == 0 || frameValid[t - 1] == 0) {
                    continue;
                }

                // Extract contact confidence for frame t.
                std::array<float, 4> contactConf = {1.0f, 1.0f, 1.0f, 1.0f};
                if (t < data.seqCases.size() && data.seqCases[t].staticConfLogits.size() >= 4) {
                    for (int i = 0; i < 4; ++i) {
                        const float logit = data.seqCases[t].staticConfLogits[static_cast<size_t>(i)];
                        contactConf[static_cast<size_t>(i)] = 1.0f / (1.0f + std::exp(-logit));
                    }
                }

                const float mt = mask[t];
                const size_t goffT = (idxCamScale + 1) + t * static_cast<size_t>(jacDimGlobal);
                const size_t goffP = (idxCamScale + 1) + (t - 1) * static_cast<size_t>(jacDimGlobal);
                for (size_t jidx = 0; jidx < cfg.contactJointIds.size(); ++jidx) {
                    const int jid = cfg.contactJointIds[jidx];
                    const auto sit = selectedJointSlot.find(jid);
                    if (sit == selectedJointSlot.end()) {
                        continue;
                    }
                    const int jslot = sit->second;
                    const std::array<float, 3> d{
                        frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t][0]
                            - (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t - 1][0]),
                        frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t][1]
                            - (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t - 1][1]),
                        frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t][2]
                            - (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t - 1][2]),
                    };
                    const float conf = contactConf[jidx];
                    // Gradient: d(sqVel)/d(displacement) = d(d²/dt²)/d(d) = 2d/dt²
                    const std::array<float, 3> dL_dd{
                        baseCoeff * mt * conf * 2.0f * d[0],
                        baseCoeff * mt * conf * 2.0f * d[1],
                        baseCoeff * mt * conf * 2.0f * d[2],
                    };
                    out.gradient[t * 3 + 0] += dL_dd[0];
                    out.gradient[t * 3 + 1] += dL_dd[1];
                    out.gradient[t * 3 + 2] += dL_dd[2];
                    out.gradient[(t - 1) * 3 + 0] -= dL_dd[0];
                    out.gradient[(t - 1) * 3 + 1] -= dL_dd[1];
                    out.gradient[(t - 1) * 3 + 2] -= dL_dd[2];

                    if (jacDimGlobal > 0) {
                        const int row = jslot * 3;
                        for (int dvar = 0; dvar < jacDimGlobal; ++dvar) {
                            const float jtx = frameLowerJacs[t].selectedJointsJacobian[(static_cast<size_t>(row) + 0) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jty = frameLowerJacs[t].selectedJointsJacobian[(static_cast<size_t>(row) + 1) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jtz = frameLowerJacs[t].selectedJointsJacobian[(static_cast<size_t>(row) + 2) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jpx = frameLowerJacs[t - 1].selectedJointsJacobian[(static_cast<size_t>(row) + 0) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jpy = frameLowerJacs[t - 1].selectedJointsJacobian[(static_cast<size_t>(row) + 1) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jpz = frameLowerJacs[t - 1].selectedJointsJacobian[(static_cast<size_t>(row) + 2) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            out.gradient[goffT + static_cast<size_t>(dvar)] += dL_dd[0] * jtx + dL_dd[1] * jty + dL_dd[2] * jtz;
                            out.gradient[goffP + static_cast<size_t>(dvar)] -= dL_dd[0] * jpx + dL_dd[1] * jpy + dL_dd[2] * jpz;
                        }
                    }
                }
            }
        }
    }
    if (cfg.weights.lossSmoothJ3dW > 0.0f && S > 1) {
        std::vector<int> smoothJoints = cfg.jointMapping25;
        smoothJoints.insert(smoothJoints.end(), cfg.contactJointIds.begin(), cfg.contactJointIds.end());

        // Count ONLY valid joints that will actually be processed (match Python behavior)
        int validJointCount = 0;
        for (int jid : smoothJoints) {
            if (selectedJointSlot.find(jid) != selectedJointSlot.end()) {
                validJointCount++;
            }
        }
        const float nj = std::max(1.0f, static_cast<float>(validJointCount));

        float j3dSum = 0.0f;
        float j3dDen = 0.0f;
        for (size_t t = 1; t < S; ++t) {
            if (mask[t] <= 0.0f || frameValid[t] == 0 || frameValid[t - 1] == 0) {
                continue;
            }
            float frameSum = 0.0f;
            for (int jid : smoothJoints) {
                const auto sit = selectedJointSlot.find(jid);
                if (sit == selectedJointSlot.end()) {
                    continue;
                }
                const int jslot = sit->second;
                const float dx =
                    (frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t][0]) -
                    (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t - 1][0]);
                const float dy =
                    (frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t][1]) -
                    (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t - 1][1]);
                const float dz =
                    (frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t][2]) -
                    (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t - 1][2]);
                frameSum += (dx * dx + dy * dy + dz * dz) * (cfg.fps * cfg.fps);
            }
            j3dSum += (frameSum / nj) * mask[t];
            j3dDen += mask[t];
        }
        lossSmoothJ3D = (j3dDen > 0.0f) ? (j3dSum / j3dDen) : 0.0f;

        if (j3dDen > 0.0f) {
            const float coeff = cfg.weights.lossSmoothJ3dW / (j3dDen * nj);
            for (size_t t = 1; t < S; ++t) {
                if (mask[t] <= 0.0f || frameValid[t] == 0 || frameValid[t - 1] == 0) {
                    continue;
                }
                const float mt = mask[t];
                const size_t goffT = (idxCamScale + 1) + t * static_cast<size_t>(jacDimGlobal);
                const size_t goffP = (idxCamScale + 1) + (t - 1) * static_cast<size_t>(jacDimGlobal);
                for (int jid : smoothJoints) {
                    const auto sit = selectedJointSlot.find(jid);
                    if (sit == selectedJointSlot.end()) {
                        continue;
                    }
                    const int jslot = sit->second;
                    const std::array<float, 3> d{
                        frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t][0]
                            - (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 0] + translTotal[t - 1][0]),
                        frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t][1]
                            - (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 1] + translTotal[t - 1][1]),
                        frameLowerJacs[t].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t][2]
                            - (frameLowerJacs[t - 1].selectedJointsXyz[static_cast<size_t>(jslot) * 3 + 2] + translTotal[t - 1][2]),
                    };
                    const float s = coeff * mt * 2.0f * cfg.fps * cfg.fps;
                    const std::array<float, 3> dL_dd{s * d[0], s * d[1], s * d[2]};
                    out.gradient[t * 3 + 0] += dL_dd[0];
                    out.gradient[t * 3 + 1] += dL_dd[1];
                    out.gradient[t * 3 + 2] += dL_dd[2];
                    out.gradient[(t - 1) * 3 + 0] -= dL_dd[0];
                    out.gradient[(t - 1) * 3 + 1] -= dL_dd[1];
                    out.gradient[(t - 1) * 3 + 2] -= dL_dd[2];

                    if (jacDimGlobal > 0) {
                        const int row = jslot * 3;
                        for (int dvar = 0; dvar < jacDimGlobal; ++dvar) {
                            const float jtx = frameLowerJacs[t].selectedJointsJacobian[(static_cast<size_t>(row) + 0) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jty = frameLowerJacs[t].selectedJointsJacobian[(static_cast<size_t>(row) + 1) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jtz = frameLowerJacs[t].selectedJointsJacobian[(static_cast<size_t>(row) + 2) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jpx = frameLowerJacs[t - 1].selectedJointsJacobian[(static_cast<size_t>(row) + 0) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jpy = frameLowerJacs[t - 1].selectedJointsJacobian[(static_cast<size_t>(row) + 1) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            const float jpz = frameLowerJacs[t - 1].selectedJointsJacobian[(static_cast<size_t>(row) + 2) * static_cast<size_t>(jacDimGlobal) + static_cast<size_t>(dvar)];
                            out.gradient[goffT + static_cast<size_t>(dvar)] += dL_dd[0] * jtx + dL_dd[1] * jty + dL_dd[2] * jtz;
                            out.gradient[goffP + static_cast<size_t>(dvar)] -= dL_dd[0] * jpx + dL_dd[1] * jpy + dL_dd[2] * jpz;
                        }
                    }
                }
            }
        }
    }

    if (cfg.weights.lossPriorW > 0.0f && S > 0) {
        float priorSum = 0.0f;
        for (size_t t = 0; t < S; ++t) {
            if (mask[t] <= 0.0f) {
                continue;
            }
            std::vector<float> initLower(lowerDim, 0.0f);
            const auto& c = data.seqCases[t];
            for (size_t j = 0; j < numLower; ++j) {
                const int jid = cfg.lowerBodyJointIds[j];
                const int aaOff = jid * 3;

                // Validate bounds to prevent out-of-range access
                if (jid < 0 || aaOff + 2 >= static_cast<int>(c.smplxPose165.size())) {
                    throw std::runtime_error("Invalid joint ID " + std::to_string(jid) +
                        " results in out-of-bounds access to smplxPose165 (size: " +
                        std::to_string(c.smplxPose165.size()) + ")");
                }

                float R0[9];
                AxisAngleToRotation3x3(
                    c.smplxPose165[static_cast<size_t>(aaOff) + 0],
                    c.smplxPose165[static_cast<size_t>(aaOff) + 1],
                    c.smplxPose165[static_cast<size_t>(aaOff) + 2],
                    R0);
                float r6d0[6];
                MatrixToRotation6d(R0, r6d0);
                const size_t base = j * 6;
                for (int k = 0; k < 6; ++k) {
                    initLower[base + static_cast<size_t>(k)] = r6d0[k];
                }
            }
            float sq = 0.0f;
            for (size_t d = 0; d < lowerDim; ++d) {
                const float diff = p.lowerBodyR6dSeq[t * lowerDim + d] - initLower[d];
                sq += diff * diff;
            }
            const float n = std::sqrt(std::max(1e-12f, sq));
            priorSum += n * mask[t];

            const float coeff = cfg.weights.lossPriorW * mask[t] / validNormDenom;
            const size_t goff = (idxCamScale + 1) + t * lowerDim;
            for (size_t d = 0; d < lowerDim; ++d) {
                const float diff = p.lowerBodyR6dSeq[t * lowerDim + d] - initLower[d];
                out.gradient[goff + d] += coeff * (diff / n);
            }
        }
        lossPrior = (maskSum > 0.0f) ? (priorSum / validNormDenom) : 0.0f;
    }

    // GMM prior (enabled if weight > 0 and data is available) - ANALYTIC GRADIENT VERSION
    if (cfg.weights.lossGmmPriorW > 0.0f && cfg.gmmNumGaussians > 0 && cfg.gmmDim > 0) {
        const int gm = cfg.gmmNumGaussians;
        const int gd = cfg.gmmDim;
        const int gmmDim = gd;  // Use full 69 dims to match Python
        const float epsJac = 1e-5f;  // For ∂AA/∂r6d finite differences

        float gmmSum = 0.0f;
        for (size_t t = 0; t < S; ++t) {
            if (mask[t] <= 0.0f) {
                continue;
            }
            const SequenceFrameCase& c = data.seqCases[t];
            std::vector<float> pose165 = c.smplxPose165;

            // Replace lower-body AA with current optimized lower-body r6d->AA.
            for (size_t j = 0; j < numLower; ++j) {
                const int jid = cfg.lowerBodyJointIds[j];
                const size_t base = t * lowerDim + j * 6;
                float r6d[6];
                for (int k = 0; k < 6; ++k) {
                    r6d[k] = p.lowerBodyR6dSeq[base + static_cast<size_t>(k)];
                }
                float R[9];
                Rotation6dToMatrix(r6d, R);
                float aa[3];
                RotationMatrixToAxisAngle(R, aa);
                pose165[static_cast<size_t>(jid) * 3 + 0] = aa[0];
                pose165[static_cast<size_t>(jid) * 3 + 1] = aa[1];
                pose165[static_cast<size_t>(jid) * 3 + 2] = aa[2];
            }

            // Extract pose: indices [3:66] from pose165 (21 joints * 3 = 63 elements)
            // Remaining 6 elements stay zero-initialized (matching Python's zero padding)
            std::vector<float> pose69(static_cast<size_t>(gmmDim), 0.0f);
            const int poseExtractCount = 63;  // 21 joints * 3 (body pose only)
            if (pose69.size() < static_cast<size_t>(poseExtractCount))
            {
                CARBON_CRITICAL("Body pose is incorrect size");
            }
            for (int i = 0; i < poseExtractCount; ++i)
            {
                pose69[static_cast<size_t>(i)] = pose165[static_cast<size_t>(3 + i)];
            }

            // Find best Gaussian (minimum NLL).
            int bestM = 0;
            float bestLl = std::numeric_limits<float>::infinity();
            for (int m = 0; m < gm; ++m) {
                const size_t moff = static_cast<size_t>(m) * static_cast<size_t>(gd);
                const size_t poff = static_cast<size_t>(m) * static_cast<size_t>(gd) * static_cast<size_t>(gd);
                std::vector<float> diffGmm(static_cast<size_t>(gmmDim), 0.0f);
                for (int i = 0; i < gmmDim; ++i) {
                    diffGmm[static_cast<size_t>(i)] = pose69[static_cast<size_t>(i)] - cfg.gmmMeans[moff + static_cast<size_t>(i)];
                }
                float quad = 0.0f;
                for (int i = 0; i < gmmDim; ++i) {
                    float accQuad = 0.0f;
                    for (int j = 0; j < gmmDim; ++j) {
                        accQuad += cfg.gmmPrecisions[poff + static_cast<size_t>(i) * static_cast<size_t>(gd) + static_cast<size_t>(j)] * diffGmm[static_cast<size_t>(j)];
                    }
                    quad += diffGmm[static_cast<size_t>(i)] * accQuad;
                }
                // gmmLogNllWeights now stores log(nll_weights) directly for numerical stability
                const float log_w = cfg.gmmLogNllWeights[static_cast<size_t>(m)];
                const float ll = 0.5f * quad - log_w;
                if (ll < bestLl) {
                    bestLl = ll;
                    bestM = m;
                }
            }

            gmmSum += bestLl * mask[t];

            // ANALYTIC GRADIENT: ∂NLL/∂pose69 = Λ * (pose69 - μ)
            const size_t bestMoff = static_cast<size_t>(bestM) * static_cast<size_t>(gd);
            const size_t bestPoff = static_cast<size_t>(bestM) * static_cast<size_t>(gd) * static_cast<size_t>(gd);

            std::vector<float> gradPose69(gmmDim, 0.0f);
            for (int i = 0; i < gmmDim; ++i) {
                float accGrad = 0.0f;
                for (int j = 0; j < gmmDim; ++j) {
                    accGrad += cfg.gmmPrecisions[bestPoff + static_cast<size_t>(i) * static_cast<size_t>(gd) + static_cast<size_t>(j)] *
                           (pose69[static_cast<size_t>(j)] - cfg.gmmMeans[bestMoff + static_cast<size_t>(j)]);
                }
                gradPose69[static_cast<size_t>(i)] = accGrad;
            }

            const size_t goff = (idxCamScale + 1) + t * lowerDim;
            const float coeff = cfg.weights.lossGmmPriorW * mask[t] / validNormDenom;

            // Compute ∂NLL/∂r6d using chain rule: ∂NLL/∂r6d = ∂NLL/∂pose69 * ∂pose69/∂AA * ∂AA/∂r6d
            for (size_t j = 0; j < numLower; ++j) {
                const int jid = cfg.lowerBodyJointIds[j];
                if (jid < 1 || jid > 21) continue;

                const int pose69Start = (jid - 1) * 3;
                if (pose69Start + 2 >= gmmDim) continue;

                // Get current r6d and AA
                const size_t rb = t * lowerDim + j * 6;
                float r6d[6];
                for (int k = 0; k < 6; ++k) {
                    r6d[k] = p.lowerBodyR6dSeq[rb + static_cast<size_t>(k)];
                }

                // Compute ∂AA/∂r6d using finite differences (only 6 dims per joint)
                for (int k = 0; k < 6; ++k) {
                    float r6dP[6], r6dM[6];
                    for (int kk = 0; kk < 6; ++kk) {
                        r6dP[kk] = r6d[kk];
                        r6dM[kk] = r6d[kk];
                    }
                    r6dP[k] += epsJac;
                    r6dM[k] -= epsJac;

                    float Rp[9], Rm[9], aap[3], aam[3];
                    Rotation6dToMatrix(r6dP, Rp);
                    Rotation6dToMatrix(r6dM, Rm);
                    RotationMatrixToAxisAngle(Rp, aap);
                    RotationMatrixToAxisAngle(Rm, aam);

                    // Chain rule: ∂NLL/∂r6d[k] = Σ_i (∂NLL/∂pose69[pose69Start+i] * ∂AA[i]/∂r6d[k])
                    float grad = 0.0f;
                    for (int i = 0; i < 3; ++i) {
                        const int pose69Idx = pose69Start + i;
                        const float daaDr6d = (aap[i] - aam[i]) / (2.0f * epsJac);
                        grad += gradPose69[static_cast<size_t>(pose69Idx)] * daaDr6d;
                    }
                    out.gradient[goff + j * 6 + static_cast<size_t>(k)] += coeff * grad;
                }
            }
        }
        lossGmmPrior = (maskSum > 0.0f) ? (gmmSum / validNormDenom) : 0.0f;
    }

    if (accDen > 0.0f) {
        const float c = smoothW * 10.0f / accDen;
        const float scale = 2.0f * cfg.fps * cfg.fps;
        for (size_t t = 1; t + 1 < S; ++t) {
            if (mask[t] <= 0.0f) {
                continue;
            }
            const float ax = translTotal[t + 1][0] + translTotal[t - 1][0] - 2.0f * translTotal[t][0];
            const float ay = translTotal[t + 1][1] + translTotal[t - 1][1] - 2.0f * translTotal[t][1];
            const float az = translTotal[t + 1][2] + translTotal[t - 1][2] - 2.0f * translTotal[t][2];
            const float k = c * mask[t] * scale;
            out.gradient[(t + 1) * 3 + 0] += k * ax;
            out.gradient[(t + 1) * 3 + 1] += k * ay;
            out.gradient[(t + 1) * 3 + 2] += k * az;
            out.gradient[(t - 1) * 3 + 0] += k * ax;
            out.gradient[(t - 1) * 3 + 1] += k * ay;
            out.gradient[(t - 1) * 3 + 2] += k * az;
            out.gradient[t * 3 + 0] -= 2.0f * k * ax;
            out.gradient[t * 3 + 1] -= 2.0f * k * ay;
            out.gradient[t * 3 + 2] -= 2.0f * k * az;
        }
    }

    out.objective = cfg.weights.lossKpPosW * lossKpPos
        + cfg.weights.lossKpEdgeW * lossKpEdge
        + cfg.weights.lossSmoothW * lossSmooth
        + cfg.weights.lossSmoothJ3dW * lossSmoothJ3D
        + cfg.weights.lossPriorW * lossPrior
        + cfg.weights.lossGmmPriorW * lossGmmPrior
        + cfg.weights.lossContactVelW * lossContactVel
        + cfg.weights.lossContactHeightW * lossContactHeight
        + cfg.weights.lossBelowFloorW * lossBelowFloor;

    // Zero out gradients for constant (non-optimized) parameters
    if (!cfg.optVars.empty()) {
        // Use already-declared variables from earlier in the function
        // Check which variables should NOT be optimized and zero their gradients
        if (!ShouldOptimizeVar(cfg, "transl")) {
            for (size_t i = 0; i < translDim; ++i) {
                out.gradient[i] = 0.0f;
            }
        }
        if (!ShouldOptimizeVar(cfg, "cam_height_offset")) {
            out.gradient[idxCamH] = 0.0f;
        }
        if (!ShouldOptimizeVar(cfg, "cam_init_r6d")) {
            for (size_t i = 0; i < 6; ++i) {
                out.gradient[idxCamR6D + i] = 0.0f;
            }
        }
        if (!ShouldOptimizeVar(cfg, "cam_scale")) {
            out.gradient[idxCamScale] = 0.0f;
        }
        if (!ShouldOptimizeVar(cfg, "smplx_pose_r6d_lower_body") &&
            !ShouldOptimizeVar(cfg, "lower_body_r6d_seq")) {
            const size_t lowerStart = idxCamScale + 1;
            for (size_t i = 0; i < S * lowerDim; ++i) {
                out.gradient[lowerStart + i] = 0.0f;
            }
        }
    }

    return out;
}

float EvaluateStage1Objective(
    const SmplxForwardLite& smplx,
    const Stage1Config& cfg,
    const Stage1Data& data,
    const std::vector<float>& x,
    std::shared_ptr<TITAN_NAMESPACE::TaskThreadPool> threadPool) {
    return EvaluateStage1ObjectiveAndGradient(smplx, cfg, data, x, threadPool).objective;
}

}

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
