// Copyright Epic Games, Inc. All Rights Reserved.

#include "bodypostprocessing/SmplxForwardLite.h"

#include <carbon/common/Log.h>
#include <carbon/common/Defs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <Eigen/Sparse>
#include <unsupported/Eigen/AutoDiff>

namespace fs = std::filesystem;

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace {

constexpr int kLowerBodyAdDim = 36;

template <typename T>
std::vector<T> ReadBinaryVector(const fs::path& path)
{
    if (!fs::exists(path))
    {
        throw std::runtime_error("Missing binary file: " + path.string());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open binary file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamsize bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes < 0 || (bytes % sizeof(T)) != 0)
    {
        throw std::runtime_error("Invalid binary size for: " + path.string());
    }
    std::vector<T> out(static_cast<size_t>(bytes / sizeof(T)));
    if (!out.empty())
    {
        in.read(reinterpret_cast<char*>(out.data()), bytes);

        // Verify the read was successful
        if (!in.good())
        {
            throw std::runtime_error("Failed to read binary file: " + path.string());
        }
        if (in.gcount() != bytes)
        {
            throw std::runtime_error("Incomplete read from binary file: " + path.string() + " (expected " + std::to_string(bytes) + " bytes, got " + std::to_string(in.gcount()) + ")");
        }
    }
    return out;
}

inline void SetIdentity4(float m[16]) {
    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

inline void Mul4x4(const float a[16], const float b[16], float out[16]) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = 0.0f;
            for (int k = 0; k < 4; ++k) {
                v += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = v;
        }
    }
}

inline void Mul4x4Vec4(const float m[16], const float v[4], float out[4]) {
    for (int r = 0; r < 4; ++r) {
        out[r] = m[r * 4 + 0] * v[0] + m[r * 4 + 1] * v[1] + m[r * 4 + 2] * v[2] + m[r * 4 + 3] * v[3];
    }
}

inline void AxisAngleToRotation3x3(float x, float y, float z, float r[9]) {
    const float theta = std::sqrt(x * x + y * y + z * z);
    if (theta <= 1e-4f) {
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

}  // namespace


namespace bodyopt
{

bool SmplxForwardLite::LoadStaticFromFolder(const std::string& folder) {
    const fs::path root(folder);
    m_vTemplate = ReadBinaryVector<float>(root / "v_template_f32.bin");
    m_weights = ReadBinaryVector<float>(root / "weights_f32.bin");
    m_jointReg = ReadBinaryVector<float>(root / "joint_regressor_f32.bin");
    m_shapeBlend = ReadBinaryVector<float>(root / "shape_blend_beta_f32.bin");
    m_poseBlend = ReadBinaryVector<float>(root / "pose_blend_f32.bin");
    const std::vector<int32_t> parentsI32 = ReadBinaryVector<int32_t>(root / "kintree_parents_i32.bin");
    const std::vector<int32_t> extraI32 = ReadBinaryVector<int32_t>(root / "extra_joints_idxs_i32.bin");
    const std::vector<int32_t> lmkFaceI32 = ReadBinaryVector<int32_t>(root / "lmk_faces_idx_i32.bin");
    const std::vector<int32_t> facesI32 = ReadBinaryVector<int32_t>(root / "faces_i32.bin");
    m_lmkBaryCoords = ReadBinaryVector<float>(root / "lmk_bary_coords_f32.bin");
    m_kintreeParents.assign(parentsI32.begin(), parentsI32.end());
    m_extraJointsIdxs.assign(extraI32.begin(), extraI32.end());
    m_lmkFacesIdx.assign(lmkFaceI32.begin(), lmkFaceI32.end());
    m_facesTri.assign(facesI32.begin(), facesI32.end());

    if (m_kintreeParents.empty()) {
        throw std::runtime_error("kintreeParents is empty");
    }
    m_numJoints = static_cast<int>(m_kintreeParents.size());

    if ((m_vTemplate.size() % 3) != 0) {
        throw std::runtime_error("vTemplate size is not divisible by 3");
    }
    m_numVerts = static_cast<int>(m_vTemplate.size() / 3);

    if (m_weights.size() != static_cast<size_t>(m_numVerts) * static_cast<size_t>(m_numJoints))
    {
        throw std::runtime_error("weights shape mismatch");
    }
    if (m_jointReg.size() != static_cast<size_t>(m_numJoints) * static_cast<size_t>(m_numVerts))
    {
        throw std::runtime_error("jointRegressor shape mismatch");
    }
    if (m_shapeBlend.size() % (static_cast<size_t>(m_numVerts) * 3) != 0) {
        throw std::runtime_error("shapeBlend shape mismatch");
    }
    m_numBetas = static_cast<int>(m_shapeBlend.size() / (static_cast<size_t>(m_numVerts) * 3));
    if (m_numBetas <= 0) {
        throw std::runtime_error("shapeBlend has invalid beta dimension");
    }

    if (m_poseBlend.size() % (static_cast<size_t>(m_numVerts) * 3) != 0) {
        throw std::runtime_error("poseBlend shape mismatch");
    }
    m_numPoseBasis = static_cast<int>(m_poseBlend.size() / (static_cast<size_t>(m_numVerts) * 3));
    const int expectedPoseBasis = 9 * (m_numJoints - 1);
    if (m_numPoseBasis != expectedPoseBasis) {
        throw std::runtime_error("poseBlend basis mismatch with numJoints");
    }
    if (m_extraJointsIdxs.size() != 21) {
        throw std::runtime_error("extraJointsIdxs expected size 21");
    }
    if (m_lmkFacesIdx.size() != 51 || m_lmkBaryCoords.size() != 51 * 3) {
        throw std::runtime_error("landmark data shape mismatch");
    }
    if (m_facesTri.empty() || (m_facesTri.size() % 3) != 0) {
        throw std::runtime_error("faces data shape mismatch");
    }

    return true;
}

bool SmplxForwardLite::LoadFromArrays(
    const std::vector<float>& verts,
    const std::vector<uint32_t>& faces,
    const std::vector<float>& jointReg,
    const std::vector<float>& weights,
    const std::vector<float>& blendShapes,
    const std::vector<int>& kintreeParents,
    const std::vector<int>& extraJointsIdxs,
    const std::vector<int>& lmkFacesIdx,
    const std::vector<float>& lmkBaryCoords) {

    // Basic validation - determine dimensions from kintreeParents
    if (kintreeParents.empty()) {
        throw std::runtime_error("kintreeParents cannot be empty");
    }
    m_numJoints = static_cast<int>(kintreeParents.size());
    m_kintreeParents = kintreeParents;

    // Validate and convert verts from column-major to row-major
    // UE stores as column-major (NumVerts, 3): [x0,x1,...,xN, y0,y1,...,yN, z0,z1,...,zN]
    // We need row-major: [x0,y0,z0, x1,y1,z1, ...]
    if ((verts.size() % 3) != 0) {
        throw std::runtime_error("verts size must be divisible by 3");
    }
    m_numVerts = static_cast<int>(verts.size() / 3);
    if (m_numVerts <= 0) {
        throw std::runtime_error("numVerts must be positive");
    }
    m_vTemplate.resize(verts.size());
    for (int v = 0; v < m_numVerts; ++v) {
        for (int c = 0; c < 3; ++c) {
            // Column-major: element (v, c) is at index v + c * numVerts
            const size_t srcIdx = static_cast<size_t>(v) + static_cast<size_t>(c) * static_cast<size_t>(m_numVerts);
            // Row-major: element (v, c) is at index v * 3 + c
            const size_t dstIdx = static_cast<size_t>(v) * 3 + static_cast<size_t>(c);
            m_vTemplate[dstIdx] = verts[srcIdx];
        }
    }

    // Convert weights from column-major to row-major
    // UE stores as column-major (NumVerts, NumJoints)
    // We need row-major [V, J] accessed as weights[v * numJoints + j]
    const size_t expectedWeightsSize = static_cast<size_t>(m_numVerts) * static_cast<size_t>(m_numJoints);
    if (weights.size() != expectedWeightsSize) {
        throw std::runtime_error("weights size mismatch: expected " +
            std::to_string(expectedWeightsSize) + " got " + std::to_string(weights.size()));
    }
    m_weights.resize(weights.size());
    for (int v = 0; v < m_numVerts; ++v) {
        for (int j = 0; j < m_numJoints; ++j) {
            // Column-major (NumVerts, NumJoints): element (v, j) at index v + j * numVerts
            const size_t srcIdx = static_cast<size_t>(v) + static_cast<size_t>(j) * static_cast<size_t>(m_numVerts);
            // Row-major [V, J]: element (v, j) at index v * numJoints + j
            const size_t dstIdx = static_cast<size_t>(v) * static_cast<size_t>(m_numJoints) + static_cast<size_t>(j);
            m_weights[dstIdx] = weights[srcIdx];
        }
    }

    // Convert jointReg from column-major to row-major
    // UE stores as column-major (NumJoints, NumVerts)
    // We need row-major [J, V] accessed as jointReg[j * numVerts + v]
    const size_t expectedJointRegSize = static_cast<size_t>(m_numJoints) * static_cast<size_t>(m_numVerts);
    if (jointReg.size() != expectedJointRegSize) {
        throw std::runtime_error("jointReg size mismatch: expected " +
            std::to_string(expectedJointRegSize) + " got " + std::to_string(jointReg.size()));
    }
    m_jointReg.resize(jointReg.size());
    for (int j = 0; j < m_numJoints; ++j) {
        for (int v = 0; v < m_numVerts; ++v) {
            // Column-major (NumJoints, NumVerts): element (j, v) at index j + v * numJoints
            const size_t srcIdx = static_cast<size_t>(j) + static_cast<size_t>(v) * static_cast<size_t>(m_numJoints);
            // Row-major [J, V]: element (j, v) at index j * numVerts + v
            const size_t dstIdx = static_cast<size_t>(j) * static_cast<size_t>(m_numVerts) + static_cast<size_t>(v);
            m_jointReg[dstIdx] = jointReg[srcIdx];
        }
    }

    // Process blendShapes: UE stores as column-major [NumVerts*3, NumShapes + NumPoseBasis]
    // We need to split into shapeBlend and poseBlend, converting to our custom indexing
    m_numPoseBasis = 9 * (m_numJoints - 1);
    const int numRows = m_numVerts * 3;

    // Determine numBetas from blendShapes size
    if ((blendShapes.size() % static_cast<size_t>(numRows)) != 0) {
        throw std::runtime_error("blendShapes size not divisible by (NumVerts * 3)");
    }
    const int totalCols = static_cast<int>(blendShapes.size() / static_cast<size_t>(numRows));

    if (totalCols < m_numPoseBasis) {
        throw std::runtime_error("blendShapes has too few columns for pose basis");
    }
    m_numBetas = totalCols - m_numPoseBasis;

    if (m_numBetas <= 0) {
        throw std::runtime_error("Invalid number of shape blend shapes");
    }

    // Convert shape blendshapes from column-major to custom indexing
    // UE format (column-major): element (row, col) at index = row + col * numRows
    // Our format: element (v, c, b) at index = (v * 3 + c) * numBetas + b
    m_shapeBlend.resize(static_cast<size_t>(numRows) * static_cast<size_t>(m_numBetas));
    for (int v = 0; v < m_numVerts; ++v) {
        for (int c = 0; c < 3; ++c) {
            const int row = v * 3 + c;
            for (int b = 0; b < m_numBetas; ++b) {
                const size_t srcIdx = static_cast<size_t>(row) + static_cast<size_t>(b) * static_cast<size_t>(numRows);
                const size_t dstIdx = static_cast<size_t>(row) * static_cast<size_t>(m_numBetas) + static_cast<size_t>(b);
                m_shapeBlend[dstIdx] = blendShapes[srcIdx];
            }
        }
    }

    // Convert pose blendshapes from column-major to custom indexing
    m_poseBlend.resize(static_cast<size_t>(numRows) * static_cast<size_t>(m_numPoseBasis));
    for (int v = 0; v < m_numVerts; ++v) {
        for (int c = 0; c < 3; ++c) {
            const int row = v * 3 + c;
            for (int p = 0; p < m_numPoseBasis; ++p) {
                const int col = m_numBetas + p;  // Pose basis starts after shape basis
                const size_t srcIdx = static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(numRows);
                const size_t dstIdx = static_cast<size_t>(row) * static_cast<size_t>(m_numPoseBasis) + static_cast<size_t>(p);
                m_poseBlend[dstIdx] = blendShapes[srcIdx];
            }
        }
    }

    // Validate faces
    if (faces.empty() || (faces.size() % 3) != 0) {
        throw std::runtime_error("faces must be non-empty and divisible by 3");
    }
    m_facesTri.assign(faces.begin(), faces.end());

    // Validate and copy extraJointsIdxs
    if (extraJointsIdxs.size() != 21) {
        throw std::runtime_error("extraJointsIdxs must have exactly 21 elements, got " +
            std::to_string(extraJointsIdxs.size()));
    }
    m_extraJointsIdxs = extraJointsIdxs;

    // Validate and copy landmark data
    if (lmkFacesIdx.size() != 51) {
        throw std::runtime_error("lmkFacesIdx must have exactly 51 elements, got " +
            std::to_string(lmkFacesIdx.size()));
    }
    if (lmkBaryCoords.size() != 51 * 3) {
        throw std::runtime_error("lmkBaryCoords must have exactly 153 elements (51*3), got " +
            std::to_string(lmkBaryCoords.size()));
    }
    m_lmkFacesIdx = lmkFacesIdx;
    m_lmkBaryCoords = lmkBaryCoords;

    return true;
}

void SmplxForwardLite::SetSmplxEvaluationConstants(
    const std::vector<float>& betas10,
    const std::vector<int>& lowerJointIds,
    const std::vector<int>& selectedJointIds,
    const std::vector<int>& selectedVertexIds)
{
    if (m_numVerts <= 0 || m_numJoints <= 0) {
        throw std::runtime_error("SmplxForwardLite: must call LoadStaticFromFolder or LoadFromArrays before SetSmplxEvaluationConstants");
    }
    if (betas10.size() < static_cast<size_t>(m_numBetas)) {
        throw std::runtime_error("SmplxForwardLite: betas10 size mismatch");
    }

    // Store constants
    m_evalBetas.assign(betas10.begin(), betas10.begin() + m_numBetas);

    // Validate and store lowerJointIds
    for (size_t i = 0; i < lowerJointIds.size(); ++i) {
        const int jid = lowerJointIds[i];
        if (jid < 0 || jid >= m_numJoints) {
            throw std::runtime_error("SmplxForwardLite: lower joint id " + std::to_string(jid) +
                                   " out of range [0, " + std::to_string(m_numJoints) + ")");
        }
    }
    m_evalLowerJointIds = lowerJointIds;

    // Validate and store selectedJointIds
    const int numJointsAll = m_numJoints + static_cast<int>(m_extraJointsIdxs.size()) + static_cast<int>(m_lmkFacesIdx.size());
    if (selectedJointIds.empty()) {
        // Handle empty selectedJointIds: empty means all joints (per documentation)
        m_evalSelectedJointIds.resize(numJointsAll);
        for (int i = 0; i < numJointsAll; ++i) {
            m_evalSelectedJointIds[i] = i;
        }
    } else {
        // Validate all selected joint IDs are in valid range
        for (size_t i = 0; i < selectedJointIds.size(); ++i) {
            const int sid = selectedJointIds[i];
            if (sid < 0 || sid >= numJointsAll) {
                throw std::runtime_error("SmplxForwardLite: selected joint id " + std::to_string(sid) +
                                       " out of range [0, " + std::to_string(numJointsAll) + ")");
            }
        }
        m_evalSelectedJointIds = selectedJointIds;
    }

    // Validate and store selectedVertexIds
    for (size_t i = 0; i < selectedVertexIds.size(); ++i) {
        const int vid = selectedVertexIds[i];
        if (vid < 0 || vid >= m_numVerts) {
            throw std::runtime_error("SmplxForwardLite: selected vertex id " + std::to_string(vid) +
                                   " out of range [0, " + std::to_string(m_numVerts) + ")");
        }
    }
    m_evalSelectedVertexIds = selectedVertexIds;

    // Build lower joint mapping
    m_evalLowerJointToSlot.clear();
    m_evalLowerJointToSlot.reserve(m_evalLowerJointIds.size());
    for (size_t i = 0; i < m_evalLowerJointIds.size(); ++i) {
        const int jid = m_evalLowerJointIds[i];
        m_evalLowerJointToSlot[jid] = static_cast<int>(i);
    }

    // Pre-compute shaped vertices and rest joints from betas
    BuildShapedAndJointsRest();

    // Pre-compute dependency data for selected vertices/joints
    if (!m_evalSelectedVertexIds.empty() || !m_evalSelectedJointIds.empty()) {
        BuildDependencyData();
        BuildSparseMatrices();
    }

    m_evalConstantsSet = true;
}

void SmplxForwardLite::BuildShapedAndJointsRest()
{
    // Build shaped vertices from template + beta blend shapes
    m_evalShaped.assign(static_cast<size_t>(m_numVerts) * 3, 0.0f);
    for (int v = 0; v < m_numVerts; ++v)
    {
        for (int c = 0; c < 3; ++c)
        {
            float val = m_vTemplate[v * 3 + c];
            for (int b = 0; b < m_numBetas; ++b)
            {
                const size_t idx = (static_cast<size_t>(v) * 3 + static_cast<size_t>(c)) * static_cast<size_t>(m_numBetas) + static_cast<size_t>(b);
                val += m_shapeBlend[idx] * m_evalBetas[static_cast<size_t>(b)];
            }
            m_evalShaped[static_cast<size_t>(v) * 3 + static_cast<size_t>(c)] = val;
        }
    }

    // Build rest joints from shaped vertices
    m_evalJointsRest.assign(static_cast<size_t>(m_numJoints) * 3, 0.0f);
    for (int j = 0; j < m_numJoints; ++j)
    {
        for (int c = 0; c < 3; ++c)
        {
            float acc = 0.0f;
            for (int v = 0; v < m_numVerts; ++v)
            {
                const float r = m_jointReg[j * m_numVerts + v];
                acc += r * m_evalShaped[static_cast<size_t>(v) * 3 + static_cast<size_t>(c)];
            }
            m_evalJointsRest[static_cast<size_t>(j) * 3 + static_cast<size_t>(c)] = acc;
        }
    }
}

void SmplxForwardLite::BuildDependencyData()
{
    m_evalRequiredVertices.clear();
    m_evalVidToLocal.clear();

    const int baseExtra = m_numJoints;
    const int baseLmk = baseExtra + static_cast<int>(m_extraJointsIdxs.size());
    std::unordered_set<int> requiredVertexSet;
    requiredVertexSet.reserve(m_evalSelectedJointIds.size() * 3);

    // Process selected joint IDs to find required vertices
    for (int sid : m_evalSelectedJointIds)
    {
        if (sid >= baseExtra && sid < baseLmk)
        {
            const size_t extraIdx = static_cast<size_t>(sid - baseExtra);
            if (extraIdx >= m_extraJointsIdxs.size())
            {
                CARBON_CRITICAL("BuildDependencyData: extra joint index {} out of range (size: {})",
                               extraIdx, m_extraJointsIdxs.size());
            }
            requiredVertexSet.insert(m_extraJointsIdxs[extraIdx]);
        }
        else if (sid >= baseLmk)
        {
            const int lmk = sid - baseLmk;
            if (lmk < 0 || static_cast<size_t>(lmk) >= m_lmkFacesIdx.size())
            {
                CARBON_CRITICAL("BuildDependencyData: landmark index {} out of range (size: {})",
                               lmk, m_lmkFacesIdx.size());
            }
            const int faceId = m_lmkFacesIdx[static_cast<size_t>(lmk)];
            if (faceId < 0 || static_cast<size_t>(faceId) * 3 + 2 >= m_facesTri.size())
            {
                CARBON_CRITICAL("BuildDependencyData: face index {} out of range (facesTri size: {})",
                               faceId, m_facesTri.size());
            }
            const int v0 = m_facesTri[static_cast<size_t>(faceId) * 3 + 0];
            const int v1 = m_facesTri[static_cast<size_t>(faceId) * 3 + 1];
            const int v2 = m_facesTri[static_cast<size_t>(faceId) * 3 + 2];
            requiredVertexSet.insert(v0);
            requiredVertexSet.insert(v1);
            requiredVertexSet.insert(v2);
        }
    }

    // Process selected vertex IDs
    for (int vid : m_evalSelectedVertexIds)
    {
        if (vid >= 0 && vid < m_numVerts)
        {
            requiredVertexSet.insert(vid);
        }
    }

    // Build sorted list and mapping
    m_evalRequiredVertices.assign(requiredVertexSet.begin(), requiredVertexSet.end());
    std::sort(m_evalRequiredVertices.begin(), m_evalRequiredVertices.end());
    m_evalVidToLocal.reserve(m_evalRequiredVertices.size());
    for (size_t i = 0; i < m_evalRequiredVertices.size(); ++i)
    {
        m_evalVidToLocal[m_evalRequiredVertices[i]] = static_cast<int>(i);
    }
}

void SmplxForwardLite::BuildSparseMatrices()
{
    // Build sparse pose blend matrix
    constexpr float nzEps = 1e-9f;
    {
        std::vector<Eigen::Triplet<float>> trips;
        trips.reserve(m_poseBlend.size() / 8);
        for (int row = 0; row < m_numVerts * 3; ++row) {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(m_numPoseBasis);
            for (int col = 0; col < m_numPoseBasis; ++col) {
                const float v = m_poseBlend[base + static_cast<size_t>(col)];
                if (std::abs(v) > nzEps) {
                    trips.emplace_back(row, col, v);
                }
            }
        }
        m_sparsePoseBlend.resize(m_numVerts * 3, m_numPoseBasis);
        m_sparsePoseBlend.setFromTriplets(trips.begin(), trips.end());
    }
    // Build sparse weights matrix
    {
        std::vector<Eigen::Triplet<float>> trips;
        trips.reserve(m_weights.size() / 8);
        for (int row = 0; row < m_numVerts; ++row) {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(m_numJoints);
            for (int col = 0; col < m_numJoints; ++col) {
                const float v = m_weights[base + static_cast<size_t>(col)];
                if (std::abs(v) > nzEps) {
                    trips.emplace_back(row, col, v);
                }
            }
        }
        m_sparseWeights.resize(m_numVerts, m_numJoints);
        m_sparseWeights.setFromTriplets(trips.begin(), trips.end());
    }
    // Build sparse joint regressor matrix
    {
        std::vector<Eigen::Triplet<float>> trips;
        trips.reserve(m_jointReg.size() / 8);
        for (int row = 0; row < m_numJoints; ++row) {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(m_numVerts);
            for (int col = 0; col < m_numVerts; ++col) {
                const float v = m_jointReg[base + static_cast<size_t>(col)];
                if (std::abs(v) > nzEps) {
                    trips.emplace_back(row, col, v);
                }
            }
        }
        m_sparseJointReg.resize(m_numJoints, m_numVerts);
        m_sparseJointReg.setFromTriplets(trips.begin(), trips.end());
    }
}


SmplxForwardOutput SmplxForwardLite::Forward(const SmplxForwardInput& in) const {
    if (!m_evalConstantsSet) {
        throw std::runtime_error("SmplxForwardLite: must call SetSmplxEvaluationConstants before Forward");
    }
    if (in.pose165.size() != static_cast<size_t>(m_numJoints) * 3) {
        throw std::runtime_error("pose165 size mismatch");
    }

    // Use pre-computed shaped and jointsRest from SetSmplxEvaluationConstants
    const float* shaped = m_evalShaped.data();
    const float* jointsRest = m_evalJointsRest.data();

    std::vector<float> poseBasis(static_cast<size_t>(m_numPoseBasis), 0.0f);
    int poseRow = 0;
    std::vector<std::array<float, 9>> rotmats(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        const float ax = in.pose165[j * 3 + 0];
        const float ay = in.pose165[j * 3 + 1];
        const float az = in.pose165[j * 3 + 2];
        AxisAngleToRotation3x3(ax, ay, az, rotmats[j].data());

        if (j > 0) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    const float identity = (r == c) ? 1.0f : 0.0f;
                    poseBasis[poseRow++] = rotmats[j][r * 3 + c] - identity;
                }
            }
        }
    }

    std::vector<float> poseDelta(static_cast<size_t>(m_numVerts) * 3, 0.0f);
    for (int vc = 0; vc < m_numVerts * 3; ++vc) {
        float acc = 0.0f;
        const size_t base = static_cast<size_t>(vc) * m_numPoseBasis;
        for (int k = 0; k < m_numPoseBasis; ++k) {
            acc += m_poseBlend[base + static_cast<size_t>(k)] * poseBasis[static_cast<size_t>(k)];
        }
        poseDelta[static_cast<size_t>(vc)] = acc;
    }

    std::vector<float> posed(static_cast<size_t>(m_numVerts) * 3, 0.0f);
    for (int i = 0; i < m_numVerts * 3; ++i) {
        posed[static_cast<size_t>(i)] = shaped[static_cast<size_t>(i)] + poseDelta[static_cast<size_t>(i)];
    }

    std::vector<std::array<float, 16>> localTf(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        float t[16];
        SetIdentity4(t);
        const auto& r = rotmats[static_cast<size_t>(j)];
        t[0] = r[0]; t[1] = r[1]; t[2] = r[2];
        t[4] = r[3]; t[5] = r[4]; t[6] = r[5];
        t[8] = r[6]; t[9] = r[7]; t[10] = r[8];
        const int p = m_kintreeParents[static_cast<size_t>(j)];
        if (p < 0) {
            t[3] = jointsRest[j * 3 + 0];
            t[7] = jointsRest[j * 3 + 1];
            t[11] = jointsRest[j * 3 + 2];
        } else {
            t[3] = jointsRest[j * 3 + 0] - jointsRest[p * 3 + 0];
            t[7] = jointsRest[j * 3 + 1] - jointsRest[p * 3 + 1];
            t[11] = jointsRest[j * 3 + 2] - jointsRest[p * 3 + 2];
        }
        for (int i = 0; i < 16; ++i) localTf[static_cast<size_t>(j)][static_cast<size_t>(i)] = t[i];
    }

    std::vector<std::array<float, 16>> worldTf(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        const int p = m_kintreeParents[static_cast<size_t>(j)];
        if (p < 0) {
            worldTf[static_cast<size_t>(j)] = localTf[static_cast<size_t>(j)];
        } else {
            float out[16];
            Mul4x4(worldTf[static_cast<size_t>(p)].data(), localTf[static_cast<size_t>(j)].data(), out);
            for (int i = 0; i < 16; ++i) worldTf[static_cast<size_t>(j)][static_cast<size_t>(i)] = out[i];
        }
    }

    std::vector<std::array<float, 16>> restInv(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        float m[16];
        SetIdentity4(m);
        m[3] = -jointsRest[j * 3 + 0];
        m[7] = -jointsRest[j * 3 + 1];
        m[11] = -jointsRest[j * 3 + 2];
        for (int i = 0; i < 16; ++i) restInv[static_cast<size_t>(j)][static_cast<size_t>(i)] = m[i];
    }

    std::vector<std::array<float, 16>> skinning(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        float out[16];
        Mul4x4(worldTf[static_cast<size_t>(j)].data(), restInv[static_cast<size_t>(j)].data(), out);
        for (int i = 0; i < 16; ++i) skinning[static_cast<size_t>(j)][static_cast<size_t>(i)] = out[i];
    }

    SmplxForwardOutput out;
    out.numVerts = m_numVerts;
    out.numJoints = m_numJoints;
    out.numJointsAll = m_numJoints + static_cast<int>(m_extraJointsIdxs.size()) + static_cast<int>(m_lmkFacesIdx.size());
    out.jointsXyz.resize(static_cast<size_t>(m_numJoints) * 3, 0.0f);
    out.jointsAllXyz.resize(static_cast<size_t>(out.numJointsAll) * 3, 0.0f);
    out.skinnedVertsXyz.resize(static_cast<size_t>(m_numVerts) * 3, 0.0f);

    for (int j = 0; j < m_numJoints; ++j) {
        out.jointsXyz[j * 3 + 0] = worldTf[static_cast<size_t>(j)][3];
        out.jointsXyz[j * 3 + 1] = worldTf[static_cast<size_t>(j)][7];
        out.jointsXyz[j * 3 + 2] = worldTf[static_cast<size_t>(j)][11];
    }

    for (int v = 0; v < m_numVerts; ++v) {
        const float x4[4] = { posed[v * 3 + 0], posed[v * 3 + 1], posed[v * 3 + 2], 1.0f };
        float final4[4] = { 0, 0, 0, 0 };
        for (int j = 0; j < m_numJoints; ++j) {
            const float w = m_weights[v * m_numJoints + j];
            if (w <= 0.0f) {
                continue;
            }
            float tmp[4];
            Mul4x4Vec4(skinning[static_cast<size_t>(j)].data(), x4, tmp);
            final4[0] += w * tmp[0];
            final4[1] += w * tmp[1];
            final4[2] += w * tmp[2];
        }
        out.skinnedVertsXyz[v * 3 + 0] = final4[0];
        out.skinnedVertsXyz[v * 3 + 1] = final4[1];
        out.skinnedVertsXyz[v * 3 + 2] = final4[2];
    }

    // Build jointsAll = [55 kinematic + 21 extra vertex joints + 51 face landmarks]
    // base joints
    for (int j = 0; j < m_numJoints; ++j) {
        out.jointsAllXyz[j * 3 + 0] = out.jointsXyz[j * 3 + 0];
        out.jointsAllXyz[j * 3 + 1] = out.jointsXyz[j * 3 + 1];
        out.jointsAllXyz[j * 3 + 2] = out.jointsXyz[j * 3 + 2];
    }
    // extra joints from skinned verts
    const int baseExtra = m_numJoints;
    for (size_t e = 0; e < m_extraJointsIdxs.size(); ++e) {
        const int vid = m_extraJointsIdxs[e];
        if (vid < 0 || vid >= m_numVerts) {
            throw std::runtime_error("extra joint vertex index out of range");
        }
        out.jointsAllXyz[(baseExtra + static_cast<int>(e)) * 3 + 0] = out.skinnedVertsXyz[vid * 3 + 0];
        out.jointsAllXyz[(baseExtra + static_cast<int>(e)) * 3 + 1] = out.skinnedVertsXyz[vid * 3 + 1];
        out.jointsAllXyz[(baseExtra + static_cast<int>(e)) * 3 + 2] = out.skinnedVertsXyz[vid * 3 + 2];
    }
    // landmarks via barycentric interpolation over mesh faces
    const int baseLmk = baseExtra + static_cast<int>(m_extraJointsIdxs.size());
    for (size_t l = 0; l < m_lmkFacesIdx.size(); ++l) {
        const int faceId = m_lmkFacesIdx[l];
        const int numFaces = static_cast<int>(m_facesTri.size() / 3);
        if (faceId < 0 || faceId >= numFaces) {
            throw std::runtime_error("landmark face index out of range");
        }
        const int v0 = m_facesTri[faceId * 3 + 0];
        const int v1 = m_facesTri[faceId * 3 + 1];
        const int v2 = m_facesTri[faceId * 3 + 2];
        if (v0 < 0 || v0 >= m_numVerts || v1 < 0 || v1 >= m_numVerts || v2 < 0 || v2 >= m_numVerts) {
            throw std::runtime_error("landmark face vertex index out of range");
        }
        const float bx = m_lmkBaryCoords[l * 3 + 0];
        const float by = m_lmkBaryCoords[l * 3 + 1];
        const float bz = m_lmkBaryCoords[l * 3 + 2];
        out.jointsAllXyz[(baseLmk + static_cast<int>(l)) * 3 + 0] =
            bx * out.skinnedVertsXyz[v0 * 3 + 0] + by * out.skinnedVertsXyz[v1 * 3 + 0] + bz * out.skinnedVertsXyz[v2 * 3 + 0];
        out.jointsAllXyz[(baseLmk + static_cast<int>(l)) * 3 + 1] =
            bx * out.skinnedVertsXyz[v0 * 3 + 1] + by * out.skinnedVertsXyz[v1 * 3 + 1] + bz * out.skinnedVertsXyz[v2 * 3 + 1];
        out.jointsAllXyz[(baseLmk + static_cast<int>(l)) * 3 + 2] =
            bx * out.skinnedVertsXyz[v0 * 3 + 2] + by * out.skinnedVertsXyz[v1 * 3 + 2] + bz * out.skinnedVertsXyz[v2 * 3 + 2];
    }

    return out;
}

SmplxForwardLowerBodyJacobianOutput SmplxForwardLite::ForwardWithLowerBodyR6dJacobian(
    const SmplxForwardInput& in,
    const std::vector<float>& lowerR6dFlat) const {
    if (!m_evalConstantsSet) {
        throw std::runtime_error("SmplxForwardLite: must call SetSmplxEvaluationConstants before ForwardWithLowerBodyR6dJacobian");
    }
    if (in.pose165.size() != static_cast<size_t>(m_numJoints) * 3) {
        throw std::runtime_error("pose165 size mismatch");
    }
    if (lowerR6dFlat.size() != m_evalLowerJointIds.size() * 6ULL) {
        throw std::runtime_error("lowerR6dFlat size mismatch");
    }

    using Deriv = Eigen::Matrix<float, kLowerBodyAdDim, 1>;
    using AD = Eigen::AutoDiffScalar<Deriv>;
    const int lowerDim = static_cast<int>(lowerR6dFlat.size());
    if (lowerDim != kLowerBodyAdDim) {
        throw std::runtime_error("ForwardWithLowerBodyR6dJacobian expects lowerDim=36");
    }

    auto makeConst = [&](float v) {
        AD out(v);
        out.derivatives().setZero();
        return out;
    };

    auto makeVar = [&](float v, int idx) {
        AD out(v);
        out.derivatives().setZero();
        out.derivatives()[idx] = 1.0f;
        return out;
    };

    auto adSqrt = [&](const AD& v) { return Eigen::numext::sqrt(v); };

    auto normalize3 = [&](const std::array<AD, 3>& a) {
        const AD n = adSqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + makeConst(1e-12f));
        std::array<AD, 3> out;
        out[0] = a[0] / n;
        out[1] = a[1] / n;
        out[2] = a[2] / n;
        return out;
    };

    auto dot3 = [&](const std::array<AD, 3>& a, const std::array<AD, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };

    auto cross3 = [&](const std::array<AD, 3>& a, const std::array<AD, 3>& b) {
        std::array<AD, 3> out;
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
        return out;
    };

    auto rotation6dToMatrix = [&](const std::array<AD, 6>& r6) {
        std::array<AD, 3> c0 = {r6[0], r6[1], r6[2]};
        std::array<AD, 3> c1 = {r6[3], r6[4], r6[5]};
        const std::array<AD, 3> b0 = normalize3(c0);
        const AD proj = dot3(b0, c1);
        std::array<AD, 3> u1;
        u1[0] = c1[0] - proj * b0[0];
        u1[1] = c1[1] - proj * b0[1];
        u1[2] = c1[2] - proj * b0[2];
        const std::array<AD, 3> b1 = normalize3(u1);
        const std::array<AD, 3> b2 = cross3(b0, b1);
        // row-major matrix with columns [b0 b1 b2]
        std::array<AD, 9> r;
        r[0] = b0[0]; r[1] = b1[0]; r[2] = b2[0];
        r[3] = b0[1]; r[4] = b1[1]; r[5] = b2[1];
        r[6] = b0[2]; r[7] = b1[2]; r[8] = b2[2];
        return r;
    };

    auto setIdentity4 = [&](std::array<AD, 16>& m) {
        for (int i = 0; i < 16; ++i) m[static_cast<size_t>(i)] = makeConst(0.0f);
        m[0] = makeConst(1.0f);
        m[5] = makeConst(1.0f);
        m[10] = makeConst(1.0f);
        m[15] = makeConst(1.0f);
    };

    auto mul4x4 = [&](const std::array<AD, 16>& a, const std::array<AD, 16>& b, std::array<AD, 16>& out) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                AD v = makeConst(0.0f);
                for (int k = 0; k < 4; ++k) {
                    v += a[static_cast<size_t>(r * 4 + k)] * b[static_cast<size_t>(k * 4 + c)];
                }
                out[static_cast<size_t>(r * 4 + c)] = v;
            }
        }
    };

    auto mul4x4Vec4 = [&](const std::array<AD, 16>& m, const std::array<AD, 4>& v, std::array<AD, 4>& out) {
        for (int r = 0; r < 4; ++r) {
            out[static_cast<size_t>(r)] =
                m[static_cast<size_t>(r * 4 + 0)] * v[0] +
                m[static_cast<size_t>(r * 4 + 1)] * v[1] +
                m[static_cast<size_t>(r * 4 + 2)] * v[2] +
                m[static_cast<size_t>(r * 4 + 3)] * v[3];
        }
    };

    // Use pre-computed data from SetSmplxEvaluationConstants
    const float* shaped = m_evalShaped.data();
    const float* jointsRest = m_evalJointsRest.data();
    const auto& lowerJointToSlot = m_evalLowerJointToSlot;

    std::vector<AD> poseBasis(static_cast<size_t>(m_numPoseBasis), makeConst(0.0f));
    std::vector<std::array<AD, 9>> rotmats(static_cast<size_t>(m_numJoints));
    int poseRow = 0;
    for (int j = 0; j < m_numJoints; ++j) {
        const auto it = lowerJointToSlot.find(j);
        if (it != lowerJointToSlot.end()) {
            const int slot = it->second;
            std::array<AD, 6> r6;
            for (int k = 0; k < 6; ++k) {
                const int didx = slot * 6 + k;
                r6[static_cast<size_t>(k)] = makeVar(lowerR6dFlat[static_cast<size_t>(didx)], didx);
            }
            rotmats[static_cast<size_t>(j)] = rotation6dToMatrix(r6);
        } else {
            float r[9];
            AxisAngleToRotation3x3(
                in.pose165[static_cast<size_t>(j) * 3 + 0],
                in.pose165[static_cast<size_t>(j) * 3 + 1],
                in.pose165[static_cast<size_t>(j) * 3 + 2],
                r);
            for (int i = 0; i < 9; ++i) {
                rotmats[static_cast<size_t>(j)][static_cast<size_t>(i)] = makeConst(r[i]);
            }
        }
        if (j > 0) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    const float ident = (r == c) ? 1.0f : 0.0f;
                    poseBasis[static_cast<size_t>(poseRow++)] =
                        rotmats[static_cast<size_t>(j)][static_cast<size_t>(r * 3 + c)] - makeConst(ident);
                }
            }
        }
    }

    std::vector<AD> poseDelta(static_cast<size_t>(m_numVerts) * 3, makeConst(0.0f));
    for (int vc = 0; vc < m_numVerts * 3; ++vc) {
        AD acc = makeConst(0.0f);
        const size_t base = static_cast<size_t>(vc) * static_cast<size_t>(m_numPoseBasis);
        for (int k = 0; k < m_numPoseBasis; ++k) {
            acc += makeConst(m_poseBlend[base + static_cast<size_t>(k)]) * poseBasis[static_cast<size_t>(k)];
        }
        poseDelta[static_cast<size_t>(vc)] = acc;
    }

    std::vector<AD> posed(static_cast<size_t>(m_numVerts) * 3, makeConst(0.0f));
    for (int i = 0; i < m_numVerts * 3; ++i) {
        posed[static_cast<size_t>(i)] = makeConst(shaped[static_cast<size_t>(i)]) + poseDelta[static_cast<size_t>(i)];
    }

    std::vector<std::array<AD, 16>> localTf(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        std::array<AD, 16> t;
        setIdentity4(t);
        const auto& r = rotmats[static_cast<size_t>(j)];
        t[0] = r[0]; t[1] = r[1]; t[2] = r[2];
        t[4] = r[3]; t[5] = r[4]; t[6] = r[5];
        t[8] = r[6]; t[9] = r[7]; t[10] = r[8];
        const int p = m_kintreeParents[static_cast<size_t>(j)];
        if (p < 0) {
            t[3] = makeConst(jointsRest[j * 3 + 0]);
            t[7] = makeConst(jointsRest[j * 3 + 1]);
            t[11] = makeConst(jointsRest[j * 3 + 2]);
        } else {
            t[3] = makeConst(jointsRest[j * 3 + 0] - jointsRest[p * 3 + 0]);
            t[7] = makeConst(jointsRest[j * 3 + 1] - jointsRest[p * 3 + 1]);
            t[11] = makeConst(jointsRest[j * 3 + 2] - jointsRest[p * 3 + 2]);
        }
        localTf[static_cast<size_t>(j)] = t;
    }

    std::vector<std::array<AD, 16>> worldTf(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        const int p = m_kintreeParents[static_cast<size_t>(j)];
        if (p < 0) {
            worldTf[static_cast<size_t>(j)] = localTf[static_cast<size_t>(j)];
        } else {
            std::array<AD, 16> outTf;
            mul4x4(worldTf[static_cast<size_t>(p)], localTf[static_cast<size_t>(j)], outTf);
            worldTf[static_cast<size_t>(j)] = outTf;
        }
    }

    std::vector<std::array<AD, 16>> restInv(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        std::array<AD, 16> m;
        setIdentity4(m);
        m[3] = makeConst(-jointsRest[j * 3 + 0]);
        m[7] = makeConst(-jointsRest[j * 3 + 1]);
        m[11] = makeConst(-jointsRest[j * 3 + 2]);
        restInv[static_cast<size_t>(j)] = m;
    }

    std::vector<std::array<AD, 16>> skinning(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        std::array<AD, 16> outTf;
        mul4x4(worldTf[static_cast<size_t>(j)], restInv[static_cast<size_t>(j)], outTf);
        skinning[static_cast<size_t>(j)] = outTf;
    }

    SmplxForwardLowerBodyJacobianOutput out;
    out.forward.numVerts = m_numVerts;
    out.forward.numJoints = m_numJoints;
    out.forward.numJointsAll = m_numJoints + static_cast<int>(m_extraJointsIdxs.size()) + static_cast<int>(m_lmkFacesIdx.size());
    out.forward.jointsXyz.resize(static_cast<size_t>(m_numJoints) * 3, 0.0f);
    out.forward.jointsAllXyz.resize(static_cast<size_t>(out.forward.numJointsAll) * 3, 0.0f);
    out.forward.skinnedVertsXyz.resize(static_cast<size_t>(m_numVerts) * 3, 0.0f);
    out.lowerDim = lowerDim;
    out.jointsAllJacobian.resize(static_cast<size_t>(out.forward.numJointsAll) * 3 * static_cast<size_t>(lowerDim), 0.0f);

    std::vector<AD> skinned(static_cast<size_t>(m_numVerts) * 3, makeConst(0.0f));
    for (int v = 0; v < m_numVerts; ++v) {
        const std::array<AD, 4> x4 = {
            posed[static_cast<size_t>(v) * 3 + 0],
            posed[static_cast<size_t>(v) * 3 + 1],
            posed[static_cast<size_t>(v) * 3 + 2],
            makeConst(1.0f)
        };
        std::array<AD, 4> final4 = {makeConst(0.0f), makeConst(0.0f), makeConst(0.0f), makeConst(0.0f)};
        for (int j = 0; j < m_numJoints; ++j) {
            const float w = m_weights[v * m_numJoints + j];
            if (w <= 0.0f) {
                continue;
            }
            std::array<AD, 4> tmp;
            mul4x4Vec4(skinning[static_cast<size_t>(j)], x4, tmp);
            final4[0] += makeConst(w) * tmp[0];
            final4[1] += makeConst(w) * tmp[1];
            final4[2] += makeConst(w) * tmp[2];
        }
        skinned[static_cast<size_t>(v) * 3 + 0] = final4[0];
        skinned[static_cast<size_t>(v) * 3 + 1] = final4[1];
        skinned[static_cast<size_t>(v) * 3 + 2] = final4[2];
        out.forward.skinnedVertsXyz[static_cast<size_t>(v) * 3 + 0] = final4[0].value();
        out.forward.skinnedVertsXyz[static_cast<size_t>(v) * 3 + 1] = final4[1].value();
        out.forward.skinnedVertsXyz[static_cast<size_t>(v) * 3 + 2] = final4[2].value();
    }

    auto writeJointWithJacobian = [&](int outJoint, const AD& x, const AD& y, const AD& z) {
        out.forward.jointsAllXyz[static_cast<size_t>(outJoint) * 3 + 0] = x.value();
        out.forward.jointsAllXyz[static_cast<size_t>(outJoint) * 3 + 1] = y.value();
        out.forward.jointsAllXyz[static_cast<size_t>(outJoint) * 3 + 2] = z.value();
        const AD vals[3] = {x, y, z};
        for (int c = 0; c < 3; ++c) {
            const int row = outJoint * 3 + c;
            for (int d = 0; d < lowerDim; ++d) {
                out.jointsAllJacobian[static_cast<size_t>(row) * static_cast<size_t>(lowerDim) + static_cast<size_t>(d)] = vals[c].derivatives()[d];
            }
        }
    };

    for (int j = 0; j < m_numJoints; ++j) {
        out.forward.jointsXyz[static_cast<size_t>(j) * 3 + 0] = worldTf[static_cast<size_t>(j)][3].value();
        out.forward.jointsXyz[static_cast<size_t>(j) * 3 + 1] = worldTf[static_cast<size_t>(j)][7].value();
        out.forward.jointsXyz[static_cast<size_t>(j) * 3 + 2] = worldTf[static_cast<size_t>(j)][11].value();
        writeJointWithJacobian(
            j,
            worldTf[static_cast<size_t>(j)][3],
            worldTf[static_cast<size_t>(j)][7],
            worldTf[static_cast<size_t>(j)][11]);
    }

    const int baseExtra = m_numJoints;
    for (size_t e = 0; e < m_extraJointsIdxs.size(); ++e) {
        const int vid = m_extraJointsIdxs[e];
        if (vid < 0 || vid >= m_numVerts) {
            throw std::runtime_error("extra joint vertex index out of range");
        }
        writeJointWithJacobian(
            baseExtra + static_cast<int>(e),
            skinned[static_cast<size_t>(vid) * 3 + 0],
            skinned[static_cast<size_t>(vid) * 3 + 1],
            skinned[static_cast<size_t>(vid) * 3 + 2]);
    }

    const int baseLmk = baseExtra + static_cast<int>(m_extraJointsIdxs.size());
    for (size_t l = 0; l < m_lmkFacesIdx.size(); ++l) {
        const int faceId = m_lmkFacesIdx[l];
        const int numFaces = static_cast<int>(m_facesTri.size() / 3);
        if (faceId < 0 || faceId >= numFaces) {
            throw std::runtime_error("landmark face index out of range");
        }
        const int v0 = m_facesTri[faceId * 3 + 0];
        const int v1 = m_facesTri[faceId * 3 + 1];
        const int v2 = m_facesTri[faceId * 3 + 2];
        if (v0 < 0 || v0 >= m_numVerts || v1 < 0 || v1 >= m_numVerts || v2 < 0 || v2 >= m_numVerts) {
            throw std::runtime_error("landmark face vertex index out of range");
        }
        const AD bx = makeConst(m_lmkBaryCoords[l * 3 + 0]);
        const AD by = makeConst(m_lmkBaryCoords[l * 3 + 1]);
        const AD bz = makeConst(m_lmkBaryCoords[l * 3 + 2]);
        writeJointWithJacobian(
            baseLmk + static_cast<int>(l),
            bx * skinned[static_cast<size_t>(v0) * 3 + 0] + by * skinned[static_cast<size_t>(v1) * 3 + 0] + bz * skinned[static_cast<size_t>(v2) * 3 + 0],
            bx * skinned[static_cast<size_t>(v0) * 3 + 1] + by * skinned[static_cast<size_t>(v1) * 3 + 1] + bz * skinned[static_cast<size_t>(v2) * 3 + 1],
            bx * skinned[static_cast<size_t>(v0) * 3 + 2] + by * skinned[static_cast<size_t>(v1) * 3 + 2] + bz * skinned[static_cast<size_t>(v2) * 3 + 2]);
    }

    return out;
}

SmplxSelectedJointsLowerBodyJacobianOutput SmplxForwardLite::ForwardSelectedJointsWithLowerBodyR6dJacobian(
    const SmplxForwardInput& in,
    const std::vector<float>& lowerR6dFlat) const {
    if (!m_evalConstantsSet) {
        throw std::runtime_error("SmplxForwardLite: must call SetSmplxEvaluationConstants before ForwardSelectedJointsWithLowerBodyR6dJacobian");
    }
    if (in.pose165.size() != static_cast<size_t>(m_numJoints) * 3) {
        throw std::runtime_error("pose165 size mismatch");
    }
    if (lowerR6dFlat.size() != m_evalLowerJointIds.size() * 6ULL) {
        throw std::runtime_error("lowerR6dFlat size mismatch");
    }

    const int baseExtra = m_numJoints;
    const int baseLmk = baseExtra + static_cast<int>(m_extraJointsIdxs.size());

    using Deriv = Eigen::Matrix<float, kLowerBodyAdDim, 1>;
    using AD = Eigen::AutoDiffScalar<Deriv>;
    const int lowerDim = static_cast<int>(lowerR6dFlat.size());
    if (lowerDim != kLowerBodyAdDim) {
        throw std::runtime_error("ForwardSelectedJointsWithLowerBodyR6dJacobian expects lowerDim=36");
    }

    // Thread-local buffers for reuse across calls to avoid repeated allocations
    struct ThreadLocalBuffers {
        std::vector<AD> poseBasis;
        std::vector<std::array<AD, 9>> rotmats;
        std::vector<std::array<AD, 16>> localTf;
        std::vector<std::array<AD, 16>> worldTf;
        std::vector<std::array<AD, 16>> restInv;
        std::vector<std::array<AD, 16>> skinning;
        std::vector<AD> skinnedReq;
    };
    thread_local ThreadLocalBuffers tlBuffers;

    auto makeConst = [&](float v) {
        AD out(v);
        out.derivatives().setZero();
        return out;
    };
    auto makeVar = [&](float v, int idx) {
        AD out(v);
        out.derivatives().setZero();
        out.derivatives()[idx] = 1.0f;
        return out;
    };
    auto adSqrt = [&](const AD& v) { return Eigen::numext::sqrt(v); };

    auto normalize3 = [&](const std::array<AD, 3>& a) {
        const AD n = adSqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + makeConst(1e-12f));
        std::array<AD, 3> out;
        out[0] = a[0] / n;
        out[1] = a[1] / n;
        out[2] = a[2] / n;
        return out;
    };
    auto dot3 = [&](const std::array<AD, 3>& a, const std::array<AD, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto cross3 = [&](const std::array<AD, 3>& a, const std::array<AD, 3>& b) {
        std::array<AD, 3> out;
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
        return out;
    };
    auto rotation6dToMatrix = [&](const std::array<AD, 6>& r6) {
        std::array<AD, 3> c0 = {r6[0], r6[1], r6[2]};
        std::array<AD, 3> c1 = {r6[3], r6[4], r6[5]};
        const std::array<AD, 3> b0 = normalize3(c0);
        const AD proj = dot3(b0, c1);
        std::array<AD, 3> u1;
        u1[0] = c1[0] - proj * b0[0];
        u1[1] = c1[1] - proj * b0[1];
        u1[2] = c1[2] - proj * b0[2];
        const std::array<AD, 3> b1 = normalize3(u1);
        const std::array<AD, 3> b2 = cross3(b0, b1);
        std::array<AD, 9> r;
        r[0] = b0[0]; r[1] = b1[0]; r[2] = b2[0];
        r[3] = b0[1]; r[4] = b1[1]; r[5] = b2[1];
        r[6] = b0[2]; r[7] = b1[2]; r[8] = b2[2];
        return r;
    };
    auto setIdentity4 = [&](std::array<AD, 16>& m) {
        for (int i = 0; i < 16; ++i) m[static_cast<size_t>(i)] = makeConst(0.0f);
        m[0] = makeConst(1.0f);
        m[5] = makeConst(1.0f);
        m[10] = makeConst(1.0f);
        m[15] = makeConst(1.0f);
    };
    auto mul4x4 = [&](const std::array<AD, 16>& a, const std::array<AD, 16>& b, std::array<AD, 16>& out) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                AD v = makeConst(0.0f);
                for (int k = 0; k < 4; ++k) {
                    v += a[static_cast<size_t>(r * 4 + k)] * b[static_cast<size_t>(k * 4 + c)];
                }
                out[static_cast<size_t>(r * 4 + c)] = v;
            }
        }
    };
    auto mul4x4Vec4 = [&](const std::array<AD, 16>& m, const std::array<AD, 4>& v, std::array<AD, 4>& out) {
        for (int r = 0; r < 4; ++r) {
            out[static_cast<size_t>(r)] =
                m[static_cast<size_t>(r * 4 + 0)] * v[0] +
                m[static_cast<size_t>(r * 4 + 1)] * v[1] +
                m[static_cast<size_t>(r * 4 + 2)] * v[2] +
                m[static_cast<size_t>(r * 4 + 3)] * v[3];
        }
    };

    // Use pre-computed data from SetSmplxEvaluationConstants
    const auto& lowerJointToSlot = m_evalLowerJointToSlot;
    const auto& requiredVertices = m_evalRequiredVertices;
    const auto& vidToLocal = m_evalVidToLocal;
    const bool needsVertexBranch = !requiredVertices.empty();

    // Use pre-computed shaped and jointsRest from SetSmplxEvaluationConstants
    const float* shaped = m_evalShaped.data();
    const float* jointsRest = m_evalJointsRest.data();
    // Use thread-local buffers to avoid repeated allocations
    std::vector<AD>& poseBasis = tlBuffers.poseBasis;
    if (needsVertexBranch) {
        poseBasis.assign(static_cast<size_t>(m_numPoseBasis), makeConst(0.0f));
    }
    std::vector<std::array<AD, 9>>& rotmats = tlBuffers.rotmats;
    rotmats.resize(static_cast<size_t>(m_numJoints));
    int poseRow = 0;
    for (int j = 0; j < m_numJoints; ++j) {
        const auto it = lowerJointToSlot.find(j);
        if (it != lowerJointToSlot.end()) {
            const int slot = it->second;
            std::array<AD, 6> r6;
            for (int k = 0; k < 6; ++k) {
                const int didx = slot * 6 + k;
                r6[static_cast<size_t>(k)] = makeVar(lowerR6dFlat[static_cast<size_t>(didx)], didx);
            }
            rotmats[static_cast<size_t>(j)] = rotation6dToMatrix(r6);
        } else {
            float r[9];
            AxisAngleToRotation3x3(
                in.pose165[static_cast<size_t>(j) * 3 + 0],
                in.pose165[static_cast<size_t>(j) * 3 + 1],
                in.pose165[static_cast<size_t>(j) * 3 + 2],
                r);
            for (int i = 0; i < 9; ++i) {
                rotmats[static_cast<size_t>(j)][static_cast<size_t>(i)] = makeConst(r[i]);
            }
        }
        if (needsVertexBranch && j > 0) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    const float ident = (r == c) ? 1.0f : 0.0f;
                    poseBasis[static_cast<size_t>(poseRow++)] =
                        rotmats[static_cast<size_t>(j)][static_cast<size_t>(r * 3 + c)] - makeConst(ident);
                }
            }
        }
    }
    std::vector<std::array<AD, 16>>& localTf = tlBuffers.localTf;
    localTf.resize(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        std::array<AD, 16> t;
        setIdentity4(t);
        const auto& r = rotmats[static_cast<size_t>(j)];
        t[0] = r[0]; t[1] = r[1]; t[2] = r[2];
        t[4] = r[3]; t[5] = r[4]; t[6] = r[5];
        t[8] = r[6]; t[9] = r[7]; t[10] = r[8];
        const int p = m_kintreeParents[static_cast<size_t>(j)];
        if (p < 0) {
            t[3] = makeConst(jointsRest[j * 3 + 0]);
            t[7] = makeConst(jointsRest[j * 3 + 1]);
            t[11] = makeConst(jointsRest[j * 3 + 2]);
        } else {
            t[3] = makeConst(jointsRest[j * 3 + 0] - jointsRest[p * 3 + 0]);
            t[7] = makeConst(jointsRest[j * 3 + 1] - jointsRest[p * 3 + 1]);
            t[11] = makeConst(jointsRest[j * 3 + 2] - jointsRest[p * 3 + 2]);
        }
        localTf[static_cast<size_t>(j)] = t;
    }

    std::vector<std::array<AD, 16>>& worldTf = tlBuffers.worldTf;
    worldTf.resize(static_cast<size_t>(m_numJoints));
    for (int j = 0; j < m_numJoints; ++j) {
        const int p = m_kintreeParents[static_cast<size_t>(j)];
        if (p < 0) {
            worldTf[static_cast<size_t>(j)] = localTf[static_cast<size_t>(j)];
        } else {
            std::array<AD, 16> outTf;
            mul4x4(worldTf[static_cast<size_t>(p)], localTf[static_cast<size_t>(j)], outTf);
            worldTf[static_cast<size_t>(j)] = outTf;
        }
    }

    std::vector<std::array<AD, 16>>& skinning = tlBuffers.skinning;
    if (needsVertexBranch) {
        std::vector<std::array<AD, 16>>& restInv = tlBuffers.restInv;
        restInv.resize(static_cast<size_t>(m_numJoints));
        for (int j = 0; j < m_numJoints; ++j) {
            std::array<AD, 16> m;
            setIdentity4(m);
            m[3] = makeConst(-jointsRest[j * 3 + 0]);
            m[7] = makeConst(-jointsRest[j * 3 + 1]);
            m[11] = makeConst(-jointsRest[j * 3 + 2]);
            restInv[static_cast<size_t>(j)] = m;
        }

        skinning.resize(static_cast<size_t>(m_numJoints));
        for (int j = 0; j < m_numJoints; ++j) {
            std::array<AD, 16> outTf;
            mul4x4(worldTf[static_cast<size_t>(j)], restInv[static_cast<size_t>(j)], outTf);
            skinning[static_cast<size_t>(j)] = outTf;
        }
    }
    std::vector<AD>& skinnedReq = tlBuffers.skinnedReq;
    if (needsVertexBranch) {
        skinnedReq.assign(static_cast<size_t>(requiredVertices.size() * 3), makeConst(0.0f));
        for (size_t i = 0; i < requiredVertices.size(); ++i) {
            const int vid = requiredVertices[i];
            std::array<AD, 4> x4 = {makeConst(0.0f), makeConst(0.0f), makeConst(0.0f), makeConst(1.0f)};
            for (int c = 0; c < 3; ++c) {
                AD acc = makeConst(0.0f);
                const int row = vid * 3 + c;
                for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(m_sparsePoseBlend, row); it; ++it) {
                    acc += makeConst(it.value()) * poseBasis[static_cast<size_t>(it.col())];
                }
                x4[static_cast<size_t>(c)] = makeConst(shaped[vid * 3 + c]) + acc;
            }
            std::array<AD, 4> final4 = {makeConst(0.0f), makeConst(0.0f), makeConst(0.0f), makeConst(0.0f)};
            for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(m_sparseWeights, vid); it; ++it) {
                const int j = static_cast<int>(it.col());
                const float w = it.value();
                std::array<AD, 4> tmp;
                mul4x4Vec4(skinning[static_cast<size_t>(j)], x4, tmp);
                final4[0] += makeConst(w) * tmp[0];
                final4[1] += makeConst(w) * tmp[1];
                final4[2] += makeConst(w) * tmp[2];
            }
            skinnedReq[i * 3 + 0] = final4[0];
            skinnedReq[i * 3 + 1] = final4[1];
            skinnedReq[i * 3 + 2] = final4[2];
        }
    }

    SmplxSelectedJointsLowerBodyJacobianOutput out;
    out.lowerDim = lowerDim;
    out.selectedJointIds = m_evalSelectedJointIds;
    out.selectedJointsXyz.resize(static_cast<size_t>(m_evalSelectedJointIds.size() * 3), 0.0f);
    out.selectedJointsJacobian.resize(static_cast<size_t>(m_evalSelectedJointIds.size() * 3 * lowerDim), 0.0f);
    out.selectedVertexIds = m_evalSelectedVertexIds;
    out.selectedVerticesXyz.resize(static_cast<size_t>(m_evalSelectedVertexIds.size() * 3), 0.0f);
    out.selectedVerticesJacobian.resize(static_cast<size_t>(m_evalSelectedVertexIds.size() * 3 * lowerDim), 0.0f);

    auto writeJoint = [&](size_t outIdx, const AD& x, const AD& y, const AD& z) {
        out.selectedJointsXyz[outIdx * 3 + 0] = x.value();
        out.selectedJointsXyz[outIdx * 3 + 1] = y.value();
        out.selectedJointsXyz[outIdx * 3 + 2] = z.value();
        const AD vals[3] = {x, y, z};
        for (int c = 0; c < 3; ++c) {
            const int row = static_cast<int>(outIdx) * 3 + c;
            for (int d = 0; d < lowerDim; ++d) {
                out.selectedJointsJacobian[static_cast<size_t>(row) * static_cast<size_t>(lowerDim) + static_cast<size_t>(d)] = vals[c].derivatives()[d];
            }
        }
    };
    auto writeVertex = [&](size_t outIdx, const AD& x, const AD& y, const AD& z) {
        out.selectedVerticesXyz[outIdx * 3 + 0] = x.value();
        out.selectedVerticesXyz[outIdx * 3 + 1] = y.value();
        out.selectedVerticesXyz[outIdx * 3 + 2] = z.value();
        const AD vals[3] = {x, y, z};
        for (int c = 0; c < 3; ++c) {
            const int row = static_cast<int>(outIdx) * 3 + c;
            for (int d = 0; d < lowerDim; ++d) {
                out.selectedVerticesJacobian[static_cast<size_t>(row) * static_cast<size_t>(lowerDim) + static_cast<size_t>(d)] = vals[c].derivatives()[d];
            }
        }
    };

    for (size_t i = 0; i < m_evalSelectedJointIds.size(); ++i) {
        const int sid = m_evalSelectedJointIds[i];
        if (sid < m_numJoints) {
            writeJoint(i, worldTf[static_cast<size_t>(sid)][3], worldTf[static_cast<size_t>(sid)][7], worldTf[static_cast<size_t>(sid)][11]);
        } else if (sid < baseLmk) {
            const int vid = m_extraJointsIdxs[static_cast<size_t>(sid - baseExtra)];
            const int loc = vidToLocal.at(vid);
            writeJoint(i, skinnedReq[static_cast<size_t>(loc) * 3 + 0], skinnedReq[static_cast<size_t>(loc) * 3 + 1], skinnedReq[static_cast<size_t>(loc) * 3 + 2]);
        } else {
            // Face landmark
            const int lmk = sid - baseLmk;

            // Validate landmark index
            if (lmk < 0 || lmk >= static_cast<int>(m_lmkFacesIdx.size())) {
                throw std::runtime_error("SmplxForwardLite: landmark index " + std::to_string(lmk) +
                                       " out of range [0, " + std::to_string(m_lmkFacesIdx.size()) + ")");
            }

            const int faceId = m_lmkFacesIdx[static_cast<size_t>(lmk)];

            // Validate face index
            const int numFaces = static_cast<int>(m_facesTri.size()) / 3;
            if (faceId < 0 || faceId >= numFaces) {
                throw std::runtime_error("SmplxForwardLite: face index " + std::to_string(faceId) +
                                       " for landmark " + std::to_string(lmk) +
                                       " out of range [0, " + std::to_string(numFaces) + ")");
            }

            const int v0 = m_facesTri[static_cast<size_t>(faceId) * 3 + 0];
            const int v1 = m_facesTri[static_cast<size_t>(faceId) * 3 + 1];
            const int v2 = m_facesTri[static_cast<size_t>(faceId) * 3 + 2];

            // Validate vertex indices
            if (v0 < 0 || v0 >= m_numVerts) {
                throw std::runtime_error("SmplxForwardLite: vertex v0=" + std::to_string(v0) +
                                       " for face " + std::to_string(faceId) +
                                       " out of range [0, " + std::to_string(m_numVerts) + ")");
            }
            if (v1 < 0 || v1 >= m_numVerts) {
                throw std::runtime_error("SmplxForwardLite: vertex v1=" + std::to_string(v1) +
                                       " for face " + std::to_string(faceId) +
                                       " out of range [0, " + std::to_string(m_numVerts) + ")");
            }
            if (v2 < 0 || v2 >= m_numVerts) {
                throw std::runtime_error("SmplxForwardLite: vertex v2=" + std::to_string(v2) +
                                       " for face " + std::to_string(faceId) +
                                       " out of range [0, " + std::to_string(m_numVerts) + ")");
            }

            // vidToLocal.at() will throw std::out_of_range if vertices not in required set
            // This is expected behavior - the dependency analysis should have included these vertices
            const int i0 = vidToLocal.at(v0);
            const int i1 = vidToLocal.at(v1);
            const int i2 = vidToLocal.at(v2);

            // Validate barycentric coordinate access
            if (static_cast<size_t>(lmk) * 3 + 2 >= m_lmkBaryCoords.size()) {
                throw std::runtime_error("SmplxForwardLite: barycentric coords for landmark " + std::to_string(lmk) +
                                       " out of range (size: " + std::to_string(m_lmkBaryCoords.size()) + ")");
            }

            const AD bx = makeConst(m_lmkBaryCoords[static_cast<size_t>(lmk) * 3 + 0]);
            const AD by = makeConst(m_lmkBaryCoords[static_cast<size_t>(lmk) * 3 + 1]);
            const AD bz = makeConst(m_lmkBaryCoords[static_cast<size_t>(lmk) * 3 + 2]);
            writeJoint(
                i,
                bx * skinnedReq[static_cast<size_t>(i0) * 3 + 0] + by * skinnedReq[static_cast<size_t>(i1) * 3 + 0] + bz * skinnedReq[static_cast<size_t>(i2) * 3 + 0],
                bx * skinnedReq[static_cast<size_t>(i0) * 3 + 1] + by * skinnedReq[static_cast<size_t>(i1) * 3 + 1] + bz * skinnedReq[static_cast<size_t>(i2) * 3 + 1],
                bx * skinnedReq[static_cast<size_t>(i0) * 3 + 2] + by * skinnedReq[static_cast<size_t>(i1) * 3 + 2] + bz * skinnedReq[static_cast<size_t>(i2) * 3 + 2]);
        }
    }
    for (size_t i = 0; i < m_evalSelectedVertexIds.size(); ++i) {
        const int vid = m_evalSelectedVertexIds[i];
        const int loc = vidToLocal.at(vid);
        writeVertex(
            i,
            skinnedReq[static_cast<size_t>(loc) * 3 + 0],
            skinnedReq[static_cast<size_t>(loc) * 3 + 1],
            skinnedReq[static_cast<size_t>(loc) * 3 + 2]);
    }
    return out;
}
}

CARBON_NAMESPACE_END(TITAN_NAMESPACE)

