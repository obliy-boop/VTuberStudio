// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/common/Defs.h>

#include <array>
#include <atomic>
#include <Eigen/Sparse>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

namespace bodyopt
{


struct SmplxForwardInput
{
    std::vector<float> pose165; // axis-angle per joint, 55*3
    // betas10 removed - now set via SetSmplxEvaluationConstants
};

struct SmplxForwardOutput
{
    int numVerts = 0;
    int numJoints = 0;
    int numJointsAll = 0;
    std::vector<float> jointsXyz; // [J,3] row-major (base 55)
    std::vector<float> jointsAllXyz; // [127,3] row-major (55 + 21 + 51)
    std::vector<float> skinnedVertsXyz; // [V,3] row-major
};

struct SmplxForwardLowerBodyJacobianOutput
{
    SmplxForwardOutput forward;
    int lowerDim = 0;
    // Dense Jacobian row-major: [numJointsAll*3, lowerDim]
    std::vector<float> jointsAllJacobian;
};

struct SmplxSelectedJointsLowerBodyJacobianOutput
{
    int lowerDim = 0;
    std::vector<int> selectedJointIds; // jointsAll indexing, output order
    std::vector<float> selectedJointsXyz; // [N,3]
    std::vector<float> selectedJointsJacobian; // [N*3, lowerDim] row-major
    std::vector<int> selectedVertexIds; // vertex indexing, output order
    std::vector<float> selectedVerticesXyz; // [Nv,3]
    std::vector<float> selectedVerticesJacobian; // [Nv*3, lowerDim] row-major
};

struct SmplxSelectedForwardTimingBreakdown
{
    int calls = 0;
    double totalMs = 0.0;
    double depCacheMs = 0.0;
    double shapedCacheMs = 0.0;
    double rotPoseMs = 0.0;
    double fkSkinPrepMs = 0.0;
    double skinVerticesMs = 0.0;
    double outputWriteMs = 0.0;
};

class SmplxForwardLite
{
public:
    bool LoadStaticFromFolder(const std::string& folder);

    // Load from raw arrays matching UE MetaHumanSMPLXData format
    // IMPORTANT: UE stores Eigen matrices in COLUMN-MAJOR format (Eigen default)
    // This method automatically converts to the row-major format used by SmplxForwardLite
    bool LoadFromArrays(
        const std::vector<float>& verts, // NumVerts * 3, column-major (NumVerts, 3)
        const std::vector<uint32_t>& faces, // NumFaces * 3
        const std::vector<float>& jointReg, // NumJoints * NumVerts, column-major (NumJoints, NumVerts)
        const std::vector<float>& weights, // NumVerts * NumJoints, column-major (NumVerts, NumJoints)
        const std::vector<float>& blendShapes, // NumVerts*3 * (NumShapes + 9*(NumJoints-1)), column-major
        const std::vector<int>& kintreeParents, // NumJoints parent indices
        const std::vector<int>& extraJointsIdxs, // 21 extra joint indices
        const std::vector<int>& lmkFacesIdx, // 51 landmark face indices
        const std::vector<float>& lmkBaryCoords); // 51*3 barycentric coords

    // Set evaluation constants that remain the same for all frames in a sequence
    // Call this once before Forward/ForwardWithLowerBodyR6dJacobian/ForwardSelectedJointsWithLowerBodyR6dJacobian calls
    // betas10: shape coefficients (must have at least NumBetas() elements)
    // lowerJointIds: joint IDs for lower body optimization (optional, empty = use default)
    // selectedJointIds: joint IDs to evaluate (optional, empty = all joints)
    // selectedVertexIds: vertex IDs to evaluate (optional, empty = no vertices)
    void SetSmplxEvaluationConstants(
        const std::vector<float>& betas10,
        const std::vector<int>& lowerJointIds = {},
        const std::vector<int>& selectedJointIds = {},
        const std::vector<int>& selectedVertexIds = {});

    SmplxForwardOutput Forward(const SmplxForwardInput& in) const;
    SmplxForwardLowerBodyJacobianOutput ForwardWithLowerBodyR6dJacobian(
        const SmplxForwardInput& in,
        const std::vector<float>& lowerR6dFlat) const;
    SmplxSelectedJointsLowerBodyJacobianOutput ForwardSelectedJointsWithLowerBodyR6dJacobian(
        const SmplxForwardInput& in,
        const std::vector<float>& lowerR6dFlat) const;

    int NumVerts() const { return m_numVerts; }
    int NumJoints() const { return m_numJoints; }
    int NumBetas() const { return m_numBetas; }

    // Getters for model data (useful for exporting or API initialization)
    const std::vector<float>& GetVerts() const { return m_vTemplate; }
    std::vector<uint32_t> GetFaces() const
    {
        std::vector<uint32_t> facesUint32(m_facesTri.size());
        std::transform(m_facesTri.begin(), m_facesTri.end(),
            facesUint32.begin(),
            [](int val)
            { return static_cast<uint32_t>(val); });
        return facesUint32;
    }
    const std::vector<float>& GetJointRegressor() const { return m_jointReg; }
    const std::vector<float>& GetWeights() const { return m_weights; }
    std::vector<float> GetBlendShapes() const
    {
        std::vector<float> blendShapes;
        blendShapes.reserve(m_shapeBlend.size() + m_poseBlend.size());
        blendShapes.insert(blendShapes.end(), m_shapeBlend.begin(), m_shapeBlend.end());
        blendShapes.insert(blendShapes.end(), m_poseBlend.begin(), m_poseBlend.end());
        return blendShapes; // RVO/move semantics
    }

    const std::vector<int>& GetKintreeParents() const { return m_kintreeParents; }
    const std::vector<int>& GetExtraJointsIdxs() const { return m_extraJointsIdxs; }
    const std::vector<int>& GetLandmarkFacesIdx() const { return m_lmkFacesIdx; }
    const std::vector<float>& GetLandmarkBaryCoords() const { return m_lmkBaryCoords; }

private:
    // Helper methods for SetSmplxEvaluationConstants
    void BuildShapedAndJointsRest();
    void BuildDependencyData();
    void BuildSparseMatrices();


    int m_numVerts = 0;
    int m_numJoints = 0;
    int m_numBetas = 0;
    int m_numPoseBasis = 0;

    std::vector<float> m_vTemplate; // [V,3]
    std::vector<float> m_weights; // [V,J]
    std::vector<float> m_jointReg; // [J,V]
    std::vector<float> m_shapeBlend; // [V,3,B]
    std::vector<float> m_poseBlend; // [V,3,P], P should be 9*(J-1)
    std::vector<int> m_kintreeParents; // [J]
    std::vector<int> m_extraJointsIdxs; // [21]
    std::vector<int> m_lmkFacesIdx; // [51]
    std::vector<float> m_lmkBaryCoords; // [51,3]
    std::vector<int> m_facesTri; // [F,3] flattened

    // Evaluation constants - set once via SetSmplxEvaluationConstants, constant for all frames
    std::vector<float> m_evalBetas;  // Shape coefficients
    std::vector<int> m_evalLowerJointIds;  // Lower body joint IDs for optimization
    std::vector<int> m_evalSelectedJointIds;  // Selected joints to evaluate
    std::vector<int> m_evalSelectedVertexIds;  // Selected vertices to evaluate

    // Pre-computed data based on evaluation constants
    std::vector<float> m_evalShaped;  // Shaped vertices based on betas
    std::vector<float> m_evalJointsRest;  // Rest joints based on betas
    std::vector<int> m_evalRequiredVertices;  // Required vertices for selected outputs
    std::unordered_map<int, int> m_evalVidToLocal;  // Vertex ID to local index mapping
    std::unordered_map<int, int> m_evalLowerJointToSlot;  // Lower joint ID to slot mapping

    // Sparse matrices for selected vertex evaluation
    Eigen::SparseMatrix<float, Eigen::RowMajor> m_sparsePoseBlend;
    Eigen::SparseMatrix<float, Eigen::RowMajor> m_sparseWeights;
    Eigen::SparseMatrix<float, Eigen::RowMajor> m_sparseJointReg;

    bool m_evalConstantsSet = false;  // Whether SetSmplxEvaluationConstants has been called
};

} // namespace bodyopt

CARBON_NAMESPACE_END(TITAN_NAMESPACE)

