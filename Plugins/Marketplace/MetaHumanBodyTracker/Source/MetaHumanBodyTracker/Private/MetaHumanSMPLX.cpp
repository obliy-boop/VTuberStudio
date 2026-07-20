// Copyright Epic Games, Inc. All Rights Reserved.

#include "MetaHumanSMPLX.h"
#include "MetaHumanResize.h"
#include "MetaHumanBodyTrackerLog.h"

#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "SkeletalMeshAttributes.h"
#include "Rendering/SkeletalMeshModel.h"
#include "ReferenceSkeleton.h"
#include "Animation/Skeleton.h"
#include "SkeletalRenderPublic.h"
#include "StaticMeshOperations.h"
#include "Engine/StaticMesh.h"

#include "SupressWarnings.h"

MH_DISABLE_EIGEN_WARNINGS
#include "Eigen/Dense"
MH_ENABLE_WARNINGS



class FMetaHumanSMPLXImpl
{
public:

	// This class, its variables, and the maths behind it, assumes you are already familiar with the SMPLX model.
	// See https://github.com/vchoutas/smplx for more detail of how the SMPLX model is defined and controlled.

	bool Init(UMetaHumanSMPLXData* InData);
	bool IsInitialized() const;
	const UMetaHumanSMPLXData* GetSmplxData() const { return SmplxData.Get(); }

	TStrongObjectPtr<UMetaHumanSMPLXData> SmplxData;

	bool SetShape(const TArray<float>& InShapeData);
	bool SetPose(const TArray<float>& InPoseData);

	bool SetAccountForHeight(bool bInAccountForHeight);
	bool GetAccountForHeight() const;

	void Calculate();

	const TArray<FVector>& GetSkinnedVerticesSMPL();
	const TArray<FVector>& GetSkinnedVerticesUE();
	const TArray<TTuple<uint32, uint32, uint32>>& GetFaces() const;
	const TArray<TTuple<FString, int32, FTransform>>& GetBoneHierarchyUE();
	float GetHeightOffset() const;

	const TArray<FString> JointNamesSMPL =
	{
		"pelvis",
		"left_hip",
		"right_hip",
		"spine1",
		"left_knee",
		"right_knee",
		"spine2",
		"left_ankle",
		"right_ankle",
		"spine3",
		"left_foot",
		"right_foot",
		"neck",
		"left_collar",
		"right_collar",
		"head",
		"left_shoulder",
		"right_shoulder",
		"left_elbow",
		"right_elbow",
		"left_wrist",
		"right_wrist",
		"jaw",
		"left_eye_smplhf",
		"right_eye_smplhf",
		"left_index1",
		"left_index2",
		"left_index3",
		"left_middle1",
		"left_middle2",
		"left_middle3",
		"left_pinky1",
		"left_pinky2",
		"left_pinky3",
		"left_ring1",
		"left_ring2",
		"left_ring3",
		"left_thumb1",
		"left_thumb2",
		"left_thumb3",
		"right_index1",
		"right_index2",
		"right_index3",
		"right_middle1",
		"right_middle2",
		"right_middle3",
		"right_pinky1",
		"right_pinky2",
		"right_pinky3",
		"right_ring1",
		"right_ring2",
		"right_ring3",
		"right_thumb1",
		"right_thumb2",
		"right_thumb3",
	};

	const TArray<FString> JointNamesUE =
	{
		"pelvis",
		"thigh_l",
		"thigh_r",
		"spine_01",
		"calf_l",
		"calf_r",
		"spine_02",
		"foot_l",
		"foot_r",
		"spine_03",
		"ball_l",
		"ball_r",
		"neck_01",
		"clavicle_l",
		"clavicle_r",
		"head",
		"upperarm_l",
		"upperarm_r",
		"lowerarm_l",
		"lowerarm_r",
		"hand_l",
		"hand_r",
		"jaw",
		"eye_l",
		"eye_r",
		"index_01_l",
		"index_02_l",
		"index_03_l",
		"middle_01_l",
		"middle_02_l",
		"middle_03_l",
		"pinky_01_l",
		"pinky_02_l",
		"pinky_03_l",
		"ring_01_l",
		"ring_02_l",
		"ring_03_l",
		"thumb_01_l",
		"thumb_02_l",
		"thumb_03_l",
		"index_01_r",
		"index_02_r",
		"index_03_r",
		"middle_01_r",
		"middle_02_r",
		"middle_03_r",
		"pinky_01_r",
		"pinky_02_r",
		"pinky_03_r",
		"ring_01_r",
		"ring_02_r",
		"ring_03_r",
		"thumb_01_r",
		"thumb_02_r",
		"thumb_03_r"
	};

	const TArray<int32> KinematicTree = { -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53 };
	Eigen::Matrix<float, -1, -1> Verts;
	TArray<TTuple<uint32, uint32, uint32>> Faces;
	Eigen::Matrix<float, -1, -1> JointReg;
	Eigen::Matrix<float, -1, -1> Weights;
	Eigen::Matrix<float, -1, -1> ShapeBlendShapes;
	Eigen::Matrix<float, -1, -1> PoseBlendShapes;

	Eigen::Matrix<float, -1, 1> Shape;
	Eigen::Matrix<float, -1, 1> Pose;

	static constexpr int32 NumVerts = 10475;
	static constexpr int32 NumFaces = 20908;

	static constexpr int32 NumJoints = 55;
	static constexpr int32 NumShapes = 400;

	bool bIsInitialized = false;
	bool bIsUpToDate = false;
	bool bAccountForHeight = true;
	float HeightOffset = 0;
	TArray<float> ShapeCached;

	TArray<FVector> SkinnedVertsSMPL;
	TArray<FVector> SkinnedVertsUE;
	TArray<TTuple<FString, int32, FTransform>> BoneHierarchyUE;
};

bool FMetaHumanSMPLXImpl::Init(UMetaHumanSMPLXData* InData)
{
	if (!InData)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "No SMPLX data asset");
		return false;
	}

	if (InData->Verts.Num() == NumVerts * 3 &&
		InData->Faces.Num() == NumFaces * 3 &&
		InData->JointReg.Num() == NumJoints * NumVerts &&
		InData->Weights.Num() == NumVerts * NumJoints &&
		InData->BlendShapes.Num() == NumVerts * 3 * (NumShapes + 9 * (NumJoints - 1)))
	{
		Verts.resize(NumVerts, 3);
		FMemory::Memcpy(Verts.data(), InData->Verts.GetData(), Verts.size() * sizeof(float));

		Eigen::Matrix<uint32, -1, -1> FacesMatrix;
		FacesMatrix.resize(NumFaces, 3);
		FMemory::Memcpy(FacesMatrix.data(), InData->Faces.GetData(), FacesMatrix.size() * sizeof(uint32));

		Faces.SetNumUninitialized(NumFaces);
		for (int32 Face = 0; Face < NumFaces; ++Face)
		{
			Faces[Face] = TTuple<uint32, uint32, uint32>(FacesMatrix(Face, 0), FacesMatrix(Face, 1), FacesMatrix(Face, 2));
		}

		JointReg.resize(NumJoints, NumVerts);
		FMemory::Memcpy(JointReg.data(), InData->JointReg.GetData(), JointReg.size() * sizeof(float));

		Weights.resize(NumVerts, NumJoints);
		FMemory::Memcpy(Weights.data(), InData->Weights.GetData(), Weights.size() * sizeof(float));

		Eigen::Matrix<float, -1, -1> BlendShapes;
		BlendShapes.resize(NumVerts * 3, NumShapes + 9 * (NumJoints - 1));
		FMemory::Memcpy(BlendShapes.data(), InData->BlendShapes.GetData(), BlendShapes.size() * sizeof(float));

		ShapeBlendShapes = BlendShapes.block(0, 0, BlendShapes.rows(), NumShapes);
		PoseBlendShapes = BlendShapes.block(0, NumShapes, BlendShapes.rows(), 9 * (NumJoints - 1));

		Shape.resize(NumShapes, 1);
		Shape.setZero();

		ShapeCached = FMetaHumanSMPLX::InvalidShape(); // An impossible shape to ensure no accidental cache hits

		Pose.resize(NumJoints * 3, 1);
		Pose.setZero();

		SmplxData = TStrongObjectPtr<UMetaHumanSMPLXData>(InData);
		bIsInitialized = true;
		bIsUpToDate = false;
	}
	else
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Incorrect SMPLX data asset content");
	}

	return bIsInitialized;
}

bool FMetaHumanSMPLXImpl::IsInitialized() const
{
	return bIsInitialized;
}

bool FMetaHumanSMPLXImpl::SetShape(const TArray<float>& InShapeData)
{
	if (!bIsInitialized)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Not initialized");
		return false;
	}

	if (InShapeData.Num() > Shape.rows()) // Only specifying a sub-set of the shape is ok
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Too many shape data items, got %i expected at most %lli", InShapeData.Num(), Shape.rows());
		return false;
	}

	if (InShapeData == ShapeCached)
	{
		return true;
	}

	ShapeCached = InShapeData;

	Shape.setZero();
	for (int32 Index = 0; Index < InShapeData.Num(); ++Index)
	{
		Shape(Index, 0) = InShapeData[Index];
	}

	const Eigen::Matrix<float, -1, -1> PoseBackup = Pose;
	SetPose(TArray<float>());

	bIsUpToDate = false;

	HeightOffset = 0.0;
	GetSkinnedVerticesSMPL();

	float YMin = SkinnedVertsSMPL[0].Y;
	for (const FVector& Vert : SkinnedVertsSMPL)
	{
		if (Vert.Y < YMin)
		{
			YMin = Vert.Y;
		}
	}

	HeightOffset = -YMin;

	Pose = PoseBackup;

	bIsUpToDate = false;

	return true;
}

bool FMetaHumanSMPLXImpl::SetPose(const TArray<float>& InPoseData)
{
	if (!bIsInitialized)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Not initialized");
		return false;
	}

	if (InPoseData.Num() > Pose.rows()) // Only specifying a sub-set of the pose is ok
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Too many pose data items, got %i expected at most %lli", InPoseData.Num(), Pose.rows());
		return false;
	}

	Pose.setZero();
	for (int32 Index = 0; Index < InPoseData.Num(); ++Index)
	{
		Pose(Index, 0) = InPoseData[Index];
	}

	bIsUpToDate = false;

	return true;
}

bool FMetaHumanSMPLXImpl::SetAccountForHeight(bool bInAccountForHeight)
{
	bAccountForHeight = bInAccountForHeight;

	bIsUpToDate = false;

	return true;
}

bool FMetaHumanSMPLXImpl::GetAccountForHeight() const
{
	return bAccountForHeight;
}

void FMetaHumanSMPLXImpl::Calculate()
{
	// Shaped vertices
	Eigen::Matrix<float, -1, -1> VertDeltas = ShapeBlendShapes * Shape; // ((NumVerts * 3) x 1) col vector 
	const Eigen::Map<Eigen::Matrix<float, -1, 3, Eigen::RowMajor>> VertDeltasReshaped(VertDeltas.data(), Verts.rows(), Verts.cols()); // (NumVerts x 3)
	Eigen::Matrix<float, -1, -1> ShapedVerts = Verts + VertDeltasReshaped;

	// Joint regression
	const Eigen::Matrix<float, -1, -1> Joints = JointReg * ShapedVerts;

	// Forward Kinematics

	// - Local Transforms
	TArray<Eigen::Matrix<float, 4, 4>> LocalTransforms;
	LocalTransforms.SetNumUninitialized(NumJoints);

	Eigen::Matrix<float, -1, 1> PoseBasisWeights;
	PoseBasisWeights.resize(9 * (NumJoints - 1), 1);
	int32 PoseBasisWeightsRow = 0;

	for (int32 Joint = 0; Joint < NumJoints; ++Joint)
	{
		Eigen::Vector3f Axis(Pose(Joint * 3), Pose(Joint * 3 + 1), Pose(Joint * 3 + 2));
		const float Angle = Axis.norm();

		Eigen::Matrix<float, 3, 3> R;

		if (Angle > 1e-4)
		{
			Axis.normalize();

			const Eigen::AngleAxisf AxisAngle(Angle, Axis);

			R = AxisAngle.toRotationMatrix();
		}
		else
		{
			R.setIdentity();
		}

		if (Joint > 0)
		{
			Eigen::Matrix<float, 3, 3> RMinusI = R - Eigen::Matrix<float, 3, 3>::Identity();

			for (int32 Row = 0; Row < 3; ++Row)
			{
				for (int32 Col = 0; Col < 3; ++Col)
				{
					PoseBasisWeights(PoseBasisWeightsRow++, 0) = RMinusI(Row, Col);
				}
			}
		}

		Eigen::Matrix<float, 4, 4> T;
		T.setIdentity();

		T.block<3, 3>(0, 0) = R; //-V521

		if (KinematicTree[Joint] == -1)
		{
			T(0, 3) = Joints(Joint, 0);
			T(1, 3) = Joints(Joint, 1);
			T(2, 3) = Joints(Joint, 2);
		}
		else
		{
			const int32 Parent = KinematicTree[Joint];
			T(0, 3) = Joints(Joint, 0) - Joints(Parent, 0);
			T(1, 3) = Joints(Joint, 1) - Joints(Parent, 1);
			T(2, 3) = Joints(Joint, 2) - Joints(Parent, 2);
		}

		LocalTransforms[Joint] = T;
	}

	// - World Transforms
	TArray<Eigen::Matrix<float, 4, 4>> WorldTransforms;
	WorldTransforms.SetNumUninitialized(NumJoints);

	for (int32 Joint = 0; Joint < NumJoints; ++Joint)
	{
		const int32 Parent = KinematicTree[Joint];

		if (Parent == -1)
		{
			WorldTransforms[Joint] = LocalTransforms[Joint];
		}
		else
		{
			WorldTransforms[Joint] = WorldTransforms[Parent] * LocalTransforms[Joint];
		}
	}

	// - For skinning, we need transformations relative to rest pose
	TArray<Eigen::Matrix<float, 4, 4>> SkinningMatrix;
	SkinningMatrix.SetNumUninitialized(NumJoints);

	const Eigen::Matrix<float, -1, -1> RestJoints = JointReg * ShapedVerts;

	for (int32 Joint = 0; Joint < NumJoints; ++Joint)
	{
		Eigen::Matrix<float, 4, 4> RestTransformInv;
		RestTransformInv.setIdentity();

		RestTransformInv(0, 3) = -RestJoints(Joint, 0);
		RestTransformInv(1, 3) = -RestJoints(Joint, 1);
		RestTransformInv(2, 3) = -RestJoints(Joint, 2);

		SkinningMatrix[Joint] = WorldTransforms[Joint] * RestTransformInv;
	}

	Eigen::Matrix<float, -1, -1> PoseDeltas = PoseBlendShapes * PoseBasisWeights;
	const Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> PoseDeltasReshaped(PoseDeltas.data(), NumVerts, 3);
	ShapedVerts += PoseDeltasReshaped;

	// Skinned verts
	SkinnedVertsSMPL.SetNumUninitialized(NumVerts);
	SkinnedVertsUE.SetNumUninitialized(NumVerts);

	for (int32 Vert = 0; Vert < NumVerts; ++Vert)
	{
		const Eigen::Vector4f InitialVertPos(ShapedVerts(Vert, 0), ShapedVerts(Vert, 1), ShapedVerts(Vert, 2), 1.0);
		Eigen::Vector4f FinalVertPos(0, 0, 0, 0);

		for (int32 Joint = 0; Joint < NumJoints; ++Joint)
		{
			const float Weight = Weights(Vert, Joint);
			if (Weight > 0)
			{
				FinalVertPos += Weight * (SkinningMatrix[Joint] * InitialVertPos);
			}
		}

		SkinnedVertsSMPL[Vert] = FVector(FinalVertPos(0), FinalVertPos(1), FinalVertPos(2));
		if (bAccountForHeight)
		{
			SkinnedVertsSMPL[Vert].Y += HeightOffset;
		}
		SkinnedVertsUE[Vert] = FMetaHumanSMPLX::SMPL2UE(SkinnedVertsSMPL[Vert]);
	}

	// Bone hierarchy
	BoneHierarchyUE.Reset(NumJoints + 1);
	BoneHierarchyUE.Add(TTuple<FString, int32, FTransform>("root", -1, FTransform::Identity));

	Eigen::Matrix<float, 4, 4> SMPL2UE; // convert SMPL (x, y, z) to UE (x, z, y). Meters to cm accounted for later.
	SMPL2UE.setZero();
	SMPL2UE(0, 0) = 1;
	SMPL2UE(1, 2) = 1;
	SMPL2UE(2, 1) = 1;
	SMPL2UE(3, 3) = 1;

	const Eigen::Matrix<float, 4, 4> SMPL2UETranspose = SMPL2UE.transpose();

	for (int32 Joint = 0; Joint < NumJoints; ++Joint)
	{
		const Eigen::Matrix<float, 4, 4> UE = SMPL2UE * LocalTransforms[Joint] * SMPL2UETranspose;

		FMatrix Matrix;
		for (int32 Row = 0; Row < 4; ++Row)
		{
			for (int32 Col = 0; Col < 4; ++Col)
			{
				Matrix.M[Col][Row] = UE(Row, Col);
			}
		}

		Matrix.M[3][0] *= 100;
		Matrix.M[3][1] *= 100;
		Matrix.M[3][2] *= 100;

		if (Joint == 0 && bAccountForHeight)
		{
			Matrix.M[3][2] += (HeightOffset * 100);
		}

		BoneHierarchyUE.Add(TTuple<FString, int32, FTransform>(JointNamesUE[Joint], KinematicTree[Joint] + 1, FTransform(Matrix)));
	}

	bIsUpToDate = true;
}

const TArray<FVector>& FMetaHumanSMPLXImpl::GetSkinnedVerticesSMPL()
{
	if (!bIsInitialized)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Not initialized");
	}

	if (!bIsUpToDate)
	{
		Calculate();
	}

	return SkinnedVertsSMPL;
}

const TArray<FVector>& FMetaHumanSMPLXImpl::GetSkinnedVerticesUE()
{
	if (!bIsInitialized)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Not initialized");
	}

	if (!bIsUpToDate)
	{
		Calculate();
	}

	return SkinnedVertsUE;
}

const TArray<TTuple<uint32, uint32, uint32>>& FMetaHumanSMPLXImpl::GetFaces() const
{
	if (!bIsInitialized)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Not initialized");
	}

	return Faces;
}

const TArray<TTuple<FString, int32, FTransform>>& FMetaHumanSMPLXImpl::GetBoneHierarchyUE()
{
	if (!bIsInitialized)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Not initialized");
	}

	if (!bIsUpToDate)
	{
		Calculate();
	}

	return BoneHierarchyUE;
}

float FMetaHumanSMPLXImpl::GetHeightOffset() const
{
	return HeightOffset;
}



FMetaHumanSMPLX::FMetaHumanSMPLX()
{
	Impl = MakeUnique<FMetaHumanSMPLXImpl>();
}

FMetaHumanSMPLX::~FMetaHumanSMPLX()
{
	Impl.Reset();
}

bool FMetaHumanSMPLX::Init(UMetaHumanSMPLXData* InData)
{
	if (!InData)
	{
		InData = LoadObject<UMetaHumanSMPLXData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/SMPLX_NEUTRAL_2020_locked_head_array_f32.SMPLX_NEUTRAL_2020_locked_head_array_f32"));
	}

	return Impl->Init(InData);
}

bool FMetaHumanSMPLX::IsInitialized() const
{
	return Impl->IsInitialized();
}

const UMetaHumanSMPLXData* FMetaHumanSMPLX::GetSmplxData() const
{
	return Impl->GetSmplxData();
}

bool FMetaHumanSMPLX::SetShape(const TArray<float>& InShapeData)
{
	return Impl->SetShape(InShapeData);
}

bool FMetaHumanSMPLX::SetPose(const TArray<float>& InPoseData)
{
	return Impl->SetPose(InPoseData);
}

bool FMetaHumanSMPLX::SetAccountForHeight(bool bInAccountForHeight)
{
	return Impl->SetAccountForHeight(bInAccountForHeight);
}

const TArray<FVector>& FMetaHumanSMPLX::GetSkinnedVerticesSMPL()
{
	return Impl->GetSkinnedVerticesSMPL();
}

const TArray<FVector>& FMetaHumanSMPLX::GetSkinnedVerticesUE()
{
	return Impl->GetSkinnedVerticesUE();
}

const TArray<TTuple<uint32, uint32, uint32>>& FMetaHumanSMPLX::GetFaces() const
{
	return Impl->GetFaces();
}

const TArray<TTuple<FString, int32, FTransform>>& FMetaHumanSMPLX::GetBoneHierarchyUE()
{
	return Impl->GetBoneHierarchyUE();
}

float FMetaHumanSMPLX::GetHeightOffset() const
{
	return Impl->GetHeightOffset();
}

FVector FMetaHumanSMPLX::SMPL2UE(const FVector& InVector)
{
	FVector Out = FVector(InVector.X, InVector.Z, InVector.Y); // Account for orientation and left/right coord system
	Out *= 100; // Meters to cm

	return Out;
}

FVector FMetaHumanSMPLX::UE2SMPL(const FVector& InVector)
{
	FVector Out = FVector(InVector.X, InVector.Z, InVector.Y); // Account for orientation and left/right coord system
	Out /= 100; // cm to meters

	return Out;
}

TArray<float> FMetaHumanSMPLX::InvalidShape()
{
	TArray<float> Shape;

	Shape.SetNumZeroed(FMetaHumanSMPLXImpl::NumShapes + 1);

	return Shape;
}

#if WITH_EDITOR
USkeleton* FMetaHumanSMPLX::CreateSMPLXSkeleton(UObject* InOuter)
{
	check(IsInGameThread());

#if 0
	USkeleton* Skeleton = NewObject<USkeleton>(InOuter, TEXT("SMPLXSkeleton"));

	FReferenceSkeletonModifier Modifier(Skeleton);

	for (const TTuple<FString, int32, FTransform>& Bone : GetBoneHierarchyUE())
	{
		const FMeshBoneInfo BoneInfo(FName(Bone.Get<0>()), Bone.Get<0>(), Bone.Get<1>());

		Modifier.Add(BoneInfo, Bone.Get<2>());
	}

	Skeleton->PostEditChange();

	return Skeleton;
#else
	return LoadObject<USkeleton>(InOuter, TEXT("/" UE_PLUGIN_NAME "/SKEL_SMPL.SKEL_SMPL"));
#endif
}

USkeleton* FMetaHumanSMPLX::GetMHSkeleton(UObject* InOuter)
{
	check(IsInGameThread());

	return LoadObject<USkeleton>(InOuter, TEXT("/" UE_PLUGIN_NAME "/metahuman_base_skel.metahuman_base_skel"));
}

USkeletalMesh* FMetaHumanSMPLX::CreateSMPLXSkeletalMesh(UObject* InOuter, USkeleton* InSMPLXSkeleton, bool bInIsSkinned)
{
	check(IsInGameThread());

#if 1
	FString Name = "SMPLXSkeletalMesh";
	Name += bInIsSkinned ? TEXT("Skinned") : TEXT("Bare");

	Name = MakeUniqueObjectName(InOuter, USkeletalMesh::StaticClass(), *Name).ToString();

	USkeletalMesh *SkeletalMesh = NewObject<USkeletalMesh>(InOuter, *Name);
	SkeletalMesh->SetSkeleton(InSMPLXSkeleton);

	FReferenceSkeleton RefSkeleton = InSMPLXSkeleton->GetReferenceSkeleton();

	{
		FReferenceSkeletonModifier Modifier(RefSkeleton, InSMPLXSkeleton); // Ensure this goes out of scope before using RefSkeleton in a skel mesh

		for (const TTuple<FString, int32, FTransform>& BoneTuple : GetBoneHierarchyUE())
		{
			const FString BoneName = BoneTuple.Get<0>();
			const int32 BoneIndex = RefSkeleton.FindBoneIndex(FName(BoneName));

			if (BoneIndex == INDEX_NONE)
			{
				UE_LOGF(LogMetaHumanBodyTracker, Error, "Bone %ls not found", *BoneName);
			}
			else
			{
				Modifier.UpdateRefPoseTransform(BoneIndex, FTransform(BoneTuple.Get<2>().GetLocation())); // Apparently its just translation you update! without this retargeting is off.
			}
		}
	}

	SkeletalMesh->SetRefSkeleton(RefSkeleton);
	SkeletalMesh->CalculateInvRefMatrices();

	if (bInIsSkinned)
	{
		FSkeletalMeshLODInfo& LODInfo = SkeletalMesh->AddLODInfo();
		LODInfo.BuildSettings.bRecomputeNormals = true;
		LODInfo.BuildSettings.bRecomputeTangents = true;

		FMeshDescription MeshDesc;
		FSkeletalMeshAttributes Attributes(MeshDesc);
		Attributes.Register();

		TVertexAttributesRef<FVector3f> Positions = Attributes.GetVertexPositions();
		FSkinWeightsVertexAttributesRef SkinWeights = Attributes.GetVertexSkinWeights();
		TPolygonGroupAttributesRef<FName> MaterialSlotNames = Attributes.GetPolygonGroupMaterialSlotNames();

		TArray<FVertexID> VertexIDs;
		int32 VertCount = 0;
		const TArray<TTuple<FString, int32, FTransform>>& BoneHierarchyUE = GetBoneHierarchyUE();

		for (const FVector& Vert : GetSkinnedVerticesUE())
		{
			const FVertexID VID = MeshDesc.CreateVertex();
			Positions[VID] = FVector3f(Vert);
			VertexIDs.Add(VID);

			TArray<UE::AnimationCore::FBoneWeight> BoneWeights;

			for (int32 Joint = 0; Joint < Impl->NumJoints; ++Joint)
			{
				const FString BoneName = BoneHierarchyUE[Joint + 1].Get<0>(); // +1 to account for root bone added to the UE skeleton thats not present in the SMPL skeleton
				const int32 BoneIndex = RefSkeleton.FindBoneIndex(FName(BoneName));

				if (BoneIndex == INDEX_NONE)
				{
					UE_LOGF(LogMetaHumanBodyTracker, Error, "Bone %ls not found", *BoneName);
				}
				else
				{
					UE::AnimationCore::FBoneWeight BoneWeight;
					BoneWeight.SetBoneIndex(BoneIndex);
					BoneWeight.SetWeight(Impl->Weights(VertCount, Joint));
					BoneWeights.Add(BoneWeight);
				}
			}

			SkinWeights.Set(VID, BoneWeights);

			VertCount++;
		}

		const FPolygonGroupID PolyGroupID = MeshDesc.CreatePolygonGroup();
		MaterialSlotNames[PolyGroupID] = FName("DefaultMaterial");

		for (const TTuple<uint32, uint32, uint32>& Face : GetFaces())
		{
			const FVertexInstanceID V1 = MeshDesc.CreateVertexInstance(VertexIDs[Face.Get<0>()]);
			const FVertexInstanceID V2 = MeshDesc.CreateVertexInstance(VertexIDs[Face.Get<1>()]);
			const FVertexInstanceID V3 = MeshDesc.CreateVertexInstance(VertexIDs[Face.Get<2>()]);

			TArray<FVertexInstanceID> Triangle;

			Triangle.Add(V1);
			Triangle.Add(V2);
			Triangle.Add(V3);

			MeshDesc.CreateTriangle(PolyGroupID, Triangle);
		}

		FSkeletalMeshModel* ImportedModel = SkeletalMesh->GetImportedModel();
		ImportedModel->LODModels.Add(new FSkeletalMeshLODModel());

		SkeletalMesh->CreateMeshDescription(0, MoveTemp(MeshDesc));
		SkeletalMesh->CommitMeshDescription(0);

		FSkeletalMaterial MaterialSlot(nullptr, FName("DefaultMaterial"));
		SkeletalMesh->GetMaterials().Add(MaterialSlot);

		SkeletalMesh->PostEditChange();
	}
	else
	{
		PopulateSkeletalMeshBare(InOuter, SkeletalMesh, false);
	}

	return SkeletalMesh;
#else
	if (bInIsSkinned)
	{
		return LoadObject<USkeletalMesh>(InOuter, TEXT("/Game/Export/SMPLXSkeletalMeshSkinned.SMPLXSkeletalMeshSkinned"));
	}
	else
	{
		return LoadObject<USkeletalMesh>(InOuter, TEXT("/Game/Export/SMPLXSkeletalMeshBare.SMPLXSkeletalMeshBare"));
	}
#endif
}

USkeletalMesh* FMetaHumanSMPLX::CreateMHSkeletalMesh(UObject* InOuter, USkeleton* InMHSkeleton, USkeleton* InSMPLXSkeleton)
{
	check(IsInGameThread());

#if 1
	FString Name = "MHSkeletalMesh";
	Name += InSMPLXSkeleton ? TEXT("Sized") : TEXT("Archetype");

	Name = MakeUniqueObjectName(InOuter, USkeletalMesh::StaticClass(), *Name).ToString();

	USkeletalMesh* SkeletalMesh = NewObject<USkeletalMesh>(InOuter, *Name);
	SkeletalMesh->SetSkeleton(InMHSkeleton);
	
	FReferenceSkeleton RefSkeleton = InMHSkeleton->GetReferenceSkeleton();

	if (InSMPLXSkeleton)
	{
		// Calculate the global position (in SMPL coords) of each joint in the the SMPL skeleton
		bool bAccountForHeight = Impl->GetAccountForHeight();
		Impl->SetAccountForHeight(false);
		const TArray<TTuple<FString, int32, FTransform>>& Bones = Impl->GetBoneHierarchyUE(); // These are local transforms relative to parent in UE cords
		Impl->SetAccountForHeight(bAccountForHeight);

		TArray<FTransform> GlobalSMPLJointTransform;
		GlobalSMPLJointTransform.SetNumUninitialized(Bones.Num());
		GlobalSMPLJointTransform[0] = FTransform::Identity; // Root

		TArray<FVector> GlobalSMPLJointPosition;
		for (int32 BoneIndex = 1; BoneIndex < Bones.Num(); ++BoneIndex)
		{
			const int32 ParentBoneIndex = Bones[BoneIndex].Get<1>();
			const FTransform LocalTransform = Bones[BoneIndex].Get<2>();

			check(BoneIndex > ParentBoneIndex);
			GlobalSMPLJointTransform[BoneIndex] = LocalTransform * GlobalSMPLJointTransform[ParentBoneIndex];

			GlobalSMPLJointPosition.Add(FMetaHumanSMPLX::UE2SMPL(GlobalSMPLJointTransform[BoneIndex].GetLocation()));
		}

		// Calculate the  global position (in SMPL coords) of a subset of joints in the the MH skeleton that match the SMPL skeleton
		TMap<FString, TPair<FString, FVector>> GlobalMHJointPosition = UE::MetaHuman::BodyTracker::FMetaHumanResize::Resize(Impl->Shape[0], GlobalSMPLJointPosition);

		// Modify the MH skeleton with these joints. Joints need to be updated as a local, not global, transform in UE coords.
		// Cant apply the global joints directly as this will effect the MH's pose. The SMPL is T pose, the MH A pose.
		// So instead update the bone length of the MH rather than absolute position. Keep proportions basically.

		FReferenceSkeletonModifier Modifier(RefSkeleton, InMHSkeleton); // Ensure this goes out of scope before using RefSkeleton in a skel mesh
		TArray<FTransform> LocalPose = RefSkeleton.GetRefBonePose();

		for (int32 BoneIndex = 0; BoneIndex < LocalPose.Num(); ++BoneIndex)
		{
			const FString BoneName = RefSkeleton.GetBoneName(BoneIndex).ToString();
			const int32 ParentBoneIndex = RefSkeleton.GetParentIndex(BoneIndex);
			const FString ParentBoneName = ParentBoneIndex == -1 ? "" : RefSkeleton.GetBoneName(ParentBoneIndex).ToString();

			if (GlobalMHJointPosition.Contains(BoneName) && GlobalMHJointPosition.Contains(ParentBoneName))
			{
				const FVector P1 = GlobalMHJointPosition[BoneName].Value;
				const FVector P2 = GlobalMHJointPosition[ParentBoneName].Value;

				const float Length = (P1 - P2).Length();

				FVector Local = LocalPose[BoneIndex].GetLocation();

				Local.Normalize();
				Local *= Length * 100; // SMPL meter to UE cm

				LocalPose[BoneIndex].SetLocation(Local);

				Modifier.UpdateRefPoseTransform(BoneIndex, LocalPose[BoneIndex]);
			}
		}
	}

	SkeletalMesh->SetRefSkeleton(RefSkeleton);
	SkeletalMesh->CalculateInvRefMatrices();

	PopulateSkeletalMeshBare(InOuter, SkeletalMesh, true);

	return SkeletalMesh;
#else
	if (InSMPLXSkeleton)
	{
		return LoadObject<USkeletalMesh>(InOuter, TEXT("/Game/Export/MHSkeletalMeshSizedBare.MHSkeletalMeshSizedBare"));
	}
	else
	{
		return LoadObject<USkeletalMesh>(InOuter, TEXT("/MetaHumanCharacter/Body/IdentityTemplate/SKM_Body.SKM_Body"));
	}
#endif
}

static FMeshDescription MakeCubeMeshDescription(const FVector& InExtent = FVector(50.0f), const FVector& InCenter = FVector::ZeroVector)
{
	FMeshDescription MeshDesc;
	FStaticMeshAttributes Attributes(MeshDesc);
	Attributes.Register();

	TVertexAttributesRef<FVector3f>         VertexPositions = Attributes.GetVertexPositions();
	TVertexInstanceAttributesRef<FVector3f> VINormals = Attributes.GetVertexInstanceNormals();
	TVertexInstanceAttributesRef<FVector2f> VIUVs = Attributes.GetVertexInstanceUVs();
	TVertexInstanceAttributesRef<FVector4f> VIColors = Attributes.GetVertexInstanceColors();

	MeshDesc.ReserveNewVertices(36);
	MeshDesc.ReserveNewVertexInstances(36);
	MeshDesc.ReserveNewTriangles(12);

	const FVector3f C(InCenter);
	const FVector3f E(InExtent);

	// 8 corner positions (copied into independent vertices per triangle).
	const FVector3f Corners[8] = {
		C + FVector3f(-E.X, -E.Y, -E.Z), // 0
		C + FVector3f(E.X, -E.Y, -E.Z), // 1
		C + FVector3f(E.X,  E.Y, -E.Z), // 2
		C + FVector3f(-E.X,  E.Y, -E.Z), // 3
		C + FVector3f(-E.X, -E.Y,  E.Z), // 4
		C + FVector3f(E.X, -E.Y,  E.Z), // 5
		C + FVector3f(E.X,  E.Y,  E.Z), // 6
		C + FVector3f(-E.X,  E.Y,  E.Z), // 7
	};

	const FPolygonGroupID PolyGroup = MeshDesc.CreatePolygonGroup();
	Attributes.GetPolygonGroupMaterialSlotNames()[PolyGroup] = FName(TEXT("DefaultMaterial"));

	// 12 triangles defined as corner-index triplets (CW when viewed from outside in Unreal LH coords),
	// paired with the outward face normal and per-corner UVs.
	struct FTri { int32 V[3]; FVector3f Normal; FVector2f UV[3]; };

	const FVector2f UV_BL(0.f, 0.f);
	const FVector2f UV_BR(1.f, 0.f);
	const FVector2f UV_TR(1.f, 1.f);
	const FVector2f UV_TL(0.f, 1.f);

	const FTri Tris[12] = {
		// Bottom (-Z): quad {1,2,3,0}
		{ {1, 3, 0}, FVector3f(0, 0, -1), { UV_BL, UV_TR, UV_BR } },
		{ {1, 2, 3}, FVector3f(0, 0, -1), { UV_BL, UV_TL, UV_TR } },
		// Top (+Z): quad {4,6,5,7}
		{ {4, 6, 5}, FVector3f(0, 0,  1), { UV_BL, UV_TR, UV_BR } },
		{ {4, 7, 6}, FVector3f(0, 0,  1), { UV_BL, UV_TL, UV_TR } },
		// Front (-Y): quad {0,5,1,4}
		{ {0, 5, 1}, FVector3f(0, -1, 0), { UV_BL, UV_TR, UV_BR } },
		{ {0, 4, 5}, FVector3f(0, -1, 0), { UV_BL, UV_TL, UV_TR } },
		// Back (+Y): quad {2,7,3,6}
		{ {2, 7, 3}, FVector3f(0,  1, 0), { UV_BL, UV_TR, UV_BR } },
		{ {2, 6, 7}, FVector3f(0,  1, 0), { UV_BL, UV_TL, UV_TR } },
		// Left (-X): quad {3,4,0,7}
		{ {3, 4, 0}, FVector3f(-1, 0, 0), { UV_BL, UV_TR, UV_BR } },
		{ {3, 7, 4}, FVector3f(-1, 0, 0), { UV_BL, UV_TL, UV_TR } },
		// Right (+X): quad {1,6,2,5}
		{ {1, 6, 2}, FVector3f(1, 0, 0),  { UV_BL, UV_TR, UV_BR } },
		{ {1, 5, 6}, FVector3f(1, 0, 0),  { UV_BL, UV_TL, UV_TR } },
	};

	for (const FTri& Tri : Tris)
	{
		FVertexInstanceID VI[3];
		for (int32 Index = 0; Index < 3; ++Index)
		{
			const FVertexID V = MeshDesc.CreateVertex();
			VertexPositions[V] = Corners[Tri.V[Index]];

			VI[Index] = MeshDesc.CreateVertexInstance(V);
			VINormals[VI[Index]] = Tri.Normal;
			VIUVs[VI[Index]] = Tri.UV[Index];
			VIColors[VI[Index]] = FVector4f(1.f, 1.f, 1.f, 1.f);
		}

		MeshDesc.CreateTriangle(PolyGroup, TArrayView<const FVertexInstanceID>(VI, 3));
	}

	return MeshDesc;
}

bool FMetaHumanSMPLX::PopulateSkeletalMeshBare(UObject* InOuter, USkeletalMesh* InSkeletalMesh, bool bInIsMHSkelMesh)
{
	FSkeletalMeshLODInfo& LODInfo = InSkeletalMesh->AddLODInfo();
	LODInfo.BuildSettings.bRecomputeNormals = true;
	LODInfo.BuildSettings.bRecomputeTangents = true;
	LODInfo.ReductionSettings.TerminationCriterion = SMTC_AbsNumOfTriangles;
	LODInfo.ReductionSettings.NumOfTrianglesPercentage = 1.0f;
	LODInfo.ReductionSettings.MaxNumOfTriangles = MAX_uint32;
	LODInfo.ReductionSettings.MaxNumOfTrianglesPercentage = MAX_uint32;
	LODInfo.ReductionSettings.NumOfVertPercentage = 1.0f;
	LODInfo.ReductionSettings.MaxNumOfVerts = MAX_uint32;
	LODInfo.bAllowCPUAccess = true;

	FMeshDescription MeshDesc;
	FSkeletalMeshAttributes Attributes(MeshDesc);
	Attributes.Register();

	FSkinWeightsVertexAttributesRef SkinWeights = Attributes.GetVertexSkinWeights();

	const FReferenceSkeleton& RefSkeleton = InSkeletalMesh->GetRefSkeleton();
	TArray<FTransform> BonePoses;
	RefSkeleton.GetBoneAbsoluteTransforms(BonePoses);

	for (int32 BoneIndex = 0; BoneIndex < RefSkeleton.GetNum(); ++BoneIndex)
	{
		const int32 ParentBoneIndex = RefSkeleton.GetParentIndex(BoneIndex);

		if (ParentBoneIndex >= 0)
		{
			const FString BoneName = RefSkeleton.GetBoneName(BoneIndex).ToString();

			// Dont show every bone. This reduces visual clutter.
			if (bInIsMHSkelMesh)
			{
				// MH case - dont show bones with no children, except for select bones at the end of bone chain
				if (!(BoneName == "head" ||
					  BoneName.StartsWith("bigtoe_02_") ||
					  BoneName.StartsWith("indextoe_02_") ||
					  BoneName.StartsWith("middletoe_02_") ||
					  BoneName.StartsWith("ringtoe_02_") ||
					  BoneName.StartsWith("littletoe_02_")))
				{
					TArray<int32> ChildBones;
					if (RefSkeleton.GetDirectChildBones(BoneIndex, ChildBones) == 0)
					{
						continue;
					}
				}
			}
			else
			{
				// SMPLX case - dont show IK bones
				if (BoneName.StartsWith("ik_"))
				{
					continue;
				}
			}

			FMeshDescription PrimativeMeshDesc = MakeCubeMeshDescription();

			TVertexAttributesRef<FVector3f> Positions = PrimativeMeshDesc.GetVertexPositions();

			const FVector BoneLocation = BonePoses[BoneIndex].GetLocation();
			const FVector ParentBoneLocation = BonePoses[ParentBoneIndex].GetLocation();
			FVector BoneToParentVector = ParentBoneLocation - BoneLocation;

			const float BoneLength = BoneToParentVector.Length();

			if (BoneLength < 0.01)
			{
				continue;
			}

			BoneToParentVector.Normalize();

			const FTransform BoneToParentAlignment = FTransform(FQuat::FindBetweenVectors(FVector(0, 0, 1), BoneToParentVector));

			TArray<float> Weights;

			for (const FVertexID& VertexID : PrimativeMeshDesc.Vertices().GetElementIDs())
			{
				FVector Vertex(Positions[VertexID]);

				Vertex.Z += 50;
				Vertex.X *= 0.01;
				Vertex.Y *= 0.01;

				Weights.Add(Vertex.Z / 100.0f);

				if (BoneName.StartsWith("index_") || // Reduce size of finger bones for clarity - same for MH and SMPLX cases
					BoneName.StartsWith("middle_") ||
					BoneName.StartsWith("pinky_") ||
					BoneName.StartsWith("ring_") ||
					BoneName.StartsWith("thumb_"))
				{
					Vertex.X *= 0.5;
					Vertex.Y *= 0.5;
				}

				Vertex.Z *= BoneLength / 100;

				Vertex = BoneToParentAlignment.TransformPosition(Vertex);
				Vertex += BoneLocation;

				Positions[VertexID] = FVector3f(Vertex);
			}

			TArray<FVertexID> VerticesBeforeAppend, VerticesAfterAppend;

			for (const FVertexID& VertexId : MeshDesc.Vertices().GetElementIDs())
			{
				VerticesBeforeAppend.Add(VertexId);
			}

			FStaticMeshOperations::FAppendSettings AppendSetting;
			FStaticMeshOperations::AppendMeshDescription(PrimativeMeshDesc, MeshDesc, AppendSetting);

			for (const FVertexID& VertexId : MeshDesc.Vertices().GetElementIDs())
			{
				VerticesAfterAppend.Add(VertexId);
			}

			for (int32 Index = VerticesBeforeAppend.Num(); Index < VerticesAfterAppend.Num(); ++Index)
			{
				if (BoneName == "pelvis") // pelvis to root bone needs to be anchored at both ends since bone length changes
				{
					UE::AnimationCore::FBoneWeight BoneWeight1;
					BoneWeight1.SetBoneIndex(ParentBoneIndex);
					BoneWeight1.SetWeight(Weights[Index - VerticesBeforeAppend.Num()]);

					UE::AnimationCore::FBoneWeight BoneWeight2;
					BoneWeight2.SetBoneIndex(BoneIndex);
					BoneWeight2.SetWeight(1 - Weights[Index - VerticesBeforeAppend.Num()]);

					SkinWeights.Set(VerticesAfterAppend[Index], { BoneWeight1, BoneWeight2 });
				}
				else // other bones, which dont change length, needs to be anchored at only one end to avoid twisting/distortion of cube
				{
					UE::AnimationCore::FBoneWeight BoneWeight;
					BoneWeight.SetBoneIndex(ParentBoneIndex);
					BoneWeight.SetWeight(1);

					SkinWeights.Set(VerticesAfterAppend[Index], { BoneWeight });
				}
			}
		}
	}

	FSkeletalMeshModel* ImportedModel = InSkeletalMesh->GetImportedModel();
	ImportedModel->LODModels.Add(new FSkeletalMeshLODModel());

	InSkeletalMesh->CreateMeshDescription(0, MoveTemp(MeshDesc));
	InSkeletalMesh->CommitMeshDescription(0);

	FSkeletalMaterial MaterialSlot(nullptr, FName("DefaultMaterial"));
	InSkeletalMesh->GetMaterials().Add(MaterialSlot);

	InSkeletalMesh->MarkPackageDirty();

	InSkeletalMesh->PostEditChange();

	return true;
}

UIKRetargeter* FMetaHumanSMPLX::CreateRetargeter(UObject* InOuter, USkeletalMesh* InSMPLXSkeletalMesh, USkeletalMesh* InMHSkeletalMesh)
{
	check(IsInGameThread());

#if 0
	UIKRigDefinition* SMPLXIKRig = NewObject<UIKRigDefinition>(InOuter, TEXT("SMPLXIKRig"));
	UIKRigController* SMPLXIKRigController = UIKRigController::GetController(SMPLXIKRig);
	SMPLXIKRigController->SetSkeletalMesh(InSMPLXSkeletalMesh);

	UIKRigDefinition* MHIKRig = NewObject<UIKRigDefinition>(InOuter, TEXT("MHIKRig"));
	UIKRigController* MHIKRigController = UIKRigController::GetController(MHIKRig);
	MHIKRigController->SetSkeletalMesh(InMHSkeletalMesh);

	UIKRetargeter* Retargeter = NewObject<UIKRetargeter>(InOuter, TEXT("SMPLX2MHRetargetter"));

	Retargeter->SourceIKRigAsset = SMPLXIKRig;
	Retargeter->TargetIKRigAsset = MHIKRig;

	return Retargeter;
#else
	return LoadObject<UIKRetargeter>(InOuter, TEXT("/" UE_PLUGIN_NAME "/RTG_SMPL_MH.RTG_SMPL_MH"));
#endif
}

#endif
