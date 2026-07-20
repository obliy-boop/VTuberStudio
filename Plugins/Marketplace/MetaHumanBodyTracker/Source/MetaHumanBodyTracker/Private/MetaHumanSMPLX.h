// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Engine/DataAsset.h"
#include "Retargeter/IKRetargeter.h"

#include "MetaHumanSMPLX.generated.h"



UCLASS(MinimalAPI, DisplayName = "MetaHuman SMPLX Data")
class UMetaHumanSMPLXData : public UDataAsset
{
public:

	GENERATED_BODY()

	UPROPERTY()
	int32 Version = 0;

	UPROPERTY()
	TArray<float> Verts;

	UPROPERTY()
	TArray<uint32> Faces;

	UPROPERTY()
	TArray<float> JointReg;

	UPROPERTY()
	TArray<float> Weights;

	UPROPERTY()
	TArray<float> BlendShapes;
};

class FMetaHumanSMPLX
{

public:

	FMetaHumanSMPLX();
	~FMetaHumanSMPLX();

	bool Init(UMetaHumanSMPLXData* InData = nullptr);
	bool IsInitialized() const;
	const UMetaHumanSMPLXData* GetSmplxData() const;

	// Set the Shape (Beta) and Pose (Theta) parameters of the SMPLX model - see https://github.com/vchoutas/smplx for more detail
	bool SetShape(const TArray<float>& InShapeData);
	bool SetPose(const TArray<float>& InPoseData);

	bool SetAccountForHeight(bool bInAccountForHeight);

	const TArray<FVector>& GetSkinnedVerticesSMPL();
	const TArray<FVector>& GetSkinnedVerticesUE();
	const TArray<TTuple<uint32, uint32, uint32>>& GetFaces() const;
	const TArray<TTuple<FString, int32, FTransform>>& GetBoneHierarchyUE();
	float GetHeightOffset() const;

	static FVector SMPL2UE(const FVector& InVector);
	static FVector UE2SMPL(const FVector& InVector);
	static TArray<float> InvalidShape();

#if WITH_EDITOR
	USkeleton* CreateSMPLXSkeleton(UObject *InOuter);
	USkeleton* GetMHSkeleton(UObject* InOuter);

	USkeletalMesh* CreateSMPLXSkeletalMesh(UObject* InOuter, USkeleton* InSMPLXSkeleton, bool bInIsSkinned);
	USkeletalMesh* CreateMHSkeletalMesh(UObject* InOuter, USkeleton* InMHSkeleton, USkeleton* InSMPLXSkeleton);

	bool PopulateSkeletalMeshBare(UObject* InOuter, USkeletalMesh* InSkeletalMesh, bool bInIsMHSkelMesh);

	UIKRetargeter* CreateRetargeter(UObject* InOuter, USkeletalMesh* InSMPLXSkeletalMesh, USkeletalMesh* InMHSkeletalMesh);
#endif

private:

	TUniquePtr<class FMetaHumanSMPLXImpl> Impl;
};
