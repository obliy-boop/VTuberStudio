// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "MetaHumanBodyTrackerInterface.h"
#include "MetaHumanSMPLX.h"

#include "Components/DynamicMeshComponent.h"
#include "Components/PoseableMeshComponent.h"

#include "MetaHumanBodyDriverActor.generated.h"


UCLASS()
class AMetaHumanBodyDriverActor : public AMetaHumanBodyDriverActorInterface
{
	GENERATED_BODY()

public:

	AMetaHumanBodyDriverActor();

	virtual void PostInitializeComponents() override;

	virtual void Update(const FFrameAnimationData& InAnimationData) override;

	virtual void SetBodyDriverSkeletalMeshComponent(TWeakObjectPtr<USkeletalMeshComponent> InSkeletalMeshComponent) override;

	virtual FVector GetDebugRelativeLocation() const override { return FVector(0, 0, 0); }
	virtual FRotator GetDebugRelativeRotation() const override { return FRotator(0, 90, 0); }
	virtual FVector GetCameraRelativeLocation() const override { return FVector(500.0, 0.0, -100.0); }

	virtual void SetVisualizationSkeletalMeshParams(bool bInVisible, const FVector& InOffset, const FColor& InColour) override;

private:

	TArray<float> CachedSMPLXShape;

	UPROPERTY()
	TObjectPtr<class UStaticMeshComponent> OriginComponent;

	UPROPERTY()
	TObjectPtr<class UStaticMeshComponent> GroundplaneComponent;

	UPROPERTY()
	TObjectPtr<UDynamicMeshComponent> SMPLXDynamicMeshComponent;

	UPROPERTY()
	TObjectPtr<USkeleton> SMPLXSkeleton;

	UPROPERTY()
	TObjectPtr<UPoseableMeshComponent> SMPLXPoseableMeshSkinnedComponent;

	UPROPERTY()
	TObjectPtr<USkeletalMesh> SMPLXSkeletalMeshBare;

	UPROPERTY()
	TObjectPtr<UPoseableMeshComponent> SMPLXPoseableMeshBareComponent;

	UPROPERTY()
	TObjectPtr<USkeleton> MHSkeleton;

	UPROPERTY()
	TObjectPtr<USkeletalMesh> MHSkeletalMeshArchetypeBare;

	UPROPERTY()
	TObjectPtr<UPoseableMeshComponent> MHPoseableMeshArchetypeBareComponent;

	UPROPERTY()
	TObjectPtr<UPoseableMeshComponent> MHPoseableMeshSizedBareComponent;

	TWeakObjectPtr<USkeletalMeshComponent> BodyDriverSkeletalMeshComponent;

	TMap<FString, FTransform> SetBoneTransforms(UPoseableMeshComponent* InPoseableMeshComponent, const TMap<FString, FTransform>& InBoneTransforms);
	void SetDynamicMeshVerts(const FVector& InTranslation);

	FMetaHumanSMPLX SMPLX;

	void SaveDebugAssetsCVarChanged(IConsoleVariable* InVariable);
	bool SaveDebugAssets(const FString& InPath);
	void ShowDebugAssetsCVarChanged(IConsoleVariable* InVariable);
};
