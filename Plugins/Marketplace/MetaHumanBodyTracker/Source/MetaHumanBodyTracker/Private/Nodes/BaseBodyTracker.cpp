// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/BaseBodyTracker.h"
#include "MetaHumanSMPLX.h"
#include "MetaHumanBodyTrackerLog.h"

#include "UObject/Package.h"



namespace UE::MetaHuman::Pipeline
{

FBaseBodyTracker::FBaseBodyTracker()
{
	SMPLX = MakePimpl<FMetaHumanSMPLX>();

	if (!SMPLX->Init())
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to initialize SMPLX");
	}
}

void FBaseBodyTracker::PrepareOutput(FFrameAnimationData& InOutAnimation)
{
	SMPLX->SetShape(InOutAnimation.RawBodyAnimationSMPLXShape);
	SMPLX->SetPose(InOutAnimation.RawBodyAnimationSMPLXPose);

#if WITH_EDITOR
	if (!SMPLXSkeleton)
	{
		TFunction<void()> CreateAssetsFn = [this]()
		{
			SMPLXSkeleton = TStrongObjectPtr<USkeleton>(SMPLX->CreateSMPLXSkeleton(GetTransientPackage()));
			MHSkeleton = TStrongObjectPtr<USkeleton>(SMPLX->GetMHSkeleton(GetTransientPackage()));

			SMPLXSkeletalMesh = TStrongObjectPtr<USkeletalMesh>(SMPLX->CreateSMPLXSkeletalMesh(GetTransientPackage(), SMPLXSkeleton.Get(), false));
			MHSkeletalMesh = TStrongObjectPtr<USkeletalMesh>(SMPLX->CreateMHSkeletalMesh(GetTransientPackage(), MHSkeleton.Get(), SMPLXSkeleton.Get()));

			Retargeter = TStrongObjectPtr<UIKRetargeter>(SMPLX->CreateRetargeter(GetTransientPackage(), SMPLXSkeletalMesh.Get(), MHSkeletalMesh.Get()));

			FRetargetInitParameters RetargetInitParameters;
			RetargetInitParameters.SourceSkeletalMesh = SMPLXSkeletalMesh.Get();
			RetargetInitParameters.TargetSkeletalMesh = MHSkeletalMesh.Get();
			RetargetInitParameters.RetargeterAsset = Retargeter.Get();

			RetargetProcessor.Initialize(RetargetInitParameters);
		};

		if (IsInGameThread())
		{
			CreateAssetsFn();
		}
		else
		{ 
			FGraphEventRef GameThreadTask = FFunctionGraphTask::CreateAndDispatchWhenReady(CreateAssetsFn, TStatId(), nullptr, ENamedThreads::GameThread);
			FTaskGraphInterface::Get().WaitUntilTaskCompletes(GameThreadTask);
		}
	}
#endif

	// For now make horizonal translations relative to first frame

	if (!bOriginSet)
	{
		bOriginSet = true;
		Origin = InOutAnimation.RawBodyAnimationSMPLXTranslation;
	}

	InOutAnimation.RawBodyAnimationSMPLXTranslation.X -= Origin.X;
	InOutAnimation.RawBodyAnimationSMPLXTranslation.Z -= Origin.Z;

	InOutAnimation.RawBodyAnimationSMPLXTranslation = FMetaHumanSMPLX::SMPL2UE(InOutAnimation.RawBodyAnimationSMPLXTranslation);

	const TArray<TTuple<FString, int32, FTransform>>& BoneHierarchyUE = SMPLX->GetBoneHierarchyUE();

	for (const TTuple<FString, int32, FTransform>& BoneTuple : BoneHierarchyUE)
	{
		InOutAnimation.RawBodyAnimationSMPLXData.Add(BoneTuple.Get<0>(), BoneTuple.Get<2>());
	}

	if (InOutAnimation.RawBodyAnimationSMPLXData.Contains("root"))
	{
		FTransform RootTransform = InOutAnimation.RawBodyAnimationSMPLXData["root"];

		RootTransform.SetLocation(RootTransform.GetLocation() + InOutAnimation.RawBodyAnimationSMPLXTranslation - FMetaHumanSMPLX::SMPL2UE(FVector(0, SMPLX->GetHeightOffset(), 0)));

		InOutAnimation.RawBodyAnimationSMPLXData["root"] = RootTransform;
	}

	const FReferenceSkeleton& RefSkeleton = SMPLXSkeleton->GetReferenceSkeleton();
	TArray<FTransform> SMPLXComponentBoneTransformArray = RefSkeleton.GetRefBonePose();

	for (const TTuple<FString, int32, FTransform>& BoneTuple : BoneHierarchyUE)
	{
		const FString BoneName = BoneTuple.Get<0>();
		const int32 BoneIndex = RefSkeleton.FindBoneIndex(FName(BoneName));

		if (BoneIndex == INDEX_NONE)
		{
			UE_LOGF(LogMetaHumanBodyTracker, Error, "Bone %ls not found", *BoneName);
		}
		else
		{
			const int32 ParentBoneIndex = RefSkeleton.GetParentIndex(BoneIndex);

			if (ParentBoneIndex == INDEX_NONE)
			{
				FTransform RootTransform = BoneTuple.Get<2>();

				RootTransform.SetLocation(RootTransform.GetLocation() + InOutAnimation.RawBodyAnimationSMPLXTranslation - FMetaHumanSMPLX::SMPL2UE(FVector(0, SMPLX->GetHeightOffset(), 0)));

				SMPLXComponentBoneTransformArray[BoneIndex] = RootTransform;
			}
			else
			{
				check(BoneIndex > ParentBoneIndex);
				SMPLXComponentBoneTransformArray[BoneIndex] = BoneTuple.Get<2>() * SMPLXComponentBoneTransformArray[ParentBoneIndex];
			}
		}
	}

	FRetargetRunParameters RetargetRunParameters;
	RetargetRunParameters.SourceGlobalPose = &SMPLXComponentBoneTransformArray;

	const TArray<FTransform> MHComponentBoneTransformArray = RetargetProcessor.RunRetargeter(RetargetRunParameters);

	const FReferenceSkeleton& MHRefSkeleton = MHSkeletalMesh->GetRefSkeleton();
	for (int32 BoneIndex = 0; BoneIndex < MHRefSkeleton.GetNum(); ++BoneIndex)
	{
		const FString BoneName = MHRefSkeleton.GetBoneName(BoneIndex).ToString();
		const int32 ParentBoneIndex = MHRefSkeleton.GetParentIndex(BoneIndex);

		if (ParentBoneIndex == INDEX_NONE)
		{
			InOutAnimation.BodyAnimationData.Add(BoneName, MHComponentBoneTransformArray[BoneIndex]);
		}
		else
		{
			check(BoneIndex > ParentBoneIndex);
			InOutAnimation.BodyAnimationData.Add(BoneName, MHComponentBoneTransformArray[BoneIndex] * MHComponentBoneTransformArray[ParentBoneIndex].Inverse());
		}
	}
}

}
