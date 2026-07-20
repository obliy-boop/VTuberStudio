// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "FrameAnimationData.h"

#include "Engine/SkeletalMesh.h"
#include "Animation/Skeleton.h"

#include "Retargeter/IKRetargeter.h"
#include "Retargeter/IKRetargetProcessor.h"

class FMetaHumanSMPLX;



namespace UE::MetaHuman::Pipeline
{

class FBaseBodyTracker
{

public:

	bool bOriginSet = false;
	FVector Origin;

protected:

	FBaseBodyTracker();

	TStrongObjectPtr<USkeleton> SMPLXSkeleton;
	TStrongObjectPtr<USkeletalMesh> SMPLXSkeletalMesh;

	TStrongObjectPtr<USkeleton> MHSkeleton;
	TStrongObjectPtr<USkeletalMesh> MHSkeletalMesh;

	TStrongObjectPtr<UIKRetargeter> Retargeter;

	FIKRetargetProcessor RetargetProcessor;

	void PrepareOutput(FFrameAnimationData& InOutAnimation);

	TPimplPtr<class FMetaHumanSMPLX> SMPLX;
};

}