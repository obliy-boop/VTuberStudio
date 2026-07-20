// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/BodyFaceAnimMergeNode.h"

#include "MetaHumanBodyTrackerLog.h"



namespace UE::MetaHuman::Pipeline
{

FBodyFaceAnimMergeNode::FBodyFaceAnimMergeNode(const FString& InName) : FNode("BodyFaceAnimMerge", InName)
{
	Pins.Add(FPin("Face Animation In", EPinDirection::Input, EPinType::Animation, 0));
	Pins.Add(FPin("Body Animation In", EPinDirection::Input, EPinType::Animation, 1));
	Pins.Add(FPin("Animation Out", EPinDirection::Output, EPinType::Animation));
}

bool FBodyFaceAnimMergeNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	const FFrameAnimationData& FaceAnimation = InPipelineData->GetData<FFrameAnimationData>(Pins[0]);
	const FFrameAnimationData& BodyAnimation = InPipelineData->GetData<FFrameAnimationData>(Pins[1]);

	FFrameAnimationData Output;

	if (FaceAnimation.AnimationQuality == EFrameAnimationQuality::Undefined && BodyAnimation.AnimationQuality == EFrameAnimationQuality::Undefined)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Both face and body animations are invalid");
		Output = FaceAnimation;
	}
	else if (FaceAnimation.AnimationQuality != EFrameAnimationQuality::Undefined && BodyAnimation.AnimationQuality == EFrameAnimationQuality::Undefined)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Body animation is invalid");
		Output = FaceAnimation;
	}
	else if (FaceAnimation.AnimationQuality == EFrameAnimationQuality::Undefined && BodyAnimation.AnimationQuality != EFrameAnimationQuality::Undefined)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Face animation is invalid");
		Output = BodyAnimation;
	}
	else
	{
		Output = FaceAnimation;
		Output.BodyAnimationData = BodyAnimation.BodyAnimationData;
		Output.RawBodyAnimationSMPLXShape = BodyAnimation.RawBodyAnimationSMPLXShape;
		Output.RawBodyAnimationSMPLXPose = BodyAnimation.RawBodyAnimationSMPLXPose;
		Output.RawBodyAnimationSMPLXTranslation = BodyAnimation.RawBodyAnimationSMPLXTranslation;
		Output.RawBodyAnimationSMPLXData = BodyAnimation.RawBodyAnimationSMPLXData;

		Output.AnimationQuality = FMath::Min(FaceAnimation.AnimationQuality, BodyAnimation.AnimationQuality);
	}

	InPipelineData->SetData<FFrameAnimationData>(Pins[2], MoveTemp(Output));

	return true;
}

}
