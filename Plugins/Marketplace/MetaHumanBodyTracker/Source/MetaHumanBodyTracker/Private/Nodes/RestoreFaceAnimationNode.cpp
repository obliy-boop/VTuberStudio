// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/RestoreFaceAnimationNode.h"



namespace UE::MetaHuman::Pipeline
{

FRestoreFaceAnimationNode::FRestoreFaceAnimationNode(const FString& InName) : FNode("RestoreFaceAnimation", InName)
{
	Pins.Add(FPin("Animation Out", EPinDirection::Output, EPinType::Animation));
}

bool FRestoreFaceAnimationNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{
	FrameCount = 0;

	return true;
}

bool FRestoreFaceAnimationNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (FrameCount >= AnimationData.Num())
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::BadData);
		InPipelineData->SetErrorNodeMessage(TEXT("Bad data"));
		return false;
	}

	InPipelineData->SetData<FFrameAnimationData>(Pins[0], MoveTemp(AnimationData[FrameCount]));

	FrameCount++;

	return true;
}

}
