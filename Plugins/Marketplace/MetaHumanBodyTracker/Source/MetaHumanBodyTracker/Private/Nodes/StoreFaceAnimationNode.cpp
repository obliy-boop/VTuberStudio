// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/StoreFaceAnimationNode.h"



namespace UE::MetaHuman::Pipeline
{

FStoreFaceAnimationNode::FStoreFaceAnimationNode(const FString& InName) : FNode("StoreFaceAnimation", InName)
{
	Pins.Add(FPin("Animation In", EPinDirection::Input, EPinType::Animation));
}

bool FStoreFaceAnimationNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{
	AnimationData.Reset();

	return true;
}

bool FStoreFaceAnimationNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	const FFrameAnimationData& Animation = InPipelineData->GetData<FFrameAnimationData>(Pins[0]);

	AnimationData.Add(Animation);

	return true;
}

}
