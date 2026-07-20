// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/NullBodyAnimationNode.h"
#include "MetaHumanSMPLX.h"



namespace UE::MetaHuman::Pipeline
{

FNullBodyAnimationNode::FNullBodyAnimationNode(const FString& InName) : FNode("NullBodyAnimation", InName)
{
	Pins.Add(FPin("Animation Out", EPinDirection::Output, EPinType::Animation));

	SMPLX->SetShape(TArray<float>());

	Height = SMPLX->GetHeightOffset();
}

bool FNullBodyAnimationNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	FFrameAnimationData Output;

	Output.RawBodyAnimationSMPLXTranslation = FVector(0, Height, 0);
	Output.AnimationQuality = ReportedAnimationQuality;

	PrepareOutput(Output);

	InPipelineData->SetData<FFrameAnimationData>(Pins[0], MoveTemp(Output));

	return true;
}

}
