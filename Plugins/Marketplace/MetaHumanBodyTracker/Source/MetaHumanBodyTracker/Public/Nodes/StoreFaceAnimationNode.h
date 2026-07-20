// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"

#define UE_API METAHUMANBODYTRACKER_API



namespace UE::MetaHuman::Pipeline
{

class FStoreFaceAnimationNode : public FNode
{
public:

	UE_API FStoreFaceAnimationNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;

	TArray<FFrameAnimationData> AnimationData;
};

}

#undef UE_API
