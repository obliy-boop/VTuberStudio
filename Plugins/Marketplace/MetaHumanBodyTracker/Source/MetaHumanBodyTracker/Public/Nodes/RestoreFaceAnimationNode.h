// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"

#define UE_API METAHUMANBODYTRACKER_API



namespace UE::MetaHuman::Pipeline
{

class FRestoreFaceAnimationNode : public FNode
{
public:

	UE_API FRestoreFaceAnimationNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;

	TArray<FFrameAnimationData> AnimationData;

	enum ErrorCode
	{
		BadData,
	};

private:

	int32 FrameCount = 0;
};

}

#undef UE_API
