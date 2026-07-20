// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"
#include "BaseBodyTracker.h"

#define UE_API METAHUMANBODYTRACKER_API



namespace UE::MetaHuman::Pipeline
{

class FDummyBodyTrackerNode : public FNode, public FBaseBodyTracker
{
public:

	UE_API FDummyBodyTrackerNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;

	enum ErrorCode
	{
		FailedToInitialize = 0,
		BadData,
	};

private:

	TArray<float> Shape;
	TArray<TArray<float>> Pose;
	TArray<FVector> Position;
};

}

#undef UE_API
