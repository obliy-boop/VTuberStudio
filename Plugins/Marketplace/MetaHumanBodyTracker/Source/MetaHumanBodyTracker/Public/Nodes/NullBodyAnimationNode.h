// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"
#include "BaseBodyTracker.h"

#define UE_API METAHUMANBODYTRACKER_API



namespace UE::MetaHuman::Pipeline
{

class FNullBodyAnimationNode : public FNode, public FBaseBodyTracker
{
public:

	UE_API FNullBodyAnimationNode(const FString& InName);

	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;

	EFrameAnimationQuality ReportedAnimationQuality = EFrameAnimationQuality::Undefined;

private:

	float Height = 0;
};

}

#undef UE_API
