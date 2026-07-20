// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"
#include "BaseBodyTracker.h"

#include "NNEModelData.h"

#define UE_API METAHUMANBODYTRACKER_API



namespace UE::MetaHuman::Pipeline
{

class FRealtimeBodyTrackerNode : public FNode, public FBaseBodyTracker
{
public:

	UE_API FRealtimeBodyTrackerNode(const FString& InName);

	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;

	EFrameAnimationQuality ReportedAnimationQuality = EFrameAnimationQuality::Undefined;
	bool bTranslation = true;

	enum ErrorCode
	{
		FailedToInitialize = 0,
	};

private:

	TStrongObjectPtr<UNNEModelData> Backbone;
	TStrongObjectPtr<UNNEModelData> Camera;
	TStrongObjectPtr<UNNEModelData> MFVHead;

	TPimplPtr<class FRealtimeBodyTrackerNodeInternal> Impl;
};

}

#undef UE_API
