// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"

#define UE_API METAHUMANBODYTRACKER_API



namespace UE::MetaHuman::Pipeline
{

class FBodyFaceAnimMergeNode : public FNode
{
public:

	UE_API FBodyFaceAnimMergeNode(const FString& InName);

	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;
};

}

#undef UE_API
