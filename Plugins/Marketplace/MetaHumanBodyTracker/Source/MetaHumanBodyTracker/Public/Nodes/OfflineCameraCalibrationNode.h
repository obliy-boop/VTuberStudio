// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Math/Vector.h"
#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"

#include "NNEModelData.h"

#define UE_API METAHUMANBODYTRACKER_API



namespace UE::MetaHuman::Pipeline
{

class FOfflineCameraCalibrationNode : public FNode
{
public:

	UE_API FOfflineCameraCalibrationNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool End(const TSharedPtr<FPipelineData>& InPipelineData) override;

	int32 Stride = 1;

	float FocalLength = 0;
	float Pitch = 0;
	float Roll = 0;

	/** Body-tracking raster (ViTPose preprocess sizing; aligned with Stage-1 image grid). Updated when frames run through this node. */
	int32 BodyTrackingRasterWidth = 0;
	int32 BodyTrackingRasterHeight = 0;
	FVector2f CameraImgCenter = FVector2f(0.f, 0.f);

	enum ErrorCode
	{
		FailedToInitialize = 0,
		BadStride,
	};

private:

	int32 FrameCount = 0;

	TStrongObjectPtr<UNNEModelData> GeoCalib;

	TPimplPtr<class FOfflineCameraCalibrationNodeInternal> Impl;
};

}

#undef UE_API
