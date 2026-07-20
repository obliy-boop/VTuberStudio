// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Pipeline/Node.h"
#include "Pipeline/PipelineData.h"
#include "BaseBodyTracker.h"

#include "NNEModelData.h"

#define UE_API METAHUMANBODYTRACKER_API



class UMetaHumanPerformance;

namespace UE::MetaHuman::Pipeline
{

class FOfflineBodyDetectionNode : public FNode
{
public:

	UE_API FOfflineBodyDetectionNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool End(const TSharedPtr<FPipelineData>& InPipelineData) override;

	float DetectionScoreThreshold = 0.7f;
	float TrackingScoreThreshold = 0.5f;

	TArray64<bool> ValidFrames;
	TArray64<FIntRect> BoundingBoxes;

	enum ErrorCode
	{
		FailedToInitialize = 0,
	};

private:

	TPimplPtr<class FOfflineBodyDetectionNodeInternal> Impl;
};

class FOffline2DKeypointEstimationNode : public FNode
{
public:

	UE_API FOffline2DKeypointEstimationNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool End(const TSharedPtr<FPipelineData>& InPipelineData) override;

	TSharedPtr<const class FOfflineBodyDetectionNode> OfflineBodyDetectionNode = nullptr;

	TArray64<float> Keypoints; // [FrameCount, 133, 3 (x, y, confidence)]

	enum ErrorCode
	{
		FailedToInitialize = 0,
		BadData
	};

private:

	TPimplPtr<class FOffline2DKeypointEstimationNodeInternal> Impl;
};

class FOfflinePoseEstimationNode : public FNode
{
public:

	UE_API FOfflinePoseEstimationNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool End(const TSharedPtr<FPipelineData>& InPipelineData) override;

	TSharedPtr<class FOfflineCameraCalibrationNode> OfflineCameraCalibrationNode = nullptr;
	TSharedPtr<const class FOfflineBodyDetectionNode> OfflineBodyDetectionNode = nullptr;
	TSharedPtr<const class FOffline2DKeypointEstimationNode> Offline2DKeypointEstimationNode = nullptr;

	TArray64<float> Pose, Shape, Translation;

	/** OpenPose BODY_25: 25*3 packed (x, y, conf) per frame. */
	TArray64<float> KeypointsTitan25x3;
	/** Bbox (x0, y0, x1, y1) per frame. */
	TArray64<float> BoundingBoxMinMaxXY;
	/** Hue static contact logits: 6 floats per frame. */
	TArray64<float> StaticContactConfidenceLogits;

	TArray64<bool> ValidFrame;

	// Values set from Performance during pipeline setup

	/** Source footage frame rate */
	float SourceFps = 30.0f;
	/** Body height in cm or 0 if unknown */
	float BodyHeight = 0.0f;


	enum ErrorCode
	{
		FailedToInitialize = 0,
		BadData
	};

private:

	TStrongObjectPtr<UNNEModelData> ViTPose;
	TStrongObjectPtr<UNNEModelData> ViTPosePost;
	TStrongObjectPtr<UNNEModelData> CHMRBackbone;
	TStrongObjectPtr<UNNEModelData> CHMRHead;
	TStrongObjectPtr<UNNEModelData> HueStep;
	TStrongObjectPtr<UNNEModelData> HueFinalStep;

	TPimplPtr<class FOfflinePoseEstimationNodeInternal> Impl;
};

class FOfflineBodyTrackerFinalizeNode : public FNode, public FBaseBodyTracker
{
public:

	UE_API FOfflineBodyTrackerFinalizeNode(const FString& InName);

	UE_API virtual bool Start(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool Process(const TSharedPtr<FPipelineData>& InPipelineData) override;
	UE_API virtual bool End(const TSharedPtr<FPipelineData>& InPipelineData) override;

	UMetaHumanPerformance* Performance = nullptr;

	TSharedPtr<FOfflinePoseEstimationNode> OfflineBodyTrackerNode = nullptr;

	enum ErrorCode
	{
		FailedToInitialize = 0,
		BadData,
		InsufficientData,
		NoPerformance,
	};

	TArray<float> AverageShape;
	bool bEnableFootLocking = true;

private:

	int32 NumFrames = 0;
	int32 FrameCount = 0;

	bool ApplyHps(float InCameraPitchRadians, float InCameraRollRadians, float& OutGlobalMaxY);

	TPimplPtr<class FOfflineBodyTrackerFinalizeNodeInternal> Impl;
};

}

#undef UE_API
