// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/OfflineBodyTrackerNode.h"
#include "Nodes/OfflineCameraCalibrationNode.h"
#include "BodyTrackingOptimizer.h"
#include "MetaHumanPerformance.h"
#include "MetaHumanBodyTracker.h"

#include "Detectron2.h"
#include "ImageCore.h"
#include "Logging/StructuredLog.h"
#include "MetaHumanBodyTrackerLog.h"
#include "MetaHumanSMPLX.h"
#include "OfflineBodyTracker.h"
#include "SAM2.h"

#include "UObject/Package.h"

#include "SupressWarnings.h"

MH_DISABLE_EIGEN_WARNINGS
#include "Eigen/Dense"
MH_ENABLE_WARNINGS



namespace UE::MetaHuman::Pipeline
{

class FOfflineBodyDetectionNodeInternal
{
public:

	// At most one of those is set at any time in order to reduce VRAM usage
	// Tracker is set only when a person was detected
	TUniquePtr<UE::NNE::Segmentation::IDetector> Detector;
	TUniquePtr<UE::NNE::Segmentation::ITracker> Tracker;

	// Detectron2 models
	TStrongObjectPtr<UNNEModelData> BackboneRpnData;
	TStrongObjectPtr<UNNEModelData> RoiHeadsBoxData;
	TStrongObjectPtr<UNNEModelData> RoiHeadsKeypointData;

	// SAM2 models
	TStrongObjectPtr<UNNEModelData> ImageEncoderData;
	TStrongObjectPtr<UNNEModelData> MaskDecoderData;
	TStrongObjectPtr<UNNEModelData> MemoryEncoderData;
	TStrongObjectPtr<UNNEModelData> MemoryAttentionData;
	TStrongObjectPtr<UDataTable>    WeightsTable;
	bool Initialized = false;

	const int MaxImageHeight = 896;
};

FOfflineBodyDetectionNode::FOfflineBodyDetectionNode(const FString& InName) : FNode("OfflineBodyDetection", InName)
{
	Impl = MakePimpl<FOfflineBodyDetectionNodeInternal>();
	UNNEModelData* BackboneRpnData, * RoiHeadsBoxData, * RoiHeadsKeypointData;
	if (!UE::NNE::Segmentation::Detectron2::TryLoadModelData(BackboneRpnData, RoiHeadsBoxData, RoiHeadsKeypointData))
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load detectron2 models");
		return;
	}
	Impl->BackboneRpnData = TStrongObjectPtr<UNNEModelData>(BackboneRpnData);
	Impl->RoiHeadsBoxData = TStrongObjectPtr<UNNEModelData>(RoiHeadsBoxData);
	Impl->RoiHeadsKeypointData = TStrongObjectPtr<UNNEModelData>(RoiHeadsKeypointData);
	UNNEModelData* ImageEncoderData, * MaskDecoderData, * MemoryEncoderData, * MemoryAttentionData;
	UDataTable* WeightsTable;
	if (!UE::NNE::Segmentation::SAM2::TryLoadModelData(
		ImageEncoderData, MaskDecoderData, MemoryEncoderData, MemoryAttentionData, WeightsTable))
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load sam2 models");
		return;
	}
	Impl->ImageEncoderData = TStrongObjectPtr<UNNEModelData>(ImageEncoderData);
	Impl->MaskDecoderData = TStrongObjectPtr<UNNEModelData>(MaskDecoderData);
	Impl->MemoryEncoderData = TStrongObjectPtr<UNNEModelData>(MemoryEncoderData);
	Impl->MemoryAttentionData = TStrongObjectPtr<UNNEModelData>(MemoryAttentionData);
	Impl->WeightsTable = TStrongObjectPtr<UDataTable>(WeightsTable);
	Impl->Initialized = true;

	Pins.Add(FPin("Image In", EPinDirection::Input, EPinType::UE_Image));
}

bool FOfflineBodyDetectionNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{
	ValidFrames.Reset();
	BoundingBoxes.Reset();
	Impl->Detector.Reset();
	Impl->Tracker.Reset();
	return true;
}

bool FOfflineBodyDetectionNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	static const TArray<UE::NNE::Segmentation::FRuntimePreference> RuntimePreferences =
	{
		{ TEXT("NNERuntimeORTDml"), UE::NNE::Segmentation::EDeviceType::GPU },
		{ TEXT("NNERuntimeORTCpu"), UE::NNE::Segmentation::EDeviceType::CPU }
	};

	if (!Impl->Initialized)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize body detection"));
		return false;
	}

	const FUEImageDataType& Input = InPipelineData->GetData<FUEImageDataType>(Pins[0]);
	FImage Frame(Input.Width, Input.Height, ERawImageFormat::BGRA8);
	Frame.RawData = Input.Data;
	TArray<UE::NNE::Segmentation::FTracking> Tracking;
	if (Impl->Tracker)
	{
		Tracking = Impl->Tracker->Track(Frame);
	}
	else
	{
		if (!Impl->Detector)
		{
			UE::NNE::Segmentation::Detectron2::FMakeParams Params =
			{
				.BackboneRpnData = Impl->BackboneRpnData.Get(),
				.RoiHeadsBoxData = Impl->RoiHeadsBoxData.Get(),
				.RoiHeadsKeypointData = Impl->RoiHeadsKeypointData.Get(),
			};
			Impl->Detector = UE::NNE::Segmentation::Detectron2::Make(RuntimePreferences, Params);
			if (!Impl->Detector)
			{
				InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
				InPipelineData->SetErrorNodeMessage(TEXT("Failed to create Detectron 2"));
				return false;
			}
		}
		TArray<UE::NNE::Segmentation::FDetection> Detections = Impl->Detector->Detect(Frame);
		if (Detections.Num() >= 1 && Detections[0].Score >= DetectionScoreThreshold) // TODO: Adjust for multi person tracking
		{
			Impl->Detector.Reset();
			if (!Impl->Tracker)
			{
				UE::NNE::Segmentation::SAM2::FMakeParams Params =
				{
					.ImageEncoderData = Impl->ImageEncoderData.Get(),
					.MaskDecoderData = Impl->MaskDecoderData.Get(),
					.MemoryEncoderData = Impl->MemoryEncoderData.Get(),
					.MemoryAttentionData = Impl->MemoryAttentionData.Get(),
					.WeightsTable = Impl->WeightsTable.Get()
				};
				Impl->Tracker = UE::NNE::Segmentation::SAM2::Make(RuntimePreferences, Params);
				if (!Impl->Tracker)
				{
					InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
					InPipelineData->SetErrorNodeMessage(TEXT("Failed to create SAM 2"));
					return false;
				}
			}
			Tracking = Impl->Tracker->Initialize(Frame, TConstArrayView<UE::NNE::Segmentation::FDetection>(Detections.GetData(), 1));
		}
	}
	if (Tracking.Num() >= 1 && Tracking[0].Score >= TrackingScoreThreshold) // TODO: Adjust for multi person tracking
	{
		double ImageScale = FMath::Min((double)Impl->MaxImageHeight / Input.Height, 1);
		FIntRect BoundingBox = Tracking[0].Box.Scale(ImageScale);
		BoundingBox.InflateRect(10);
		BoundingBoxes.Add(BoundingBox);
		ValidFrames.Add(true);
	}
	else
	{
		Impl->Tracker.Reset();

		ValidFrames.Add(false);
		BoundingBoxes.Add(FIntRect());
	}
	return true;
}

bool FOfflineBodyDetectionNode::End(const TSharedPtr<FPipelineData>& InPipelineData)
{
	Impl->Detector.Reset();
	Impl->Tracker.Reset();

	return true;
}

class FOffline2DKeypointEstimationNodeInternal
{
public:

	TUniquePtr<UE::BodyTracker::FEstimateKeypoints> Estimator;
	int FrameIndex = 0;
};

FOffline2DKeypointEstimationNode::FOffline2DKeypointEstimationNode(const FString& InName) : FNode("Offline2DKeypointEstimation", InName)
{
	Impl = MakePimpl<FOffline2DKeypointEstimationNodeInternal>();

	Pins.Add(FPin("Image In", EPinDirection::Input, EPinType::UE_Image));

	UNNEModelData* ViTPose = LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Offline/ViTPose.ViTPose"));
	if (!ViTPose)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load ViTPose model");
		return;
	}

	UNNEModelData* ViTPosePost = LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Offline/ViTPosePost.ViTPosePost"));
	if (!ViTPosePost)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load ViTPosePost model");
		return;
	}

	static const TArray<UE::BodyTracker::FRuntimePreference> RuntimePreferences =
	{
		{ TEXT("NNERuntimeORTDml"), UE::BodyTracker::EDeviceType::GPU },
		{ TEXT("NNERuntimeORTCpu"), UE::BodyTracker::EDeviceType::CPU }
	};
	Impl->Estimator = UE::BodyTracker::FEstimateKeypoints::Make(ViTPose, ViTPosePost, RuntimePreferences);
	if (!Impl->Estimator)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to create 2D Keypoint estimator");
		return;
	}
}

bool FOffline2DKeypointEstimationNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{
	Keypoints.Reset();
	Impl->FrameIndex = 0;
	return true;
}

bool FOffline2DKeypointEstimationNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!Impl->Estimator)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize 2d keypoint estimator"));
		return false;
	}
	if (!OfflineBodyDetectionNode)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("OfflineBodyDetectionNode not set"));
		return false;
	}
	const TArray64<bool>& ValidFrames = OfflineBodyDetectionNode->ValidFrames;
	const TArray64<FIntRect>& BoundingBoxes = OfflineBodyDetectionNode->BoundingBoxes;
	if (ValidFrames.Num() != BoundingBoxes.Num() || ValidFrames.Num() <= Impl->FrameIndex)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::BadData);
		InPipelineData->SetErrorNodeMessage(TEXT("Bad data"));
		return false;
	}

	if (ValidFrames[Impl->FrameIndex])
	{
		const FUEImageDataType& Frame = InPipelineData->GetData<FUEImageDataType>(Pins[0]);
		Impl->Estimator->Run(Frame, BoundingBoxes[Impl->FrameIndex], Keypoints);
	}
	else
	{
		Keypoints.AddZeroed(UE::BodyTracker::FEstimateKeypoints::NumFloatsPerKeypoint);
	}
	Impl->FrameIndex++;
	return true;
}

bool FOffline2DKeypointEstimationNode::End(const TSharedPtr<FPipelineData>& InPipelineData)
{
	Impl->Estimator.Reset();

	return true;
}

class FOfflinePoseEstimationNodeInternal
{
public:

	TUniquePtr<UE::BodyTracker::FCHMR> CHMR;
	TUniquePtr<UE::BodyTracker::FHue> Hue;

	int FrameIndex = 0;
	bool bLastWindow = false;

	TArray64<float> TmpHuman3DTokens;
};

FOfflinePoseEstimationNode::FOfflinePoseEstimationNode(const FString& InName) : FNode("OfflinePoseEstimation", InName)
{
	Impl = MakePimpl<FOfflinePoseEstimationNodeInternal>();

	Pins.Add(FPin("Image In", EPinDirection::Input, EPinType::UE_Image));

	CHMRBackbone = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Offline/chmr_backbone.chmr_backbone")));
	if (!CHMRBackbone)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load CHMRBackbone model");
		return;
	}

	CHMRHead = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Offline/chmr_head.chmr_head")));
	if (!CHMRHead)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load CHMRHead model");
		return;
	}

	HueStep = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Offline/hue_step_simplified.hue_step_simplified")));
	if (!HueStep)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load HueStep model");
		return;
	}
	HueFinalStep = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Offline/hue_finalStep_simplified.hue_finalStep_simplified")));
	if (!HueFinalStep)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load HueFinalStep model");
		return;
	}
	static const TArray<UE::BodyTracker::FRuntimePreference> RuntimePreferences =
	{
		{ TEXT("NNERuntimeORTDml"), UE::BodyTracker::EDeviceType::GPU },
		{ TEXT("NNERuntimeORTCpu"), UE::BodyTracker::EDeviceType::CPU }
	};
	Impl->CHMR = UE::BodyTracker::FCHMR::Make(CHMRBackbone.Get(), CHMRHead.Get(), RuntimePreferences);
	Impl->Hue = UE::BodyTracker::FHue::Make(HueStep.Get(), HueFinalStep.Get(), RuntimePreferences);
	if (!Impl->CHMR)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to create contact hmr");
		return;
	}
	if (!Impl->Hue)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to create hue");
		return;
	}
}

bool FOfflinePoseEstimationNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!OfflineBodyDetectionNode)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("OfflineBodyDetectionNode not set"));
		return false;
	}

	Impl->FrameIndex = 0;
	Impl->bLastWindow = false;

	Pose.Reset();
	Shape.Reset();
	Translation.Reset();
	KeypointsTitan25x3.Reset();
	BoundingBoxMinMaxXY.Reset();
	StaticContactConfidenceLogits.Reset();
	ValidFrame = OfflineBodyDetectionNode->ValidFrames;

	return true;
}

bool FOfflinePoseEstimationNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!Impl->CHMR)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize contact hmr"));
		return false;
	}

	if (!Impl->Hue)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize hue"));
		return false;
	}

	if (!OfflineCameraCalibrationNode)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("OfflineCameraCalibrationNode not set"));
		return false;
	}

	if (!OfflineBodyDetectionNode)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("OfflineBodyDetectionNode not set"));
		return false;
	}

	if (!Offline2DKeypointEstimationNode)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Offline2DKeypointEstimationNode not set"));
		return false;
	}

	const TArray64<FIntRect>& BoundingBoxes = OfflineBodyDetectionNode->BoundingBoxes;
	TConstArrayView64<float> Keypoints = Offline2DKeypointEstimationNode->Keypoints;
	if (ValidFrame.Num() != BoundingBoxes.Num() || 
		Keypoints.Num() % UE::BodyTracker::FEstimateKeypoints::NumFloatsPerKeypoint != 0 ||
		ValidFrame.Num() != Keypoints.Num() / UE::BodyTracker::FEstimateKeypoints::NumFloatsPerKeypoint ||
		ValidFrame.Num() <= Impl->FrameIndex)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::BadData);
		InPipelineData->SetErrorNodeMessage(TEXT("Bad data"));
		return false;
	}

	const FUEImageDataType& Frame = InPipelineData->GetData<FUEImageDataType>(Pins[0]);

	if (ValidFrame[Impl->FrameIndex])
	{
		const FIntRect& Box = BoundingBoxes[Impl->FrameIndex];
		TConstArrayView64<float> CurrentKeypoints = Keypoints.Slice(
			Impl->FrameIndex * UE::BodyTracker::FEstimateKeypoints::NumFloatsPerKeypoint,
			UE::BodyTracker::FEstimateKeypoints::NumFloatsPerKeypoint);
		const float FocalLength = OfflineCameraCalibrationNode->FocalLength;
		const FIntVector2 ImageSize(Frame.Width, Frame.Height);
		Impl->TmpHuman3DTokens.Reset();
		Impl->CHMR->Run(Frame, Box, CurrentKeypoints, FocalLength, Impl->TmpHuman3DTokens);
		bool bWindowFull = Impl->Hue->EnqueValidFrame(
			FocalLength, 
			ImageSize, 
			Impl->TmpHuman3DTokens, 
			Box, 
			CurrentKeypoints,
			BodyHeight
		);
		if (bWindowFull)
		{
			Impl->bLastWindow = Impl->Hue->Run(Pose, Translation, Shape, StaticContactConfidenceLogits).LastWindow;
		}
		
		BoundingBoxMinMaxXY.Add(Box.Min.X);
		BoundingBoxMinMaxXY.Add(Box.Min.Y);
		BoundingBoxMinMaxXY.Add(Box.Max.X);
		BoundingBoxMinMaxXY.Add(Box.Max.Y);
		UE::BodyTracker::FEstimateKeypoints::Op25FromKeypoints(CurrentKeypoints, KeypointsTitan25x3);
	}
	else
	{
		Impl->Hue->EnqueInvalidFrame();
		BoundingBoxMinMaxXY.AddZeroed(4);
		KeypointsTitan25x3.AddZeroed(UE::BodyTracker::FEstimateKeypoints::NumFloatsPerOp25);
	}
	Impl->FrameIndex++;
	return true;
}

bool FOfflinePoseEstimationNode::End(const TSharedPtr<FPipelineData>& InPipelineData)
{
	while (Impl->Hue && !Impl->bLastWindow)
	{
		Impl->bLastWindow = Impl->Hue->Run(Pose, Translation, Shape, StaticContactConfidenceLogits).LastWindow;
	}
	Impl->CHMR.Reset();
	Impl->Hue.Reset();

	return true;
}



class FOfflineBodyTrackerFinalizeNodeInternal
{
public:

	TSharedPtr<FMetaHumanSMPLX> SMPLX;
};

FOfflineBodyTrackerFinalizeNode::FOfflineBodyTrackerFinalizeNode(const FString& InName) : FNode("OfflineBodyTrackerFinalize", InName)
{
	Impl = MakePimpl<FOfflineBodyTrackerFinalizeNodeInternal>();

	Impl->SMPLX = MakeShared<FMetaHumanSMPLX>();
	Impl->SMPLX->Init();
	Impl->SMPLX->SetAccountForHeight(false);

	Pins.Add(FPin("Animation Out", EPinDirection::Output, EPinType::Animation));
}

bool FOfflineBodyTrackerFinalizeNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{

	if (!OfflineBodyTrackerNode || !OfflineBodyTrackerNode->OfflineCameraCalibrationNode)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("OfflineBodyTrackerNode not set"));
		return false;
	}

	NumFrames = OfflineBodyTrackerNode->Shape.Num() / 10;

	if (NumFrames == 0 || 
		(OfflineBodyTrackerNode->Pose.Num() / 165) != NumFrames || 
		(OfflineBodyTrackerNode->Translation.Num() / 3) != NumFrames || 
		OfflineBodyTrackerNode->ValidFrame.Num() != NumFrames)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::BadData);
		InPipelineData->SetErrorNodeMessage(TEXT("Bad data"));
		return false;
	}

	FrameCount = 0;

	if (AverageShape.IsEmpty())
	{
		AverageShape.SetNumZeroed(10);

		int32 ValidFrameCount = 0;

		for (int32 Frame = 0; Frame < NumFrames; ++Frame)
		{
			if (OfflineBodyTrackerNode->ValidFrame[Frame])
			{
				ValidFrameCount++;

				for (int32 Shape = 0; Shape < 10; ++Shape)
				{
					AverageShape[Shape] += OfflineBodyTrackerNode->Shape[Frame * 10 + Shape];
				}
			}
		}

		if (ValidFrameCount == 0)
		{
			InPipelineData->SetErrorNodeCode(ErrorCode::BadData);
			InPipelineData->SetErrorNodeMessage(TEXT(
				"Body tracking could not lock onto the subject in any frame. "
				"Try lowering the body tracking thresholds in the Performance asset, "
				"or check that the subject is clearly visible in the footage."));
			return false;
		}

		for (int32 Shape = 0; Shape < 10; ++Shape)
		{
			AverageShape[Shape] /= ValidFrameCount;
		}
	}

	for (int32 Index = 0; Index < OfflineBodyTrackerNode->Shape.Num(); ++Index)
	{
		OfflineBodyTrackerNode->Shape[Index] = AverageShape[Index % 10];
	}

	const float CameraPitch = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->Pitch;
	const float CameraRoll = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->Roll;
	float HpsGlobalMaxY = 0.0f;

	if (!ApplyHps(CameraPitch, CameraRoll, HpsGlobalMaxY))
	{
		return false;
	}

	const FVector CameraWorldPosition(0.0f, HpsGlobalMaxY, 0.0f);

	const TSharedPtr<const FOfflineCameraCalibrationNode>& CalibNode = OfflineBodyTrackerNode->OfflineCameraCalibrationNode;
	if (CalibNode.IsValid() && CalibNode->BodyTrackingRasterWidth > 0 && CalibNode->BodyTrackingRasterHeight > 0)
	{
		const UMetaHumanSMPLXData* SmplxData = Impl->SMPLX->GetSmplxData();
		if (SmplxData != nullptr)
		{
			FBodyTrackingOptimizer Optimizer;
			if (Optimizer.Initialize(SmplxData->Verts, SmplxData->Faces, SmplxData->JointReg, SmplxData->Weights, SmplxData->BlendShapes))
			{
				Optimizer.SetCamera(CalibNode->FocalLength, CalibNode->CameraImgCenter, CameraPitch, CameraRoll, CameraWorldPosition);
				Optimizer.EnableFootlocking(bEnableFootLocking);

				FBodyTrackingOptimizerFrameData FrameData;
				FrameData.Pose = OfflineBodyTrackerNode->Pose;
				FrameData.Translation = OfflineBodyTrackerNode->Translation;
				FrameData.Shape = OfflineBodyTrackerNode->Shape;
				FrameData.Keypoints = OfflineBodyTrackerNode->KeypointsTitan25x3;
				FrameData.BoundingBoxes = OfflineBodyTrackerNode->BoundingBoxMinMaxXY;
				FrameData.StaticContactLogits = OfflineBodyTrackerNode->StaticContactConfidenceLogits;
				FrameData.ValidFrame = OfflineBodyTrackerNode->ValidFrame;
				FrameData.NumFrames = NumFrames;
				FrameData.Fps = OfflineBodyTrackerNode->SourceFps;

				if (Optimizer.Run(FrameData))
				{
					Optimizer.ApplyResult(OfflineBodyTrackerNode->Pose, OfflineBodyTrackerNode->Translation, NumFrames);
				}
			}
		}
	}

	return true;
}

bool FOfflineBodyTrackerFinalizeNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!SMPLX->IsInitialized())
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize"));
		return false;
	}

	if (FrameCount >= NumFrames)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::InsufficientData);
		InPipelineData->SetErrorNodeMessage(TEXT("Insufficient data"));
		return false;
	}

	FFrameAnimationData Output;

	// Only produce body animation data for frames where a person was detected.
	// Invalid frames produce empty BodyAnimationData, which downstream export
	// gap fill handles (hold last frame / reference pose).
	const bool bIsValidFrame = FrameCount < OfflineBodyTrackerNode->ValidFrame.Num() 
		&& OfflineBodyTrackerNode->ValidFrame[FrameCount];

	if (bIsValidFrame)
	{
		Output.RawBodyAnimationSMPLXShape = AverageShape;

		Output.RawBodyAnimationSMPLXPose.SetNumZeroed(165);
		for (int32 Index = 0; Index < 165; ++Index)
		{
			Output.RawBodyAnimationSMPLXPose[Index] = OfflineBodyTrackerNode->Pose[FrameCount * 165 + Index];
		}

		Output.RawBodyAnimationSMPLXTranslation = FVector(OfflineBodyTrackerNode->Translation[FrameCount * 3 + 0], OfflineBodyTrackerNode->Translation[FrameCount * 3 + 1], OfflineBodyTrackerNode->Translation[FrameCount * 3 + 2]);

		Output.AnimationQuality = EFrameAnimationQuality::PostFiltered;

		PrepareOutput(Output);
	}

	InPipelineData->SetData<FFrameAnimationData>(Pins[0], MoveTemp(Output));

	FrameCount++;

	return true;
}

bool FOfflineBodyTrackerFinalizeNode::ApplyHps(
	float InCameraPitchRadians,
	float InCameraRollRadians,
	float& OutGlobalMaxY)
{
	FOfflinePoseEstimationNode* OfflineTracker = OfflineBodyTrackerNode.Get();
	if (OfflineTracker == nullptr)
	{
		UE_LOGFMT(LogMetaHumanBodyTracker, Error, "ApplyHps: OfflineBodyTrackerNode is null");
		return false;
	}

	const int32 HpsNumFrames = OfflineTracker->Shape.Num() / 10;
	if (HpsNumFrames == 0
		|| (OfflineTracker->Pose.Num() / 165) != HpsNumFrames
		|| (OfflineTracker->Translation.Num() / 3) != HpsNumFrames
		|| OfflineTracker->ValidFrame.Num() != HpsNumFrames)
	{
		UE_LOGFMT(LogMetaHumanBodyTracker, Error, "ApplyHps: buffer size mismatch");
		return false;
	}
	
	Impl->SMPLX->SetShape(AverageShape);

	// Calculate position of pelvis (the center for rotation)
	FVector PelvisVector = Impl->SMPLX->GetBoneHierarchyUE()[1].Get<2>().GetLocation(); // Bone 1 = pelvis
	PelvisVector = FMetaHumanSMPLX::UE2SMPL(PelvisVector);
	const Eigen::Vector3f Pelvis(PelvisVector.X, PelvisVector.Y, PelvisVector.Z);

	// Account for camera orientation and find global min Y position of all skinned vertex across all frames.
	// The anim is then translated so that the global min Y = 0 thus locating the anim in the 3D world at the groundplane.
	// However, this bit of code is still working in smpl-camera coords, where +Y is down; so its the max Y we actually need.
	// The conversion to smpl-world coords where +Y is up (so flipping Y and Z) comes later.

	const Eigen::Matrix<float, 3, 3> CameraPitch = Eigen::AngleAxisf(-InCameraPitchRadians, Eigen::Vector3f::UnitX()).toRotationMatrix();
	const Eigen::Matrix<float, 3, 3> CameraRoll = Eigen::AngleAxisf(InCameraRollRadians, Eigen::Vector3f::UnitY()).toRotationMatrix();
	const Eigen::Matrix<float, 3, 3> CameraRotation = CameraPitch * CameraRoll;
	float GlobalMaxY = 0;

	// Pass 1: Apply camera rotation and find ground contact
	for (int32 Frame = 0; Frame < HpsNumFrames; ++Frame)
	{
		if (!OfflineTracker->ValidFrame[Frame])
		{
			continue;
		}

		// Pelvis pose
		Eigen::Vector3f Axis(OfflineTracker->Pose[Frame * 165 + 0], OfflineTracker->Pose[Frame * 165 + 1], OfflineTracker->Pose[Frame * 165 + 2]);
		const float Angle = Axis.norm();

		Eigen::Matrix<float, 3, 3> Pose;

		if (Angle > 1e-4)
		{
			Axis.normalize();

			const Eigen::AngleAxisf AxisAngle(Angle, Axis);

			Pose = AxisAngle.toRotationMatrix();
		}
		else
		{
			Pose.setIdentity();
		}

		Pose = CameraRotation * Pose;

		Eigen::Vector3f Translation(OfflineTracker->Translation[Frame * 3 + 0], OfflineTracker->Translation[Frame * 3 + 1], OfflineTracker->Translation[Frame * 3 + 2]);

		Translation = CameraRotation * (Pelvis + Translation) - Pelvis;

		// Store modified values

		const Eigen::AngleAxisf AxisAngle = Eigen::AngleAxisf(Pose);
		const Eigen::Vector3f CompactRodrigues = AxisAngle.axis() * AxisAngle.angle();

		OfflineTracker->Pose[Frame * 165 + 0] = CompactRodrigues[0];
		OfflineTracker->Pose[Frame * 165 + 1] = CompactRodrigues[1];
		OfflineTracker->Pose[Frame * 165 + 2] = CompactRodrigues[2];

		OfflineTracker->Translation[Frame * 3 + 0] = Translation(0);
		OfflineTracker->Translation[Frame * 3 + 1] = Translation(1);
		OfflineTracker->Translation[Frame * 3 + 2] = Translation(2);

		// Find min (actually max!) skinned vertex position of the pose

		TArray<float> FramePose;
		FramePose.SetNumUninitialized(165);
		for (int32 Index = 0; Index < 165; ++Index)
		{
			FramePose[Index] = OfflineTracker->Pose[Frame * 165 + Index];
		}

		Impl->SMPLX->SetPose(FramePose);

		const TArray<FVector>& SkinnedVerts = Impl->SMPLX->GetSkinnedVerticesSMPL();
		float MaxY = SkinnedVerts[0].Y;
		for (const FVector& Vert : SkinnedVerts)
		{
			if (Vert.Y > MaxY)
			{
				MaxY = Vert.Y;
			}
		}
		MaxY += Translation(1);

		if (Frame == 0 || MaxY > GlobalMaxY)
		{
			GlobalMaxY = MaxY;
		}
	}

	// Pass 2: Transform from camera coord to SMPLX's Y-up world coord. Negate Y and Z axis.
	// Also place animation on groundplane by translating by GlobalMaxY.

	Eigen::Matrix<float, 3, 3> Flip;
	Flip << 1, 0, 0,  0, -1, 0,  0, 0, -1;

	for (int32 Frame = 0; Frame < HpsNumFrames; ++Frame)
	{
		if (!OfflineTracker->ValidFrame[Frame])
		{
			continue;
		}

		// Pelvis pose
		Eigen::Vector3f Axis(OfflineTracker->Pose[Frame * 165 + 0], OfflineTracker->Pose[Frame * 165 + 1], OfflineTracker->Pose[Frame * 165 + 2]);
		const float Angle = Axis.norm();

		Eigen::Matrix<float, 3, 3> Pose;

		if (Angle > 1e-4)
		{
			Axis.normalize();

			const Eigen::AngleAxisf AxisAngle(Angle, Axis);

			Pose = AxisAngle.toRotationMatrix();
		}
		else
		{
			Pose.setIdentity();
		}

		Pose = Flip * Pose;

		Eigen::Vector3f Translation(OfflineTracker->Translation[Frame * 3 + 0], OfflineTracker->Translation[Frame * 3 + 1], OfflineTracker->Translation[Frame * 3 + 2]);

		Translation = Flip * (Pelvis + Translation) + Eigen::Vector3f(0, GlobalMaxY, 0) - Pelvis;

		// Store modified values

		const Eigen::AngleAxisf AxisAngle = Eigen::AngleAxisf(Pose);
		const Eigen::Vector3f CompactRodrigues = AxisAngle.axis() * AxisAngle.angle();

		OfflineTracker->Pose[Frame * 165 + 0] = CompactRodrigues[0];
		OfflineTracker->Pose[Frame * 165 + 1] = CompactRodrigues[1];
		OfflineTracker->Pose[Frame * 165 + 2] = CompactRodrigues[2];

		OfflineTracker->Translation[Frame * 3 + 0] = Translation(0);
		OfflineTracker->Translation[Frame * 3 + 1] = Translation(1);
		OfflineTracker->Translation[Frame * 3 + 2] = Translation(2);
	}

	OutGlobalMaxY = GlobalMaxY;

	return true;
}

bool FOfflineBodyTrackerFinalizeNode::End(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!Performance || !OfflineBodyTrackerNode || !OfflineBodyTrackerNode->OfflineCameraCalibrationNode)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::NoPerformance);
		InPipelineData->SetErrorNodeMessage(TEXT("No performance or nodes set"));
		return false;
	}

	TArray<TArray<uint8>> ChunkDataArray = FMetaHumanBodyTracker::GetChunkDataArray(Performance->AdditionalBodyTrackerData);

	int32 SplitProcessingChunkIndex = FMetaHumanBodyTracker::GetChunkDataIndex(ChunkDataArray, FMetaHumanBodyTracker::SplitProcessingChunkHeader);
	if (SplitProcessingChunkIndex != -1)
	{
		ChunkDataArray.RemoveAt(SplitProcessingChunkIndex);
	}

	FMetaHumanBodyTracker::FSplitProcessingChunk SplitProcessingChunk;

	SplitProcessingChunk.AverageShape = AverageShape;

	SplitProcessingChunk.FocalLength = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->FocalLength;
	SplitProcessingChunk.Pitch = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->Pitch;
	SplitProcessingChunk.Roll = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->Roll;
	SplitProcessingChunk.BodyTrackingRasterWidth = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->BodyTrackingRasterWidth;
	SplitProcessingChunk.BodyTrackingRasterHeight = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->BodyTrackingRasterHeight;
	SplitProcessingChunk.CameraImgCenter = OfflineBodyTrackerNode->OfflineCameraCalibrationNode->CameraImgCenter;

	SplitProcessingChunk.Origin = Origin;

	TArray<uint8> SplitProcessingChunkData;
	SplitProcessingChunk.Write(SplitProcessingChunkData);

	ChunkDataArray.Add(SplitProcessingChunkData);

	FMetaHumanBodyTracker::SetChunkDataArray(ChunkDataArray, Performance->AdditionalBodyTrackerData);

	return true;
}

}
