// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/RealtimeBodyTrackerNode.h"

#include "MetaHumanBodyTrackerLog.h"
#include "RealtimeBodyTracker.h"
#include "MetaHumanSMPLX.h"

#include "UObject/Package.h"

#include "SupressWarnings.h"

MH_DISABLE_EIGEN_WARNINGS
#include "Eigen/Dense"
MH_ENABLE_WARNINGS



namespace UE::MetaHuman::Pipeline
{

class FRealtimeBodyTrackerNodeInternal
{
public:

	TUniquePtr<UE::BodyTracker::FRealtimeBodyTracker> Tracker;
};



static void RotationMatrixToCompactRodrigues(float* InData, float* OutData)
{
	Eigen::Matrix<float, 3, 3> RotMatrix;

	for (int32 Row = 0; Row < 3; ++Row)
	{
		for (int32 Col = 0; Col < 3; ++Col)
		{
			RotMatrix(Row, Col) = InData[Row * 3 + Col];
		}
	}

	const Eigen::AngleAxisf AxisAngle = Eigen::AngleAxisf(RotMatrix);
	const Eigen::Vector3f CompactRodrigues = AxisAngle.axis() * AxisAngle.angle();

	OutData[0] = CompactRodrigues[0];
	OutData[1] = CompactRodrigues[1];
	OutData[2] = CompactRodrigues[2];
}



FRealtimeBodyTrackerNode::FRealtimeBodyTrackerNode(const FString& InName) : FNode("RealtimeBodyTracker", InName)
{
	Impl = MakePimpl<FRealtimeBodyTrackerNodeInternal>();

	Pins.Add(FPin("Image In", EPinDirection::Input, EPinType::UE_Image));
	Pins.Add(FPin("Animation Out", EPinDirection::Output, EPinType::Animation));

	if (!SMPLX->IsInitialized()) // Message already written to log
	{
		return;
	}

	Backbone = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Realtime/backbone.backbone")));
	if (!Backbone)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load backbone model");
		return;
	}

	Camera = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Realtime/camera_embedding.camera_embedding")));
	if (!Camera)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load camera model");
		return;
	}

	MFVHead = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Realtime/mfv_head_withoutSigmoid.mfv_head_withoutSigmoid")));
	if (!MFVHead)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load MFV head model");
		return;
	}

	const int32 ImageSize = 1024;
	const TArray<float> CameraIntrinsics = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	const TArray<float> BodyShape = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	const TArray<float> BodyHeight = { 0, 0 };
	static const TArray<UE::BodyTracker::FRuntimePreference> RuntimePreferences =
	{
		{ TEXT("NNERuntimeORTDml"), UE::BodyTracker::EDeviceType::GPU },
		{ TEXT("NNERuntimeORTCpu"), UE::BodyTracker::EDeviceType::CPU }
	};
	Impl->Tracker = UE::BodyTracker::FRealtimeBodyTracker::Make(ImageSize, CameraIntrinsics, BodyShape, BodyHeight, Backbone.Get(), Camera.Get(), MFVHead.Get(), RuntimePreferences);
	if (!Impl->Tracker)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to create realtime tracker");
		return;
	}
}

bool FRealtimeBodyTrackerNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!Impl->Tracker)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize"));
		return false;
	}

	const FUEImageDataType& Input = InPipelineData->GetData<FUEImageDataType>(Pins[0]);

	Impl->Tracker->Run(Input);

	// Transform model output from camera coord to SMPLX's Y-up world coord. Negate Y and Z axis of position and orientation
	Impl->Tracker->Output[5][1] *= -1;
	Impl->Tracker->Output[5][2] *= -1;
	for (int32 Index = 3; Index < 9; ++Index)
	{
		Impl->Tracker->Output[0][Index] *= -1;
	}

	FFrameAnimationData Output;

	// Shape
	Output.RawBodyAnimationSMPLXShape.SetNumZeroed(10);
#if 0 // Use zero shape to match python implementation. Likely to change when specifying a calibration.
	for (int32 Shape = 0; Shape < 10; ++Shape)
	{
		Output.RawBodyAnimationSMPLXShape[Shape] = Impl->InternalNode->Output[4][Shape];
	}
#endif

	// Pose
	Output.RawBodyAnimationSMPLXPose.SetNumZeroed(165);

	RotationMatrixToCompactRodrigues(&Impl->Tracker->Output[0][0], &Output.RawBodyAnimationSMPLXPose[0]); // Global orientation

	for (int32 Joint = 0; Joint < 21; ++Joint) // Body pose
	{
		RotationMatrixToCompactRodrigues(&Impl->Tracker->Output[1][Joint * 9], &Output.RawBodyAnimationSMPLXPose[(Joint + 1) * 3]);
	}

	for (int32 Joint = 0; Joint < 15; ++Joint) // Left hand pose
	{
		RotationMatrixToCompactRodrigues(&Impl->Tracker->Output[2][Joint * 9], &Output.RawBodyAnimationSMPLXPose[(Joint + 25) * 3]);
	}

	for (int32 Joint = 0; Joint < 15; ++Joint) // Right hand pose
	{
		RotationMatrixToCompactRodrigues(&Impl->Tracker->Output[3][Joint * 9], &Output.RawBodyAnimationSMPLXPose[(Joint + 40) * 3]);
	}

	// Position
	if (bTranslation)
	{
		Output.RawBodyAnimationSMPLXTranslation = FVector(Impl->Tracker->Output[5][0], Impl->Tracker->Output[5][1], Impl->Tracker->Output[5][2]);
	}

	Output.AnimationQuality = ReportedAnimationQuality;

	PrepareOutput(Output);

	InPipelineData->SetData<FFrameAnimationData>(Pins[1], MoveTemp(Output));

	FPlatformProcess::Sleep(1.0); // Without this the game thread is overloaded and UE is unresponsive. Could be a lower value no doubt.

	return true;
}

}
