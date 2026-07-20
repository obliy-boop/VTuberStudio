// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/OfflineCameraCalibrationNode.h"

#include "BodyTrackerUtils.h"
#include "MetaHumanBodyTrackerLog.h"
#include "CameraCalibration.h"

#include "Math/IntPoint.h"

#include "UObject/Package.h"



namespace UE::MetaHuman::Pipeline
{

class FOfflineCameraCalibrationNodeInternal
{
public:

	TUniquePtr<UE::BodyTracker::FCameraCalibration> CameraCalibration;
};

FOfflineCameraCalibrationNode::FOfflineCameraCalibrationNode(const FString& InName) : FNode("OfflineCameraCalibration", InName)
{
	Impl = MakePimpl<FOfflineCameraCalibrationNodeInternal>();

	Pins.Add(FPin("Image In", EPinDirection::Input, EPinType::UE_Image));

	GeoCalib = TStrongObjectPtr<UNNEModelData>(LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/" UE_PLUGIN_NAME "/Models/Offline/camera_calib.camera_calib")));
	if (!GeoCalib)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to load GeoCalib model");
		return;
	}
	static const TArray<UE::BodyTracker::FRuntimePreference> RuntimePreferences =
	{
		{ TEXT("NNERuntimeORTDml"), UE::BodyTracker::EDeviceType::GPU },
		{ TEXT("NNERuntimeORTCpu"), UE::BodyTracker::EDeviceType::CPU }
	};
	Impl->CameraCalibration = UE::BodyTracker::FCameraCalibration::Make(GeoCalib.Get(), RuntimePreferences);
	if (!Impl->CameraCalibration)
	{
		UE_LOGF(LogMetaHumanBodyTracker, Warning, "Failed to create offline camera calibration");
		return;
	}
}

bool FOfflineCameraCalibrationNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{
	FrameCount = 0;
	FocalLength = 0;
	Pitch = 0;
	Roll = 0;
	CameraImgCenter = FVector2f(0.f, 0.f);
	BodyTrackingRasterWidth = 0;
	BodyTrackingRasterHeight = 0;

	return true;
}

bool FOfflineCameraCalibrationNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!Impl->CameraCalibration)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize"));
		return false;
	}

	if (Stride < 1)
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::BadStride);
		InPipelineData->SetErrorNodeMessage(TEXT("Bad stride value"));
		return false;
	}

	if (FrameCount % Stride == 0)
	{
		const FUEImageDataType& Input = InPipelineData->GetData<FUEImageDataType>(Pins[0]);

		// Same raster rule as ViTPose preprocess / Stage-1 image dimensions (max height 896).
		constexpr int32 ViTPosePreprocessMaxHeight = 896;
		const FIntPoint Raster = UE::BodyTracker::Utils::CalculateAdjustedSize(Input.Width, Input.Height, ViTPosePreprocessMaxHeight);
		BodyTrackingRasterWidth = Raster.X;
		BodyTrackingRasterHeight = Raster.Y;
		CameraImgCenter = FVector2f(
			static_cast<float>(BodyTrackingRasterWidth) * 0.5f,
			static_cast<float>(BodyTrackingRasterHeight) * 0.5f);

		Impl->CameraCalibration->Run(Input);

		FocalLength = Impl->CameraCalibration->Fov;
		Pitch = Impl->CameraCalibration->Pitch;
		Roll = Impl->CameraCalibration->Roll;
	}

	FrameCount++;

	return true;
}

bool FOfflineCameraCalibrationNode::End(const TSharedPtr<FPipelineData>& InPipelineData)
{
	Impl->CameraCalibration.Reset();
	return true;
}

}
