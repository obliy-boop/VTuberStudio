// Copyright Epic Games, Inc. All Rights Reserved.

#include "RealtimeBodyTracker.h"

#include "BodyTrackerUtils.h"
#include "NNE.h"
#include "NNEUtils.h"
#include "Serialization/Archive.h"
#include "String/ParseTokens.h"

namespace UE::BodyTracker
{
	
	TUniquePtr<FRealtimeBodyTracker> FRealtimeBodyTracker::Make(int ImageSize,
		const TArray<float>& CameraIntrinsics, const TArray<float>& BodyShape, const TArray<float>& BodyHeight,
		UNNEModelData* Backbone, UNNEModelData* Camera, UNNEModelData* MFVHead,
		TConstArrayView<FRuntimePreference> RuntimePreferences)
	{
		using namespace UE::NNEUtils::Private;

		if (CameraIntrinsics.Num() != 9 || BodyShape.Num() != 11 || BodyHeight.Num() != 2)
		{
			UE_LOG(LogBodyTracker, Warning, TEXT("CameraIntrinsics, BodyShape and BodyHeight need to have Num 9, 11 and 2 respectively, but %i, %i and %i where given."),
				CameraIntrinsics.Num(), BodyShape.Num(), BodyHeight.Num());
			return {};
		}
		if (!Backbone || !Camera || !MFVHead)
		{
			UE_LOG(LogBodyTracker, Warning, TEXT("Backbone, Camera and MFVHead need to be none NULL."));
			return {};
		}
#if USE_TRT
#if USE_TRT_RDG
		TSharedPtr<UE::NNE::IModelInstanceRDG> BackboneModel = CreateRDGModelInstance(*Backbone, TEXT("NNERuntimeTRT"), true);
		TSharedPtr<UE::NNE::IModelInstanceRDG> CameraModel = CreateRDGModelInstance(*Camera, TEXT("NNERuntimeTRT"), true);
		TSharedPtr<UE::NNE::IModelInstanceRDG> MFVHeadModel = CreateRDGModelInstance(*MFVHead, TEXT("NNERuntimeTRT"), true);
#else
		TSharedPtr<UE::NNE::IModelInstanceGPU> BackboneModel = CreateGPUModelInstance(*Backbone, TEXT("NNERuntimeTRT"), true);
		TSharedPtr<UE::NNE::IModelInstanceGPU> CameraModel = CreateGPUModelInstance(*Camera, TEXT("NNERuntimeTRT"), true);
		TSharedPtr<UE::NNE::IModelInstanceGPU> MFVHeadModel = CreateGPUModelInstance(*MFVHead, TEXT("NNERuntimeTRT"), true);
#endif
#else
		TSharedPtr<UE::NNE::IModelInstanceRDG> BackboneModel = CreateRDGModelInstance(*Backbone, TEXT("NNERuntimeIREERdg"), true);
		TSharedPtr<UE::NNE::IModelInstanceCPU> CameraModel = CreateCPUModelInstance(*Camera, TEXT("NNERuntimeORTCpu"), true);
		TSharedPtr<UE::NNE::IModelInstanceRunSync> MFVHeadModel = TryCreateInstance(MFVHead, RuntimePreferences, true);
#endif
		if (!BackboneModel.IsValid() || !CameraModel.IsValid() || !MFVHeadModel.IsValid())
		{
			return {};
		}
		return TUniquePtr<FRealtimeBodyTracker>(new FRealtimeBodyTracker(ImageSize, CameraIntrinsics, BodyShape, BodyHeight, BackboneModel.ToSharedRef(), CameraModel.ToSharedRef(), MFVHeadModel.ToSharedRef()));
	}



	void FRealtimeBodyTracker::Run(const FUEImageDataType& Image)
	{
		SCOPED_NAMED_EVENT_TEXT("RealtimeBodyTracker::Run", FColor::Magenta);

		using namespace Private;
		using namespace UE::NNEUtils::Private;

		Utils::FPreprocessImageArgs PreprocessArgs = 
		{
			.NewSize = {ImageSize, ImageSize},
			.Mean = {0.485f, 0.456f, 0.406f},
			.StdInv = {1/0.229f, 1/0.224f, 1/0.225f}
		};
		Utils::PreprocessImage(Image, PreprocessArgs, NormalizeResult);

		TArray<UE::NNE::FTensorBindingCPU> Inputs;
		TArray<UE::NNE::FTensorBindingCPU> Outputs;
		{
			SCOPED_NAMED_EVENT_TEXT("RealtimeBodyTracker::Backbone::RunSync", FColor::Magenta);
			Inputs = { ToTensorBindingCPU(NormalizeResult) };
			Outputs = { ToTensorBindingCPU(ImageFeatures) };
#if USE_TRT && !USE_TRT_RDG
			Backbone->RunSync(Inputs, Outputs);
#else
			RunSync(Backbone.Get(), Inputs, Outputs);
#endif
		}
		if (CameraEmbeddings.IsEmpty())
		{
			SCOPED_NAMED_EVENT_TEXT("RealtimeBodyTracker::Camera::RunSync", FColor::Magenta);
			CameraEmbeddings.SetNumUninitialized(ImageFeatures.Num());
			Inputs = { ToTensorBindingCPU(ImageFeatures),
						ToTensorBindingCPU(CameraIntrinsics) };
			Outputs = { ToTensorBindingCPU(CameraEmbeddings) };
#if USE_TRT && USE_TRT_RDG
			RunSync(Camera.Get(), Inputs, Outputs);
#else
			Camera->RunSync(Inputs, Outputs);
#endif
		}
		{
			SCOPED_NAMED_EVENT_TEXT("RealtimeBodyTracker::MFVHead::RunSync", FColor::Magenta);
			CameraEmbeddings.SetNumUninitialized(ImageFeatures.Num());
			Inputs =
			{
				ToTensorBindingCPU(ImageFeatures),
				ToTensorBindingCPU(CameraEmbeddings),
				ToTensorBindingCPU(CameraIntrinsics),
				ToTensorBindingCPU(BodyShape),
				ToTensorBindingCPU(BodyHeight),
			};
			Outputs.SetNum(Output.Num());
			for (int Index = 0; Index < Output.Num(); Index++)
			{
				Outputs[Index] = ToTensorBindingCPU(Output[Index]);
			}
#if USE_TRT && USE_TRT_RDG
			RunSync(MFVHead.Get(), Inputs, Outputs);
#else
			MFVHead->RunSync(Inputs, Outputs);
#endif
			// Apply sigmoid to joint contact because we moved this ouside the model
			for (float& Value : Output[5])
			{
				Value = 1 / (1 + FMath::Exp(-Value));
			}
		}
	}
} // namespace UE::BodyTracker
