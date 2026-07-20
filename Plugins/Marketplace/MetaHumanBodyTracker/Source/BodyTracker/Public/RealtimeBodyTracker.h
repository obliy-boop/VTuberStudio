// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "BodyTrackerModule.h"
#include "Containers/ContainersFwd.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeRDG.h"
#include "Pipeline/DataTreeTypes.h"
#include "RuntimePreference.h"

#define USE_TRT 0
#define USE_TRT_RDG 0

namespace UE::BodyTracker
{
	using UE::MetaHuman::Pipeline::FUEImageDataType;

	class BODYTRACKER_API FRealtimeBodyTracker
	{
	public:

		static TUniquePtr<FRealtimeBodyTracker> Make(int ImageSize,
			const TArray<float>& CameraIntrinsics, const TArray<float>& BodyShape, const TArray<float>& BodyHeight,
			UNNEModelData* Backbone, UNNEModelData* Camera, UNNEModelData* MFVHead,
			TConstArrayView<FRuntimePreference> RuntimePreferences);
		void Run(const FUEImageDataType& Image);

		void ClearResults()
		{
			TransformResult.SetNum(ImageSize * ImageSize * 3);
			NormalizeResult.SetNum(ImageSize * ImageSize * 3);
			ImageFeatures.SetNum(1 * 512 * 64 * 64);
			Output[0].SetNum(1 * 9);  //GlobalOrient
			Output[1].SetNum(21 * 9); //BodyPose
			Output[2].SetNum(15 * 9); //LeftHand
			Output[3].SetNum(15 * 9); //RightHand
			Output[4].SetNum(10);     //Betas
			Output[5].SetNum(9);      //Joint contact
			Output[6].SetNum(12);	  //Kp uncertainty
			Output[7].SetNum(3);      //PredCam
			Output[8].SetNum(2);      //FocalLength
		}

		TArray64<uint8> TransformResult;
		TArray64<float> NormalizeResult;
		TArray64<float> ImageFeatures;
		TArray64<float> CameraEmbeddings;
		TArray<TArray64<float>> Output;

		TArray64<float> InBackboneIn;
		TArray64<float> InBackboneOut;
		TArray64<float> InCameraEmbeddings;



	private:

		FRealtimeBodyTracker(int ImageSize, const TArray<float>& CameraIntrinsics, const TArray<float>& BodyShape, const TArray<float>& BodyHeight,
#if USE_TRT
#if USE_TRT_RDG
			TSharedRef<UE::NNE::IModelInstanceRDG> Backbone, TSharedRef<UE::NNE::IModelInstanceRDG> Camera, TSharedRef<UE::NNE::IModelInstanceRDG> MFVHead)
#else
			TSharedRef<UE::NNE::IModelInstanceGPU> Backbone, TSharedRef<UE::NNE::IModelInstanceGPU> Camera, TSharedRef<UE::NNE::IModelInstanceGPU> MFVHead)
#endif
#else
			TSharedRef<UE::NNE::IModelInstanceRDG> Backbone, TSharedRef<UE::NNE::IModelInstanceCPU> Camera, TSharedRef<UE::NNE::IModelInstanceRunSync> MFVHead)
#endif
			: ImageSize{ ImageSize }, CameraIntrinsics{ CameraIntrinsics }, BodyShape{ BodyShape }, BodyHeight{ BodyHeight },
			Backbone{ Backbone }, Camera{ Camera }, MFVHead{ MFVHead }

		{
			Output.SetNum(9);
			ClearResults();
		}

		int ImageSize;
		TArray64<float> CameraIntrinsics;
		TArray64<float> BodyShape;
		TArray64<float> BodyHeight;

#if USE_TRT
#if USE_TRT_RDG
		TSharedRef<UE::NNE::IModelInstanceRDG> Backbone;
		TSharedRef<UE::NNE::IModelInstanceRDG> Camera;
		TSharedRef<UE::NNE::IModelInstanceRDG> MFVHead;
#else
		TSharedRef<UE::NNE::IModelInstanceGPU> Backbone;
		TSharedRef<UE::NNE::IModelInstanceGPU> Camera;
		TSharedRef<UE::NNE::IModelInstanceGPU> MFVHead;
#endif
#else
		TSharedRef<UE::NNE::IModelInstanceRDG> Backbone;
		TSharedRef<UE::NNE::IModelInstanceCPU> Camera;
		TSharedRef<UE::NNE::IModelInstanceRunSync> MFVHead;
#endif
	};
} // namespace UE::BodyTracker
