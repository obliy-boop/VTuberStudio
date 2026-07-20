// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "BodyTrackerModule.h"
#include "BodyTrackerUtils.h"
#include "Containers/ContainersFwd.h"
#include "Containers/StaticArray.h"
#include "Detectron2.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeRDG.h"
#include "Pipeline/DataTreeTypes.h"
#include "RuntimePreference.h"
#include "SAM2.h"

//#define WINDOW_SIMULATION

namespace UE::BodyTracker
{
	using UE::MetaHuman::Pipeline::FUEImageDataType;
	
	class BODYTRACKER_API FEstimateKeypoints
	{
	public:
		static TUniquePtr<FEstimateKeypoints> Make(
			UNNEModelData* ViTPose, 
			UNNEModelData* ViTPosePost, 
			TConstArrayView<FRuntimePreference> RuntimePreferences);

		void Run(const UE::MetaHuman::Pipeline::FUEImageDataType& Frame, 
				 const FIntRect& InBox,
				 TArray64<float>& OutKeypoints);

		static void Op25FromKeypoints(TConstArrayView64<float> Keypoints, TArray64<float>& OutOp25);

		static constexpr int NumFloatsPerKeypoint = 133 * 3;
		static constexpr int NumFloatsPerOp25 = 25 * 3;

	private:

		FEstimateKeypoints(TSharedRef<UE::NNE::IModelInstanceRunSync> ViTPose, TSharedRef<UE::NNE::IModelInstanceRunSync> ViTPosePost) : ViTPose{ ViTPose }, ViTPosePost{ViTPosePost}
		{
			ViTPoseIn.SetNum(3 * 256 * 192);
			ViTPoseOut.SetNum(133 * 64 * 48);
			ViTPosePostOut.SetNum(NumFloatsPerKeypoint);
		}

		TArray64<float> ViTPoseIn;
		TArray64<float> ViTPoseOut;
		TArray64<float> ViTPosePostOut;

		TSharedRef<UE::NNE::IModelInstanceRunSync> ViTPose;
		TSharedRef<UE::NNE::IModelInstanceRunSync> ViTPosePost;
	};

	class BODYTRACKER_API FCHMR
	{
	public:

		static TUniquePtr<FCHMR> Make(
			UNNEModelData* Backbone, 
			UNNEModelData* Head, 
			TConstArrayView<FRuntimePreference> RuntimePreferences);

		static void KeypointsToPromptBodyPoints(TConstArrayView64<float> Keypoints, TArray64<float>& PromptBodyPoints, float Scale, const FVector2f& Offset);

		void Run(
			const UE::MetaHuman::Pipeline::FUEImageDataType& InFrame, 
			const FIntRect& Box, 
			TConstArrayView64<float> Keypoints, 
			float CameraImgFocal,
			TArray64<float>& OutHuman3DTokens);

	private:

		static void BoxToPromptPoints(const FIntRect& InBox, TArray64<float>& PromptPoints, float Scale, const FVector2f& Offset);
		FCHMR(TSharedRef<UE::NNE::IModelInstanceRunSync> Backbone, TSharedRef<UE::NNE::IModelInstanceRunSync> Head) : Backbone{ Backbone }, Head{ Head }
		{
			BackboneOut.SetNum(BackboneCount);
		}

		static constexpr int BackboneCount = 1024 * 64 * 64;
		static constexpr int Human3DTokensCount = 8 * 1024;

		TArray64<float> BackboneIn;
		TArray64<float> BackboneOut;
		TArray64<float> CameraIntrinsicsInv;
		TArray64<float> PromptPoints;
		TArray64<float> PromptBodyKpts;

		TSharedRef<UE::NNE::IModelInstanceRunSync> Backbone;
		TSharedRef<UE::NNE::IModelInstanceRunSync> Head;
	};

	class BODYTRACKER_API FHue
	{
	public:

		struct FRunInfo
		{
			int RelFirstMLFrame;
			int RelLastMLFrame;
			int AbsFirstMLFrame;
			int AbsLastMLFrame;
			int FirstValidFrame;
			int LastValidFrame;
			bool LastWindow;
			int OutputFrameCount;
			int LeftShiftCount;
		};

		static TUniquePtr<FHue> Make(
			UNNEModelData* Step, 
			UNNEModelData* FinalStep,
			TConstArrayView<FRuntimePreference> RuntimePreferences);

		void EnqueInvalidFrame();

		bool EnqueValidFrame(
			float CameraImgFocal, 
			const FIntVector2& FrameSize, 
			TConstArrayView64<float>  Human3DTokens,
			const FIntRect& InBox,
			TConstArrayView64<float>  WholeBody,
			float BodyHeight = 0.0f, // Body height in cm, or 0.0f if unknown
			TConstArrayView64<float> InNoise = {});

		FRunInfo Run(
			     TArray64<float>& OutSmplxPose,
			     TArray64<float>& OutTrans,
				 TArray64<float>& OutBetas,
			     TArray64<float>& OutStaticConfLogits);
	private:

		void AppendRandn(TArray64<float>& OutArray, int Count);
		static void MedianFilter(TConstArrayView64<double> InValues, TArrayView64<double> OutValues);
		static void GaussianFilter(TConstArrayView64<double> InValues, TArrayView64<float> OutValues);
		static void GetNormalizedVitposes(TConstArrayView64<float> InWholebodys, TConstArrayView64<float> BbxXYS, TArrayView64<float> OutVitposes);

		static constexpr int64 InferenceStepCount = 50;
		static constexpr int64 TrainingStepCount = 1000;
		static constexpr int64 StepRatio = TrainingStepCount / InferenceStepCount;

		static constexpr int MedianKernelSize = 11;
		static constexpr int MedianPadding = (MedianKernelSize - 1) / 2;
		static constexpr int GaussianKernelSize = 25;
		static constexpr int GaussianPadding = (GaussianKernelSize - 1) / 2;
		static constexpr int ExtraPadding = 4;
		static constexpr int ContextPadding = 32;
		static constexpr int InnerWindow = 192;
		static constexpr int SmoothPadding = MedianPadding + GaussianPadding;
		static constexpr int TotalPadding = MedianPadding + GaussianPadding + ExtraPadding + ContextPadding;
		static constexpr int MaxWindowSize = TotalPadding + InnerWindow + TotalPadding;
		static constexpr int FirstWindowSize = ContextPadding + InnerWindow + TotalPadding;
		static constexpr int MaxMLWindowSize = ExtraPadding + ContextPadding + InnerWindow + ContextPadding + ExtraPadding;

		static constexpr int Human3dTokensElementCount = 1024;
		static constexpr int Human3DTokensCount = 5 * Human3dTokensElementCount;

		static constexpr int BbxXYSCount = 3;
		static constexpr int VitposeCount = 17 * 3;
		static constexpr int NoiseCount = 331;
		static constexpr int CliffCamCount = 3;
		static constexpr int CamAngvelCount = 6;
		static constexpr int CameraIntrinsicsCount = 9;
		static constexpr int SmplxPoseCount = 165;
		static constexpr int BetasCount = 10;
		static constexpr int TransCount = 3;
		static constexpr int StaticConfLogitsCount = 6;

		static constexpr int MaxImageHeight = 896;

		static constexpr TStaticArray<int, 5> Human3DTokensKeepIndices = { 0, 1, 2, 3, 5 };
		inline static const TArray<double> GaussianKernel = Utils::CreateGaussianFilter(3, GaussianKernelSize);

		bool FirstWindow = true;
		int EnquedFrameCount = 0;
		int MLFrameOffset = 0;

		// Values stored during EnqueFrame
		TArray64<bool>   ValidFrames;        // (FrameCount)
		TArray64<float>  CameraImgFocals;    // (1, ValidFrameCount, 1)
		TArray64<FVector2f> CameraImgCenters;// (1, ValidFrameCount, 1)
		TArray64<float>  Human3DTokens;      // (1, ValidFrameCount, 5, 1024)
		TArray64<double> BbxXYS;             // (1, ValidFrameCount, 3)
		TArray64<float>  VitposeRaw;         // (1, ValidFrameCount, 17, 3)
		TArray64<float>  CameraIntrinsics;   // (1, ValidFrameCount, 9)
		TArray64<float>  Noise;              // (1, ValidFrameCount, 331)
		TArray64<float>  BodyHeights;        // (1, ValidFrameCount, 1)

		// Temporary buffers used during Run
		TArray64<double> TmpBbxXYS;          // (1, ValidFrameCount, 3)
		TArray64<float>  SmoothBbxXYS;       // (1, ValidFrameCount, 3)
		TArray64<float>  VitposeNormalized;  // (1, ValidFrameCount, 17, 3)
		TArray64<float>  CliffCam;           // (1, ValidFrameCount, 3)
		TArray64<float>  CamAngvel;          // (1, ValidFrameCount, 6)
		TArray64<float> PrevSample;

		// Output buffers
		TArray64<float> SmplxPose;
		TArray64<float> Betas;
		TArray64<float> Transl;
		TArray64<float> StaticConfLogits;


		int SetFrameCount = 0;
		TSharedRef<UE::NNE::IModelInstanceRunSync> Step;
		TSharedRef<UE::NNE::IModelInstanceRunSync> FinalStep;
		FRandomStream Stream{ FName(TEXT("HueRandn")) };


		FHue(TSharedRef<UE::NNE::IModelInstanceRunSync> Step, TSharedRef<UE::NNE::IModelInstanceRunSync> FinalStep) : Step{ Step }, FinalStep{ FinalStep }
		{
			Human3DTokens.Reserve(MaxWindowSize * Human3DTokensCount);
			BbxXYS.Reserve(MaxWindowSize * BbxXYSCount);
			VitposeRaw.Reserve(MaxWindowSize * VitposeCount);
			Noise.Reserve(MaxWindowSize * NoiseCount);
			TmpBbxXYS.Reserve(MaxWindowSize * BbxXYSCount);
			SmoothBbxXYS.Reserve(MaxWindowSize * BbxXYSCount);
			VitposeNormalized.Reserve(MaxWindowSize * VitposeCount);
			CamAngvel.Init(0, MaxWindowSize * CamAngvelCount);
		}

	};
	

	class BODYTRACKER_API FOfflineBodyTracker
	{
	public:

		static TUniquePtr<FOfflineBodyTracker> Make(
			UNNEModelData* ViTPose, 
			UNNEModelData* ViTPosePost,
			UNNEModelData* CHMRBackbone, 
			UNNEModelData* CHMRHead,
			UNNEModelData* HueStep, 
			UNNEModelData* HueFinalStep
		);

		void SetCameraFocal(const float Focal) { CameraImgFocal = Focal; }

		bool EnqueFrame(const FUEImageDataType& Image);

		bool GetResults(
			TArray64<bool>&  OutValidFrames,		// Seq * 1
			TArray64<float>& OutBBox,				// Seq * 4 (x0, y0, x1, y1)
			TArray64<float>& OutKeypoints2D,		// Seq * 25 * 3 (x, y, conf)
			TArray64<float>& OutSmplxPose,			// Seq * 165 (axis angles)
			TArray64<float>& OutSmplxTrans,			// Seq * 3
			TArray64<float>& OutSmplxBetas,			// Seq * 10
			TArray64<float>& OutStaticConfLogits	// Seq * 6
		);

		static constexpr int MaxImageHeight = 896;

		TUniquePtr<FEstimateKeypoints> EstimateKeypoints;
		TUniquePtr<FCHMR> CHMR;
		TUniquePtr<FHue> Hue;

	private:

		FOfflineBodyTracker(
			TUniquePtr<UE::NNE::Segmentation::IDetector> PersonDetector,
			TUniquePtr<UE::NNE::Segmentation::ITracker> PersonTracker,
			TUniquePtr<FEstimateKeypoints> EstimateKeypoints,
			TUniquePtr<FCHMR> CHMR,
			TUniquePtr<FHue> Hue
		) :
			EstimateKeypoints{ MoveTemp(EstimateKeypoints) },
			CHMR{ MoveTemp(CHMR) },
			Hue{ MoveTemp(Hue) },
			PersonDetector{ MoveTemp(PersonDetector) },
			PersonTracker{ MoveTemp(PersonTracker) }
		{ }

		static constexpr int Keypoints2DCount = 25 * 3;
		static constexpr int BBoxesCount = 4;

		TUniquePtr<UE::NNE::Segmentation::IDetector> PersonDetector;
		TUniquePtr<UE::NNE::Segmentation::ITracker> PersonTracker;

		int FrameIndex = 0;
		TArray64<bool> ValidFrames;
		TArray64<float> BBoxes;
		TArray64<float> Keypoints2D;

		bool bTrackerInitialized = false;
		float ImageScale;
		int AdjustedImageHeight;
		float CameraImgFocal;

	};
} //namespace UE::BodyTracker
