// Copyright Epic Games, Inc. All Rights Reserved.

#include "OfflineBodyTracker.h"

#include "BodyTrackerModule.h"
#include "NNEModelData.h"
#include "NNEUtils.h"
#include "NNETypes.h"
#include "Serialization/Archive.h"
#include "String/ParseTokens.h"

using namespace UE::NNEUtils::Private;

namespace UE::BodyTracker
{

	void ScalePoints(TArray64<float>&Points, const FVector2f OriginalSize, const FVector2f CurrentSize, const FVector2f Offset)
	{
		check(Points.Num() % 3 == 0);

		FVector2f Scale = OriginalSize / (CurrentSize - 1);

		for (int Index = 0; Index < Points.Num(); Index += 3)
		{
			FVector2f Point{ Points[Index], Points[Index + 1] };
			Point *= Scale;
			Point += Offset;
			Points[Index] = Point.X;
			Points[Index + 1] = Point.Y;
		}
	}

	TUniquePtr<FEstimateKeypoints> FEstimateKeypoints::Make(
		UNNEModelData* ViTPose, 
		UNNEModelData* ViTPosePost,
		TConstArrayView<FRuntimePreference> RuntimePreferences)
	{
		TSharedPtr<UE::NNE::IModelInstanceRunSync> ViTPoseModel = TryCreateInstance(ViTPose, RuntimePreferences, true);
		TSharedPtr<UE::NNE::IModelInstanceRunSync> ViTPosePostModel = TryCreateInstance(ViTPosePost, RuntimePreferences, true);
		if (!ViTPoseModel || !ViTPosePostModel)
		{
			return {};
		}
		return TUniquePtr<FEstimateKeypoints>{new FEstimateKeypoints(ViTPoseModel.ToSharedRef(), ViTPosePostModel.ToSharedRef())};
	}

	void FEstimateKeypoints::Run(
		const UE::MetaHuman::Pipeline::FUEImageDataType& Frame, 
		const FIntRect& InBox,
		TArray64<float>& OutKeypoints)
	{
		SCOPED_NAMED_EVENT_TEXT("FEstimateKeypoints::Run", FColor::Orange);

		const Utils::FPreprocessImageArgs PreprocessArgs =
		{
			.MaxHeight = 896,
			.Crop = InBox,
			.SampleOffset = {0.5, 0.5},
			.NewSize = {192, 256},
			.Mean = {0.485f, 0.456f, 0.406f},
			.StdInv = {1/0.229f, 1/0.224f, 1/0.225f}
		};
		Utils::FPreprocessImageInfo PreprocessInfo = Utils::PreprocessImage(Frame, PreprocessArgs, ViTPoseIn);

		TArray<UE::NNE::FTensorBindingCPU> Inputs{ ToTensorBindingCPU(ViTPoseIn) };
		TArray<UE::NNE::FTensorBindingCPU> Outputs{ ToTensorBindingCPU(ViTPoseOut) };
		ViTPose->RunSync(Inputs, Outputs);


		Inputs = { ToTensorBindingCPU(ViTPoseOut) };
		Outputs = { ToTensorBindingCPU(ViTPosePostOut) };
		ViTPosePost->RunSync(Inputs, Outputs);

		const FIntPoint OriginalSize = (FVector2f(PreprocessArgs.NewSize) * PreprocessInfo.Scale).IntPoint();
		ScalePoints(ViTPosePostOut, OriginalSize, { 48, 64 }, PreprocessInfo.Offset);
		OutKeypoints.Append(ViTPosePostOut);
	}

	void FEstimateKeypoints::Op25FromKeypoints(TConstArrayView64<float> Keypoints, TArray64<float>& OutOp25)
	{
		static const TArray<int> Map{ 0, -1, 6, 8, 10, 5, 7, 9, -1, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 18, 19, 20, 21, 22 };

		check(Keypoints.Num() % NumFloatsPerKeypoint == 0);
		int NumKeypoints = Keypoints.Num() / NumFloatsPerKeypoint;
		for (int64 KeypointIndex = 0; KeypointIndex < NumKeypoints; KeypointIndex++)
		{
			int64 BaseIndex = KeypointIndex * NumFloatsPerKeypoint;
			for (int Index : Map)
			{
				if (Index < 0)
				{
					OutOp25.AddZeroed(3);
				}
				else
				{
					OutOp25.Add(Keypoints[BaseIndex + Index * 3 + 0]);
					OutOp25.Add(Keypoints[BaseIndex + Index * 3 + 1]);
					OutOp25.Add(Keypoints[BaseIndex + Index * 3 + 2]);
				}
			}
		}
	}

	TUniquePtr<FCHMR> FCHMR::Make(
		UNNEModelData* Backbone,
		UNNEModelData* Head,
		TConstArrayView<FRuntimePreference> RuntimePreferences)
	{
		TSharedPtr<UE::NNE::IModelInstanceRunSync> BackboneModel = TryCreateInstance(Backbone, RuntimePreferences, true);
		// Prefer ORT GPU for accuracy; ORT CPU is the fp32 fallback when DML is unavailable (e.g. Mac/Linux).
		TSharedPtr<UE::NNE::IModelInstanceRunSync> HeadModel = TryCreateInstance(Head, RuntimePreferences, true);
		if (!BackboneModel || !HeadModel)
		{
			return {};
		}
		return TUniquePtr<FCHMR>{new FCHMR(BackboneModel.ToSharedRef(), HeadModel.ToSharedRef())};
	}

	void FCHMR::KeypointsToPromptBodyPoints(TConstArrayView64<float> Keypoints, TArray64<float>& PromptBodyPoints, float Scale, const FVector2f& Offset)
	{
		check(Keypoints.Num() == 133 * 3);
		static const TArray<int> Map{ 0, 2, 1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 17, 19, 20, 22 };

		PromptBodyPoints.Empty(Map.Num() * 3);
		for (int SrcIndex : Map)
		{
			FVector2f Point{-1, -1};
			float Confidence = 0;
			if (SrcIndex >= 0)
			{
				Point.X    = Keypoints[SrcIndex * 3 + 0];
				Point.Y    = Keypoints[SrcIndex * 3 + 1];
				Confidence = Keypoints[SrcIndex * 3 + 2];
			}
			Point = Point * Scale + Offset;
			Point = Point / 896;  //normalize values into range (0, 1)
			Confidence = Confidence > 0.7 ? 1 : 0;

			PromptBodyPoints.Add(Point.X);
			PromptBodyPoints.Add(Point.Y);
			PromptBodyPoints.Add(Confidence);
		}
	}

	void FCHMR::BoxToPromptPoints(const FIntRect& InBox, TArray64<float>& PromptPoints, float Scale, const FVector2f& Offset)
	{
		PromptPoints.Empty(5);
		TArray<FVector2f, TInlineAllocator<2>> Points{ InBox.Min, InBox.Max };
		for (FVector2f& Point : Points)
		{
			Point = Point * Scale + Offset;
			Point = Point / 896;  //normalize values into range (0, 1)
			PromptPoints.Add(Point.X);
			PromptPoints.Add(Point.Y);
		}
		PromptPoints.Add(1);
	}

	void FCHMR::Run(const UE::MetaHuman::Pipeline::FUEImageDataType& InFrame, 
					const FIntRect& Box, 
					TConstArrayView64<float> Keypoints, 
					float CameraImgFocal,
					TArray64<float>& OutHuman3DTokens)
	{
		SCOPED_NAMED_EVENT_TEXT("FCHMR::Run", FColor::Orange);

		const Utils::FPreprocessImageArgs PreprocessArgs =
		{
			.MaxHeight = 896,
			.NewSize = {896, 896},
			.Mean = {0.485f, 0.456f, 0.406f},
			.StdInv = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f}
		};
		Utils::FPreprocessImageInfo PreprocessInfo = Utils::PreprocessImage(InFrame, PreprocessArgs, BackboneIn);

		TArray<UE::NNE::FTensorBindingCPU> Inputs{ ToTensorBindingCPU(BackboneIn) };
		TArray<UE::NNE::FTensorBindingCPU> Outputs{ ToTensorBindingCPU(BackboneOut) };
		Backbone->RunSync(Inputs, Outputs);

		float Scale = 1 / PreprocessInfo.Scale;
		FVector2f Offset = -PreprocessInfo.Offset * Scale;
		BoxToPromptPoints(Box, PromptPoints, Scale, Offset);
		KeypointsToPromptBodyPoints(Keypoints, PromptBodyKpts, Scale, Offset);
		const FIntPoint ImageSize = Utils::CalculateAdjustedSize(InFrame.Width, InFrame.Height, 896);
		const FVector2f CameraImgCenter = FVector2f(ImageSize) / 2;
		float Focal = CameraImgFocal / CameraImgCenter.GetMax() / 2;
		float FocalInv = 2 * CameraImgCenter.GetMax() / CameraImgFocal;
		CameraIntrinsicsInv.Init(0, 9);
		CameraIntrinsicsInv[0] = FocalInv;
		CameraIntrinsicsInv[2] = FocalInv * (-0.5);
		CameraIntrinsicsInv[4] = FocalInv;
		CameraIntrinsicsInv[5] = FocalInv * (-0.5);
		CameraIntrinsicsInv[8] = 1;
		int64 Human3DTokensStart = OutHuman3DTokens.Num();
		OutHuman3DTokens.AddUninitialized(Human3DTokensCount);
		Inputs = {
			ToTensorBindingCPU(BackboneOut),
			ToTensorBindingCPU(CameraIntrinsicsInv),
			ToTensorBindingCPU(PromptPoints),
			ToTensorBindingCPU(PromptBodyKpts)
		};
		Outputs = { 
			ToTensorBindingCPU(MakeArrayView64(OutHuman3DTokens).Slice(Human3DTokensStart, Human3DTokensCount))
		};
		Head->RunSync(Inputs, Outputs);
	}

	void FHue::MedianFilter(TConstArrayView64<double> InValues, TArrayView64<double> OutValues)
	{
		constexpr int Stride = 3;
		check(InValues.Num() == OutValues.Num());
		check(InValues.Num() % Stride == 0);

		TArray<double, TInlineAllocator<MedianKernelSize>> Kernel;
		Kernel.Init(0, MedianKernelSize);
		for (int ValueBase = 0; ValueBase < InValues.Num(); ValueBase++)
		{
			for (int KernelIndex = 0; KernelIndex < MedianKernelSize; KernelIndex++)
			{
				const int ValueIndex = ValueBase + (KernelIndex - MedianPadding) * Stride;
				Kernel[KernelIndex] = ValueIndex < 0 || ValueIndex >= InValues.Num() ? 0 : InValues[ValueIndex];
			}
			Kernel.Sort();
			OutValues[ValueBase] = Kernel[MedianPadding];
		}
	}

	void FHue::GaussianFilter(TConstArrayView64<double> InValues, TArrayView64<float> OutValues)
	{
		constexpr int Stride = 3;
		check(InValues.Num() == OutValues.Num());
		check(InValues.Num() % Stride == 0);
		const int ElementCount = InValues.Num() / Stride;
		for (int ElementIndex = 0; ElementIndex < ElementCount; ElementIndex++)
		{
			for (int ElementOffset = 0; ElementOffset < Stride; ElementOffset++)
			{
				double Sum = 0;
				for (int KernelIndex = 0; KernelIndex < GaussianKernelSize; KernelIndex++)
				{
					int ValueIndex = ElementIndex + (KernelIndex - GaussianPadding);
					// Reflect index
					while (ValueIndex < 0 || ValueIndex >= ElementCount)
					{
						ValueIndex = ValueIndex < 0 ? -(ValueIndex + 1) : ValueIndex;
						ValueIndex = ValueIndex >= ElementCount ? 2 * ElementCount - 1 - ValueIndex : ValueIndex;
					}
					Sum += InValues[ValueIndex * Stride + ElementOffset] * GaussianKernel[KernelIndex];
				}
				OutValues[ElementIndex * Stride + ElementOffset] = Sum;
			}
		}
	}

	FVector3d BoxToXYS(const FIntRect& Box)
	{
		FVector2d Center = (FVector2d(Box.Min) + FVector2d(Box.Max)) * 0.5;
		FVector2d Size = Box.Size();
		double BbxSize = FMath::Max(Size.X / 0.75, Size.Y) * 1.2;
		FVector3d Result(Center, BbxSize);
		return Result;
	}

	void FHue::GetNormalizedVitposes(TConstArrayView64<float> InWholebodies, TConstArrayView64<float> BbxXYS, TArrayView64<float> OutVitposes)
	{
		check(BbxXYS.Num() % 3 == 0);
		int FrameCount = BbxXYS.Num() / 3;
		check(InWholebodies.Num() == FrameCount * VitposeCount);
		check(OutVitposes.Num() == FrameCount * VitposeCount);
		for (int FrameIndex = 0; FrameIndex < FrameCount; FrameIndex++)
		{
			FVector2D Center(BbxXYS[FrameIndex * 3 + 0], BbxXYS[FrameIndex * 3 + 1]);
			float Size = BbxXYS[FrameIndex * 3 + 2];
			FVector2D Extend(Size * 0.5);
			FBox2d Box = FBox2d::BuildAABB(Center, Extend);
			for (int PointIndex = 0; PointIndex < 17; PointIndex++)
			{
				int ArrayIndex = (FrameIndex * 17 + PointIndex) * 3;
				FVector2D Point(InWholebodies[ArrayIndex + 0], InWholebodies[ArrayIndex + 1]);
				double Confidence = Box.IsInside(Point) ? InWholebodies[ArrayIndex + 2] : 0;
				Point = 2 * (Point - Center) / Size;
				OutVitposes[ArrayIndex + 0] = Point.X;
				OutVitposes[ArrayIndex + 1] = Point.Y;
				OutVitposes[ArrayIndex + 2] = Confidence;
			}
		}
	}

	void CalculateCliffCam(
		TArray64<float>& OutCliffCam,
		TConstArrayView64<float> SmoothBbxXYS,
		TConstArrayView64<float> CameraImgFocals,
		TConstArrayView64<FVector2f> CameraImgCenters)
	{
		const int ElementCount = CameraImgFocals.Num();
		check(SmoothBbxXYS.Num() == ElementCount * 3);
		check(CameraImgCenters.Num() == ElementCount);

		OutCliffCam.SetNumUninitialized(SmoothBbxXYS.Num());
		for (int ElementIndex = 0; ElementIndex < ElementCount; ElementIndex++)
		{
			const float CameraImgFocal = CameraImgFocals[ElementIndex];
			const FVector2f CameraImgCenter = CameraImgCenters[ElementIndex];
			OutCliffCam[ElementIndex * 3 + 0] = (SmoothBbxXYS[ElementIndex * 3 + 0] - CameraImgCenter.X) / CameraImgFocal;
			OutCliffCam[ElementIndex * 3 + 1] = (SmoothBbxXYS[ElementIndex * 3 + 1] - CameraImgCenter.Y) / CameraImgFocal;
			OutCliffCam[ElementIndex * 3 + 2] = SmoothBbxXYS[ElementIndex * 3 + 2] / CameraImgFocal;
		}
	}

	TUniquePtr<FHue> FHue::Make(
		UNNEModelData* Step, 
		UNNEModelData* FinalStep,
		TConstArrayView<FRuntimePreference> RuntimePreferences)
	{
		if (!FinalStep)
		{
			return {};
		}
		TSharedPtr<UE::NNE::IModelInstanceRunSync> StepModel = TryCreateInstance(Step, RuntimePreferences, false);
		//Final Step needs to run with ORT CPU in order to produce correct result
		TSharedPtr<UE::NNE::IModelInstanceRunSync> FinalStepModel = CreateCPUModelInstance(*FinalStep, TEXT("NNERuntimeORTCpu"), false);
		if (!StepModel || !FinalStepModel)
		{
			return {};
		}
		return TUniquePtr<FHue>{new FHue(StepModel.ToSharedRef(), FinalStepModel.ToSharedRef())};
	}

	void FHue::AppendRandn(TArray64<float>& OutArray, int Count)
	{
		SCOPED_NAMED_EVENT_TEXT("FHue::AppendRandn", FColor::Yellow);

		for (int Index = 0; Index < Count; Index+=2)
		{
			FVector2d Point;
			double S;
			do
			{
				Point.X = Stream.FRand() * 2 - 1;
				Point.Y = Stream.FRand() * 2 - 1;
				S = Point.SquaredLength();
			} while (S >= 1 || S <= 0);
			Point *= FMath::Sqrt((-2 * FMath::Loge(S)) / S);
			OutArray.Add(Point.X);
			if (Index + 1 < Count)
			{
				OutArray.Add(Point.Y);
			}
		}
	}

	void FHue::EnqueInvalidFrame()
	{
		ValidFrames.Add(false);
	}

	bool FHue::EnqueValidFrame(
		float CameraImgFocal, 
		const FIntVector2& FrameSize, 
		TConstArrayView64<float> InHuman3DTokens,
		const FIntRect& InBox,
		TConstArrayView64<float> InWholeBody,
		float BodyHeight,
		TConstArrayView64<float> InNoise)
	{
		SCOPED_NAMED_EVENT_TEXT("FHue::EnqueFrame", FColor::Orange);

		const int WindowSize = FirstWindow ? FirstWindowSize : MaxWindowSize;
		check(EnquedFrameCount < WindowSize);
#ifndef WINDOW_SIMULATION

		ValidFrames.Add(true);

		CameraImgFocals.Add(CameraImgFocal);

		const FIntPoint ImageSize = Utils::CalculateAdjustedSize(FrameSize.X, FrameSize.Y, MaxImageHeight);
		FVector2f CameraImgCenter = FVector2f(ImageSize) / 2;
		CameraImgCenters.Add(CameraImgCenter);

		for (int Index : Human3DTokensKeepIndices)
		{
			const int Count = Human3dTokensElementCount;
			const int Start = Index * Count;
			Human3DTokens.Append(InHuman3DTokens.Slice(Start, Count));
		}

		FVector3d Box = BoxToXYS(InBox);
		BbxXYS.Add(Box.X);
		BbxXYS.Add(Box.Y);
		BbxXYS.Add(Box.Z);

		CameraIntrinsics.Add(CameraImgFocal);
		CameraIntrinsics.Add(0);
		CameraIntrinsics.Add(CameraImgCenter.X);
		CameraIntrinsics.Add(0);
		CameraIntrinsics.Add(CameraImgFocal);
		CameraIntrinsics.Add(CameraImgCenter.Y);
		CameraIntrinsics.Add(0);
		CameraIntrinsics.Add(0);
		CameraIntrinsics.Add(1);

		check(InWholeBody.Num() == 133 * 3);
		VitposeRaw.Append(InWholeBody.GetData(), VitposeCount);

		if (InNoise.IsEmpty())
		{
			AppendRandn(Noise, NoiseCount);
		}
		else
		{
			check(InNoise.Num() == NoiseCount);
			Noise.Append(InNoise);
		}
#endif
		// Convert BodyHeight from cm to m
		BodyHeights.Add(BodyHeight / 100);
		EnquedFrameCount++;

		return EnquedFrameCount == WindowSize;
	}

	FHue::FRunInfo FHue::Run(
		TArray64<float>& OutSmplxPose,
		TArray64<float>& OutTrans,
		TArray64<float>& OutBetas,
		TArray64<float>& OutStaticConfLogits)
	{
		SCOPED_NAMED_EVENT_TEXT("FHue::Run", FColor::Orange);

		const int FirstMLFrame = FirstWindow ? 0 : SmoothPadding;
		const int MaxMLFrameCount = MaxMLWindowSize - (FirstWindow ? ExtraPadding : 0);
		const int MLFrameCount = FMath::Min(MaxMLFrameCount, EnquedFrameCount - FirstMLFrame);
		const int LastMLFrame = FirstMLFrame + MLFrameCount;

		const bool LastWindow = LastMLFrame == EnquedFrameCount;

		const int FirstValidFrame = FirstWindow ? 0 : TotalPadding;
		const int MaxLastValidFrame = (FirstWindow ? ContextPadding : TotalPadding) + InnerWindow;
		const int LastValidFrame = LastWindow ? LastMLFrame : FMath::Min(LastMLFrame - ContextPadding, MaxLastValidFrame);
		const int ValidFrameCount = LastValidFrame - FirstValidFrame;
		const int LeftShiftCount = LastWindow ? LastValidFrame : LastValidFrame - TotalPadding;
		int OutputFramesCount = 0;
		{
			int CurrentValidFramesCount = 0;
			for (; OutputFramesCount < ValidFrames.Num(); OutputFramesCount++)
			{
				if (ValidFrames[OutputFramesCount])
				{
					if (CurrentValidFramesCount < ValidFrameCount)
					{
						CurrentValidFramesCount++;
					}
					else
					{
						break;
					}
				}
			}
			check(CurrentValidFramesCount == ValidFrameCount);
		}

		if (MLFrameCount > 0)
		{
			const uint32 L = MLFrameCount;
			if (L != SetFrameCount)
			{
				TArray<UE::NNE::FTensorShape> StepInputShapes
				{
					UE::NNE::FTensorShape::Make({}),
					UE::NNE::FTensorShape::Make({1, L, NoiseCount}),
					UE::NNE::FTensorShape::Make({1}),
					UE::NNE::FTensorShape::Make({1, L, 17, 3}),
					UE::NNE::FTensorShape::Make({1, L, CliffCamCount}),
					UE::NNE::FTensorShape::Make({1, L, CamAngvelCount}),
					UE::NNE::FTensorShape::Make({1, L, 5, 1024}),
					UE::NNE::FTensorShape::Make({1, L}),
				};
				TArray<UE::NNE::FTensorShape> FinalStepInputShapes
				{
					UE::NNE::FTensorShape::Make({1, L, NoiseCount}),
					UE::NNE::FTensorShape::Make({1}),
					UE::NNE::FTensorShape::Make({1, L, 17, 3}),
					UE::NNE::FTensorShape::Make({1, L, CliffCamCount}),
					UE::NNE::FTensorShape::Make({1, L, CamAngvelCount}),
					UE::NNE::FTensorShape::Make({1, L, 5, 1024}),
					UE::NNE::FTensorShape::Make({1, L, BbxXYSCount}),
					UE::NNE::FTensorShape::Make({1, L, 3, 3}),
					UE::NNE::FTensorShape::Make({1, L}),
				};
				Step->SetInputTensorShapes(StepInputShapes);
				FinalStep->SetInputTensorShapes(FinalStepInputShapes);
				SetFrameCount = L;
			}
		
			TmpBbxXYS.SetNumUninitialized(EnquedFrameCount * BbxXYSCount);
			SmoothBbxXYS.SetNumUninitialized(EnquedFrameCount * BbxXYSCount);
			VitposeNormalized.SetNumUninitialized(EnquedFrameCount * VitposeCount);
			MedianFilter(BbxXYS, TmpBbxXYS);
			GaussianFilter(TmpBbxXYS, SmoothBbxXYS);
			GetNormalizedVitposes(VitposeRaw, SmoothBbxXYS, VitposeNormalized);
			CalculateCliffCam(CliffCam, SmoothBbxXYS, CameraImgFocals, CameraImgCenters);

			int64 TimeStep = TrainingStepCount -1;
			int64 Length = L;
			PrevSample.SetNumUninitialized(0, EAllowShrinking::No);
			PrevSample.Append(Noise.GetData() + (FirstMLFrame * NoiseCount), L * NoiseCount);
			SmplxPose.SetNumUninitialized(L * SmplxPoseCount);
			Betas.SetNumUninitialized(L * BetasCount);
			Transl.SetNumUninitialized(L * TransCount);
			StaticConfLogits.SetNumUninitialized(L * StaticConfLogitsCount);

			TArray<UE::NNE::FTensorBindingCPU> Inputs
			{
				{&TimeStep, sizeof(TimeStep)},
				ToTensorBindingCPU(PrevSample),
				{&Length, sizeof(Length)},
				ToTensorBindingCPU(VitposeNormalized, FirstMLFrame, L, VitposeCount),
				ToTensorBindingCPU(CliffCam, FirstMLFrame, L, CliffCamCount),
				ToTensorBindingCPU(CamAngvel, FirstMLFrame, L, CamAngvelCount),
				ToTensorBindingCPU(Human3DTokens, FirstMLFrame, L, Human3DTokensCount),
				ToTensorBindingCPU(BodyHeights, FirstMLFrame, L, 1)
			};
			TArray<UE::NNE::FTensorBindingCPU> Outputs{ ToTensorBindingCPU(PrevSample) };
			for (int StepIndex = 0; StepIndex < InferenceStepCount - 1; StepIndex++)
			{
				Step->RunSync(Inputs, Outputs);
				TimeStep -= StepRatio;
			}
			check(TimeStep == StepRatio - 1);
			Inputs = {
				ToTensorBindingCPU(PrevSample),
				{&Length, sizeof(Length)},
				ToTensorBindingCPU(VitposeNormalized, FirstMLFrame, L, VitposeCount),
				ToTensorBindingCPU(CliffCam, FirstMLFrame, L, CliffCamCount),
				ToTensorBindingCPU(CamAngvel, FirstMLFrame, L, CamAngvelCount),
				ToTensorBindingCPU(Human3DTokens, FirstMLFrame, L, Human3DTokensCount),
				ToTensorBindingCPU(SmoothBbxXYS, FirstMLFrame, L, BbxXYSCount),
				ToTensorBindingCPU(CameraIntrinsics, FirstMLFrame, L, CameraIntrinsicsCount),
				ToTensorBindingCPU(BodyHeights, FirstMLFrame, L, 1),
			};
			Outputs = {
				ToTensorBindingCPU(SmplxPose),
				ToTensorBindingCPU(Betas),
				ToTensorBindingCPU(Transl),
				ToTensorBindingCPU(StaticConfLogits),
			};
			FinalStep->RunSync(Inputs, Outputs);

			auto AppendOutput = [this, FirstValidFrame, ValidFrameCount, FirstMLFrame, OutputFramesCount]<class T>(TArray64<T>&OutArray, const TArray64<T>&InArray, int ElementSize)
			{
				TConstArrayView<T> Src = MakeConstArrayView64(InArray).Slice((FirstValidFrame - FirstMLFrame) * ElementSize, ValidFrameCount * ElementSize);
				int SrcIndex = 0;
				for (int FrameIndex = 0; FrameIndex < OutputFramesCount; FrameIndex++)
				{
					if (ValidFrames[FrameIndex])
					{
						OutArray.Append(Src.Slice(SrcIndex * ElementSize, ElementSize));
						SrcIndex++;
					}
					else
					{
						OutArray.AddZeroed(ElementSize);
					}
				}
				check(SrcIndex == ValidFrameCount);
			};
			AppendOutput(OutSmplxPose, SmplxPose, SmplxPoseCount);
			AppendOutput(OutBetas, Betas, BetasCount);
			AppendOutput(OutTrans, Transl, TransCount);
			AppendOutput(OutStaticConfLogits, StaticConfLogits, StaticConfLogitsCount);

			Utils::ShiftDataLeft(CameraImgFocals, LeftShiftCount);
			Utils::ShiftDataLeft(CameraImgCenters, LeftShiftCount);
			Utils::ShiftDataLeft(Human3DTokens, Human3DTokensCount* LeftShiftCount);
			Utils::ShiftDataLeft(BbxXYS, BbxXYSCount* LeftShiftCount);
			Utils::ShiftDataLeft(VitposeRaw, VitposeCount* LeftShiftCount);
			Utils::ShiftDataLeft(CameraIntrinsics, CameraIntrinsicsCount* LeftShiftCount);
			Utils::ShiftDataLeft(Noise, NoiseCount* LeftShiftCount);
			Utils::ShiftDataLeft(BodyHeights, LeftShiftCount);
		}
		Utils::ShiftDataLeft(ValidFrames, OutputFramesCount);
		FRunInfo Info =
		{
			.RelFirstMLFrame = FirstMLFrame,
			.RelLastMLFrame = LastMLFrame,
			.AbsFirstMLFrame = FirstMLFrame + MLFrameOffset,
			.AbsLastMLFrame = LastMLFrame + MLFrameOffset,
			.FirstValidFrame = FirstValidFrame - (FirstWindow ? 0 : SmoothPadding),
			.LastValidFrame = LastValidFrame - (FirstWindow ? 0 : SmoothPadding),
			.LastWindow = LastWindow,
			.OutputFrameCount = OutputFramesCount,
			.LeftShiftCount = LeftShiftCount
		};

		EnquedFrameCount -= LeftShiftCount;
		FirstWindow = LastWindow;
		MLFrameOffset += LastWindow ? -MLFrameOffset : LeftShiftCount;
		return Info;
	}

	TUniquePtr<FOfflineBodyTracker> FOfflineBodyTracker::Make(
		UNNEModelData* ViTPose,
		UNNEModelData* ViTPosePost,
		UNNEModelData* CHMRBackbone,
		UNNEModelData* CHMRHead,
		UNNEModelData* HueStep,
		UNNEModelData* HueFinalStep
	)
	{
		static const TArray<FRuntimePreference> RuntimePreferencesArray =
		{
			//{ TEXT("NNERuntimeTRT"), EDeviceType::GPU },
			{ TEXT("NNERuntimeORTDml"), EDeviceType::GPU },
			{ TEXT("NNERuntimeORTCpu"), EDeviceType::CPU }
		};
		static const TConstArrayView64<FRuntimePreference> RuntimePreferences = RuntimePreferencesArray;
		static const TConstArrayView64<FRuntimePreference> ORTRuntimePreferences = RuntimePreferences.Right(2);

		TUniquePtr<FEstimateKeypoints> EstimateKeypoints = FEstimateKeypoints::Make(ViTPose, ViTPosePost, ORTRuntimePreferences);
		TUniquePtr<FCHMR> CHMR = FCHMR::Make(CHMRBackbone, CHMRHead, RuntimePreferences);
		TUniquePtr<FHue> Hue = FHue::Make(HueStep, HueFinalStep, ORTRuntimePreferences);

		static const TArray<UE::NNE::Segmentation::FRuntimePreference> SegmentationRuntimePreferences =
		{
			//{ TEXT("NNERuntimeTRT"), UE::NNE::Segmentation::EDeviceType::GPU },
			{ TEXT("NNERuntimeORTDml"), UE::NNE::Segmentation::EDeviceType::GPU },
			{ TEXT("NNERuntimeORTCpu"), UE::NNE::Segmentation::EDeviceType::CPU }
		};
		TUniquePtr<UE::NNE::Segmentation::IDetector> PersonDetector = UE::NNE::Segmentation::Detectron2::Make(SegmentationRuntimePreferences);
		TUniquePtr<UE::NNE::Segmentation::ITracker> PersonTracker = UE::NNE::Segmentation::SAM2::Make(SegmentationRuntimePreferences);

		if (!EstimateKeypoints.IsValid() || !CHMR.IsValid() || !Hue.IsValid() || !PersonDetector.IsValid() || !PersonTracker.IsValid())
		{
			return nullptr;
		}

		return TUniquePtr<FOfflineBodyTracker>(new FOfflineBodyTracker(MoveTemp(PersonDetector), MoveTemp(PersonTracker), MoveTemp(EstimateKeypoints), MoveTemp(CHMR), MoveTemp(Hue)));
	}

	bool FOfflineBodyTracker::EnqueFrame(const FUEImageDataType& Image)
	{
		SCOPED_NAMED_EVENT_TEXT("OfflineBodyTracker::Run", FColor::Magenta);

		if (FrameIndex == 0)
		{
			ImageScale = FMath::Min((float)MaxImageHeight / Image.Height, 1);
		}
		FrameIndex++;

		const float DetectionScoreThreshold = 0.7f; // TODO: Promote detection score threshold to a variable
		const float TrackingScoreThreshold = 0.5f; // TODO: Promote detection score threshold to a variable
		FImage Frame(Image.Width,Image.Height,ERawImageFormat::BGRA8);
		Frame.RawData = Image.Data;
		TArray<UE::NNE::Segmentation::FTracking> Tracking;
		bool Detected = true;
		if (!bTrackerInitialized) 
		{
			TArray<UE::NNE::Segmentation::FDetection> Detections = PersonDetector->Detect(Frame);
			if (Detections.Num() < 1 || Detections[0].Score < DetectionScoreThreshold) // TODO: Adjust for multi person tracking
			{
				Detected = false;
			}
			else
			{
				Tracking = PersonTracker->Initialize(Frame, TConstArrayView<UE::NNE::Segmentation::FDetection>(Detections.GetData(), 1));
				bTrackerInitialized = true;
			}
		}
		else
		{
			Tracking = PersonTracker->Track(Frame);
		}
		if (Detected && (Tracking.Num() < 1 || Tracking[0].Score < TrackingScoreThreshold)) // TODO: Adjust for multi person tracking
		{
			Detected = false;
		}
		if (Detected)
		{
			FIntRect BBox = Tracking[0].Box.Scale(ImageScale);
			BBox.InflateRect(10);

			TArray64<float> Keypoints;
			EstimateKeypoints->Run(Image, BBox, Keypoints);
			TArray64<float> Human3DTokens;
			CHMR->Run(Image, BBox, Keypoints, CameraImgFocal, Human3DTokens);
			bool bWindowFull = Hue->EnqueValidFrame(CameraImgFocal, FIntVector2(Image.Width, Image.Height),
											   Human3DTokens, BBox, Keypoints);

			ValidFrames.Add(true);
			BBoxes.Add(BBox.Min.X);
			BBoxes.Add(BBox.Min.Y);
			BBoxes.Add(BBox.Max.X);
			BBoxes.Add(BBox.Max.Y);
			FEstimateKeypoints::Op25FromKeypoints(Keypoints, Keypoints2D);
			return bWindowFull;
		}
		else
		{
			ValidFrames.Add(false);
			BBoxes.AddZeroed(BBoxesCount);
			Keypoints2D.AddZeroed(Keypoints2DCount);
			Hue->EnqueInvalidFrame();
			return false;
		}
	}

	bool FOfflineBodyTracker::GetResults(
		TArray64<bool>& OutValidFrames,
		TArray64<float>& OutBBox,
		TArray64<float>& OutKeypoints2D,
		TArray64<float>& OutSmplxPose,
		TArray64<float>& OutSmplxTrans,
		TArray64<float>& OutSmplxBetas,
		TArray64<float>& OutStaticConfLogits
	)
	{
		FHue::FRunInfo Info = Hue->Run(OutSmplxPose, OutSmplxTrans, OutSmplxBetas, OutStaticConfLogits);
		OutValidFrames.Append(ValidFrames.GetData(), Info.OutputFrameCount);
		OutBBox.Append(BBoxes.GetData(), Info.OutputFrameCount * BBoxesCount);
		OutKeypoints2D.Append(Keypoints2D.GetData(), Info.OutputFrameCount * Keypoints2DCount);
		Utils::ShiftDataLeft(ValidFrames, Info.OutputFrameCount);
		Utils::ShiftDataLeft(BBoxes, Info.OutputFrameCount * BBoxesCount);
		Utils::ShiftDataLeft(Keypoints2D, Info.OutputFrameCount * Keypoints2DCount);
		return Info.LastWindow;
	}
} // namespace UE::BodyTracker
