// Copyright Epic Games, Inc. All Rights Reserved.

#include "CameraCalibration.h"

#include "BodyTrackerModule.h"
#include "BodyTrackerUtils.h"
#include "NNEModelData.h"
#include "NNEUtils.h"
#include "NNETypes.h"
#include "Serialization/Archive.h"
#include "String/ParseTokens.h"

using namespace UE::NNEUtils::Private;

namespace UE::BodyTracker
{
	TUniquePtr<FCameraCalibration> FCameraCalibration::Make(UNNEModelData* Model, TConstArrayView<FRuntimePreference> RuntimePreferences)
	{
		if (!Model)
		{
			return {};
		}

		TSharedPtr<UE::NNE::IModelInstanceRunSync> ModelInstance = TryCreateInstance(Model, RuntimePreferences, false);
		if (!ModelInstance)
		{
			return {};
		}
		return TUniquePtr<FCameraCalibration>(new FCameraCalibration(ModelInstance.ToSharedRef()));
	}

	void GaussianFilter2D(TArrayView64<float> InOutValues, 
						  const int Width, const int Height, 
						  TConstArrayView64<double> Kernel,
						  TArrayView64<double> TmpValues)
	{
		SCOPED_NAMED_EVENT_TEXT("GaussianFilter2D", FColor::Magenta);

		check(InOutValues.Num() == TmpValues.Num());
		check(InOutValues.Num() == Width * Height * 4);
		const int KernelSize = Kernel.Num();
		check(KernelSize % 2);
		const int Padding = (KernelSize - 1) / 2;

#if 1
		ParallelFor(Height, [&](int Y)
			{
				for (int BaseX = 0; BaseX < Width; BaseX++)
				{
					FVector4d Sum{};
					for (int KernelIndex = 0; KernelIndex < KernelSize; KernelIndex++)
					{
						int X = BaseX + (KernelIndex - Padding);
						X = X < 0 ? -(X + 1) : X;
						X = X >= Width ? 2 * Width - 1 - X : X;
						int Index = (Y * Width + X) * 4;
						FVector4d Pixel(InOutValues[Index + 0], InOutValues[Index + 1], InOutValues[Index + 2], InOutValues[Index + 3]);
						Sum += Pixel * Kernel[KernelIndex];
					}
					int OutIndex = (Y * Width + BaseX) * 4;
					TmpValues[OutIndex + 0] = Sum.X;
					TmpValues[OutIndex + 1] = Sum.Y;
					TmpValues[OutIndex + 2] = Sum.Z;
					TmpValues[OutIndex + 3] = Sum.W;
				}
			}, EParallelForFlags::None);

		ParallelFor(Width, [&](int X)
			{
				for (int BaseY = 0; BaseY < Height; BaseY++)
				{
					FVector4d Sum{};
					for (int KernelIndex = 0; KernelIndex < KernelSize; KernelIndex++)
					{
						int Y = BaseY + (KernelIndex - Padding);
						Y = Y < 0 ? -(Y + 1) : Y;
						Y = Y >= Height ? 2 * Height - 1 - Y : Y;
						int Index = (Y * Width + X) * 4;
						FVector4d Pixel(TmpValues[Index + 0], TmpValues[Index + 1], TmpValues[Index + 2], TmpValues[Index + 3]);
						Sum += Pixel * Kernel[KernelIndex];
					}
					int OutIndex = (BaseY * Width + X) * 4;
					InOutValues[OutIndex + 0] = Sum.X;
					InOutValues[OutIndex + 1] = Sum.Y;
					InOutValues[OutIndex + 2] = Sum.Z;
					InOutValues[OutIndex + 3] = Sum.W;
				}
			}, EParallelForFlags::None);
#else
		for (int Y = 0; Y < Height; Y++)
		{
			for (int BaseX = 0; BaseX < Width; BaseX++)
			{
				FVector4d Sum{};
				for (int KernelIndex = 0; KernelIndex < KernelSize; KernelIndex++)
				{
					int X = BaseX + (KernelIndex - Padding);
					X = X < 0 ? -(X + 1) : X;
					X = X >= Width ? 2 * Width - 1 - X : X;
					int Index = (Y * Width + X) * 4;
					FVector4d Pixel(InOutValues[Index + 0], InOutValues[Index + 1], InOutValues[Index + 2], InOutValues[Index + 3]);
					Sum += Pixel * Kernel[KernelIndex];
				}
				int OutIndex = (Y * Width + BaseX) * 4;
				TmpValues[OutIndex + 0] = Sum.X;
				TmpValues[OutIndex + 1] = Sum.Y;
				TmpValues[OutIndex + 2] = Sum.Z;
				TmpValues[OutIndex + 3] = Sum.W;
			}
		}

		for (int X = 0; X < Width; X++)
		{
			for (int BaseY = 0; BaseY < Height; BaseY++)
			{
				FVector4d Sum{};
				for (int KernelIndex = 0; KernelIndex < KernelSize; KernelIndex++)
				{
					int Y = BaseY + (KernelIndex - Padding);
					Y = Y < 0 ? -(Y + 1) : Y;
					Y = Y >= Height ? 2 * Height - 1 - Y : Y;
					int Index = (Y * Width + X) * 4;
					FVector4d Pixel(TmpValues[Index + 0], TmpValues[Index + 1], TmpValues[Index + 2], TmpValues[Index + 3]);
					Sum += Pixel * Kernel[KernelIndex];
				}
				int OutIndex = (BaseY * Width + X) * 4;
				InOutValues[OutIndex + 0] = Sum.X;
				InOutValues[OutIndex + 1] = Sum.Y;
				InOutValues[OutIndex + 2] = Sum.Z;
				InOutValues[OutIndex + 3] = Sum.W;
			}
		}
#endif
	}

	float GetMean(TArray<float>& Values)
	{
		Values.Sort();
		if (Values.Num() % 2)
		{
			return Values[Values.Num() / 2];
		}
		else
		{
			int Index = (Values.Num() - 1) / 2;
			return (Values[Index] + Values[Index + 1]) / 2;
		}
	}

	void FCameraCalibration::Run(const UE::MetaHuman::Pipeline::FUEImageDataType& Frame)
	{
		SCOPED_NAMED_EVENT_TEXT("FCameraCalibration::Run", FColor::Magenta);

		const int ShortEdge = 320;
		FIntPoint NewSize(ShortEdge, ShortEdge);
		const FIntPoint OriginalSize = Utils::CalculateAdjustedSize(Frame.Width, Frame.Height, 896);
		float Scale;
		if (OriginalSize.X < OriginalSize.Y)
		{
			NewSize.Y = ShortEdge * OriginalSize.Y / OriginalSize.X;
			NewSize.Y = FMath::Floor(NewSize.Y / 32) * 32;
			Scale = (float)OriginalSize.X / ShortEdge;
		}
		else
		{
			NewSize.X = ShortEdge * OriginalSize.X / OriginalSize.Y;
			NewSize.X = FMath::Floor(NewSize.X / 32) * 32;
			Scale = (float)OriginalSize.Y / ShortEdge;
		}
		const FIntPoint CropSize = (FVector2f(NewSize) * Scale).IntPoint();
		const FIntPoint CropMin = (OriginalSize - CropSize) / 2;
		const FIntRect Box(CropMin, OriginalSize - CropMin);
		const Utils::FPreprocessImageArgs PreprocessArgs =
		{
			.MaxHeight = 896,
			.Crop = Box,
			.SampleOffset = {1, 1},
			.NewSize = NewSize,
			.Mean = {0, 0, 0},
			.StdInv = {1, 1, 1}
		};
		TmpImage.SetNumUninitialized(Frame.Data.Num());
		SmoothedImage.SetNumUninitialized(Frame.Data.Num());
#if 1
		static const int Parallelization = 32;
		ParallelFor(Parallelization, [&](int ThreadIndex)
			{
				const int WorkSize = FMath::DivideAndRoundDown(Frame.Data.Num(), Parallelization);
				const int StartIndex = ThreadIndex * WorkSize;
				const int OnePastEndIndex = FMath::Min(StartIndex + WorkSize, Frame.Data.Num());
				for (int Index = StartIndex; Index < OnePastEndIndex; Index++)
				{
					constexpr float inv255 = 1.f / 255.f;
					SmoothedImage[Index] = Frame.Data[Index] * inv255;
				}
			}, EParallelForFlags::None);
#else
		for (int Index = 0; Index < Frame.Data.Num(); Index++)
		{
			constexpr float inv255 = 1.f / 255.f;
			SmoothedImage[Index] = Frame.Data[Index] * inv255;
		}
#endif
		FVector2d Factors((double)Frame.Width / NewSize.X, (double)Frame.Height / NewSize.Y);
		double Sigma = FVector2d::Max((Factors - 1) / 2, FVector2d(0.001)).GetMax();
		int KernelSize = FMath::Max(4 * Sigma, 3);
		KernelSize = KernelSize % 2 == 0 ? KernelSize + 1 : KernelSize;
		TArray<double> GaussianKernel = Utils::CreateGaussianFilter(Sigma, KernelSize);
		GaussianFilter2D(SmoothedImage, Frame.Width, Frame.Height, GaussianKernel, TmpImage);
		Utils::FPreprocessImageInfo PreprocessInfo = Utils::PreprocessImage<float, float>(SmoothedImage, Frame.Width, Frame.Height, PreprocessArgs, PreprocessedImage);
		if (!bShapeSet)
		{
			TArray<UE::NNE::FTensorShape> InputShapes
			{
				UE::NNE::FTensorShape::Make({1, 3, (uint32)NewSize.Y, (uint32)NewSize.X})
			};
			int PixelCount = NewSize.X * NewSize.Y;
			UpField.Init(0, 2 * PixelCount);
			UpConfidence.Init(0, PixelCount);
			LatitudeField.Init(0, PixelCount);
			LatitudeConfidence.Init(0, PixelCount);
			GeoCalib->SetInputTensorShapes(InputShapes);
			bShapeSet = true;
		}
		TArray<UE::NNE::FTensorBindingCPU> Inputs{ ToTensorBindingCPU(PreprocessedImage) };
		TArray<UE::NNE::FTensorBindingCPU> Outputs;
		if (CalculateFields)
		{
			Outputs = {
				ToTensorBindingCPU(UpField),
				ToTensorBindingCPU(UpConfidence),
				ToTensorBindingCPU(LatitudeField),
				ToTensorBindingCPU(LatitudeConfidence)
			};
		}
		else
		{
			Outputs = { ToTensorBindingCPU(Output) };
		}
		GeoCalib->RunSync(Inputs, Outputs);
		if (!CalculateFields)
		{
			Fovs.Add(Output[0] * Scale);
			Pitches.Add(Output[1]);
			Rolls.Add(Output[2]);

			Fov = GetMean(Fovs);
			Pitch = GetMean(Pitches);
			Roll = GetMean(Rolls);
		}
	}
}