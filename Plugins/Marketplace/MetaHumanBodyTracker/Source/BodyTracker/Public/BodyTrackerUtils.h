// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "BodyTrackerModule.h"
#include "Pipeline/DataTreeTypes.h"
#include "Async/ParallelFor.h"

namespace UE::BodyTracker::Utils
{
	using UE::MetaHuman::Pipeline::FUEImageDataType;
	

	struct FPreprocessImageArgs
	{
		int MaxHeight;
		FIntRect Crop;
		FVector2f SampleOffset;
		FIntPoint NewSize;
		FLinearColor Mean;
		FLinearColor StdInv;
	};

	struct FPreprocessImageInfo
	{
		FIntRect Crop;
		float Scale;
		FVector2f Offset;
	};

	FORCEINLINE
	FLinearColor LoadPixel(TConstArrayView64<uint8> InPixels, int Height, int Width, int X, int Y)
	{
		X = FMath::Clamp(X, 0, Width - 1);
		Y = FMath::Clamp(Y, 0, Height - 1);
		int Index = (Y * Width + X) * 4;
		const FColor Color(InPixels[Index + 2], InPixels[Index + 1], InPixels[Index + 0], InPixels[Index + 3]);
		return Color.ReinterpretAsLinear();
	}

	FORCEINLINE
	FLinearColor LoadPixel(TConstArrayView64<float> InPixels, int Height, int Width, int X, int Y)
	{
		X = FMath::Clamp(X, 0, Width - 1);
		Y = FMath::Clamp(Y, 0, Height - 1);
		int Index = (Y * Width + X) * 4;
		const FLinearColor Color(InPixels[Index + 2], InPixels[Index + 1], InPixels[Index + 0], InPixels[Index + 3]);
		return Color;
	}

	BODYTRACKER_API FIntPoint CalculateAdjustedSize(int Width, int Height, int MaxHeight);
	BODYTRACKER_API TArray<double> CreateGaussianFilter(double StdDev, int KernelSize);

	template<class TIn, class TOut>
	FPreprocessImageInfo PreprocessImage(TConstArrayView64<TIn> InPixels, const int Width, const int Height, const FPreprocessImageArgs& Args, TArray64<TOut>& OutImage)
	{
		SCOPED_NAMED_EVENT_TEXT("PreprocessImage", FColor::Magenta);
		const FIntPoint AdjustedSize = CalculateAdjustedSize(Width, Height, Args.MaxHeight);
		const FVector2f AdjustedScale((float)Width / AdjustedSize.X, (float)Height / AdjustedSize.Y);

		const FIntRect OriginalRect(0, 0, AdjustedSize.X, AdjustedSize.Y);
		FIntRect Crop(Args.Crop);
		if (Crop.IsEmpty())
		{
			Crop = OriginalRect;
		}
		Crop.Clip(OriginalRect);
		ensure(!Crop.IsEmpty());

		const FVector2f NewSize(Args.NewSize);
		const FVector2f CropSize(Crop.Size());
		const FVector2f CropOffset(Crop.Min);
		const float Scale = (CropSize / NewSize).GetMax();
		const FVector2f PadOffset = (NewSize - CropSize / Scale) / 2;
		const FIntRect ColorRegion(PadOffset.IntPoint(), Args.NewSize - PadOffset.IntPoint());
		const FVector2f Offset = CropOffset - PadOffset * Scale;
		const FVector2f SampleOffset = Offset * AdjustedScale + Args.SampleOffset;
		const FVector2f SampleScale = AdjustedScale * Scale;

		const int NewPixelCount = Args.NewSize.X * Args.NewSize.Y;
		OutImage.SetNumUninitialized(NewPixelCount * 3);
		

#if 1
		ParallelFor(Args.NewSize.Y, [&](int Y)
			{
				TOut* OutColors = OutImage.GetData() + Y * Args.NewSize.X;
				FInt32Point DstPixel(0, Y);
				for (DstPixel.X = 0; DstPixel.X < Args.NewSize.X; DstPixel.X++)
				{
					FLinearColor Color{};
					if (ColorRegion.Contains(DstPixel))
					{
						const FVector2f SrcPixel = FVector2f(DstPixel) * SampleScale + SampleOffset;
						const FIntVector2 SrcPixelI(SrcPixel.X, SrcPixel.Y);
						const FLinearColor Color0 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 0, SrcPixelI.Y + 0);
						const FLinearColor Color1 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 1, SrcPixelI.Y + 0);
						const FLinearColor Color2 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 0, SrcPixelI.Y + 1);
						const FLinearColor Color3 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 1, SrcPixelI.Y + 1);
						Color = FMath::BiLerp(Color0, Color1, Color2, Color3, SrcPixel.X - SrcPixelI.X, SrcPixel.Y - SrcPixelI.Y);
					}
					const FLinearColor NormalizeColor = (Color - Args.Mean) * Args.StdInv;
					OutColors[NewPixelCount * 0] = NormalizeColor.R;
					OutColors[NewPixelCount * 1] = NormalizeColor.G;
					OutColors[NewPixelCount * 2] = NormalizeColor.B;
					OutColors++;
				}
			}, EParallelForFlags::None);
#else
		TOut* OutColors = OutImage.GetData();
		FInt32Point DstPixel{};
		for (DstPixel.Y = 0; DstPixel.Y < Args.NewSize.Y; DstPixel.Y++)
		{
			for (DstPixel.X = 0; DstPixel.X < Args.NewSize.X; DstPixel.X++)
			{
				FLinearColor Color{};
				if (ColorRegion.Contains(DstPixel))
				{
					const FVector2f SrcPixel = FVector2f(DstPixel) * SampleScale + SampleOffset;
					const FIntVector2 SrcPixelI(SrcPixel.X, SrcPixel.Y);
					const FLinearColor Color0 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 0, SrcPixelI.Y + 0);
					const FLinearColor Color1 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 1, SrcPixelI.Y + 0);
					const FLinearColor Color2 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 0, SrcPixelI.Y + 1);
					const FLinearColor Color3 = LoadPixel(InPixels, Height, Width, SrcPixelI.X + 1, SrcPixelI.Y + 1);
					Color = FMath::BiLerp(Color0, Color1, Color2, Color3, SrcPixel.X - SrcPixelI.X, SrcPixel.Y - SrcPixelI.Y);
				}
				const FLinearColor NormalizeColor = (Color - Args.Mean) * Args.StdInv;
				OutColors[NewPixelCount * 0] = NormalizeColor.R;
				OutColors[NewPixelCount * 1] = NormalizeColor.G;
				OutColors[NewPixelCount * 2] = NormalizeColor.B;
				OutColors++;
			}
		}
#endif
		FPreprocessImageInfo Info =
		{
			.Crop = Crop,
			.Scale = Scale,
			.Offset = Offset
		};
		return Info;
	}

	template<class T>
	FPreprocessImageInfo PreprocessImage(const FUEImageDataType& InImage, const FPreprocessImageArgs& Args, TArray64<T>& OutImage)
	{
		return PreprocessImage<uint8, T>(InImage.Data, InImage.Width, InImage.Height, Args, OutImage);
	}

	template<class T>
	void ShiftDataLeft(TArray64<T>& Array, int Count)
	{
		check(Array.Num() >= Count);
		FMemory::Memmove(Array.GetData(), Array.GetData() + Count, (Array.Num() - Count) * sizeof(typename TArray64<T>::ElementType));
		Array.SetNumUninitialized(Array.Num() - Count, EAllowShrinking::No);
	};

} // UE::BodyTracker::Utils