// Copyright Epic Games, Inc. All Rights Reserved.

#include "BodyTrackerUtils.h"


namespace UE::BodyTracker::Utils
{
	using UE::MetaHuman::Pipeline::FUEImageDataType;

	FIntPoint CalculateAdjustedSize(int Width, int Height, int MaxHeight)
	{
		FIntPoint AdjustedSize(Width, Height);
		if (MaxHeight > 0 && Height > MaxHeight)
		{
			AdjustedSize.X = Width * MaxHeight / Height;
			AdjustedSize.Y = MaxHeight;
		}
		return AdjustedSize;
	}

	TArray<double> CreateGaussianFilter(double StdDev, int KernelSize)
	{
		check(KernelSize % 2);
		TArray<double> Result;
		Result.SetNumUninitialized(KernelSize);
		const int KernelHalfSize = (KernelSize - 1) / 2;
		double Sum = 0;
		for (int Index = 0; Index < KernelSize; Index++)
		{
			double Element = Index - KernelHalfSize;
			Element = FMath::Exp(-(Element * Element) / (2 * StdDev * StdDev));
			Sum += Element;
			Result[Index] = Element;
		}
		for (double& Element : Result)
		{
			Element /= Sum;
		}
		return Result;
	}

} // UE::BodyTracker::Utils
