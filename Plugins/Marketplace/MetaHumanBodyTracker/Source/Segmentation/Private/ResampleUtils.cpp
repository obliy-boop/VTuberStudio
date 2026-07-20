// Copyright Epic Games, Inc. All Rights Reserved.

#include "ResampleUtils.h"

// ---------------------------------------------------------------------------
// Kernel: bilinear (tent / hat), support = 1.0
// ---------------------------------------------------------------------------

FResampleCoeffs PrecomputeBilinearCoeffs(int32 InSize, int32 OutSize)
{
	const double Scale       = static_cast<double>(InSize) / OutSize;
	const double FilterScale = FMath::Max(Scale, 1.0);
	const double Support     = FilterScale;  // bilinear support = 1.0 × filterscale
	const int32  KSize       = FMath::CeilToInt32(Support) * 2 + 1;

	FResampleCoeffs Coeffs;
	Coeffs.KSize = KSize;
	Coeffs.Bounds.SetNumZeroed(OutSize * 2);
	Coeffs.Weights.SetNumZeroed(OutSize * KSize);

	for (int32 OutIdx = 0; OutIdx < OutSize; ++OutIdx)
	{
		const double Center         = (OutIdx + 0.5) * Scale;
		const double InvFilterScale = 1.0 / FilterScale;

		int32 XMin = static_cast<int32>(Center - Support + 0.5);
		if (XMin < 0) XMin = 0;
		int32 XMax = static_cast<int32>(Center + Support + 0.5);
		if (XMax > InSize) XMax = InSize;
		const int32 Count = XMax - XMin;

		TArray<double> DWeights;
		DWeights.SetNumUninitialized(Count);
		double WSum = 0.0;
		for (int32 X = 0; X < Count; ++X)
		{
			const double W = FMath::Max(0.0, 1.0 - FMath::Abs((X + XMin + 0.5 - Center) * InvFilterScale));
			DWeights[X] = W;
			WSum += W;
		}

		int32* WPtr = Coeffs.Weights.GetData() + OutIdx * KSize;
		if (WSum != 0.0)
		{
			for (int32 X = 0; X < Count; ++X)
			{
				WPtr[X] = static_cast<int32>(DWeights[X] / WSum * (1 << PilPrecisionBits) + 0.5);
			}
		}

		Coeffs.Bounds[OutIdx * 2 + 0] = XMin;
		Coeffs.Bounds[OutIdx * 2 + 1] = Count;
	}

	return Coeffs;
}

// ---------------------------------------------------------------------------
// Kernel: PIL BICUBIC (Keys / Catmull-Rom), a = -0.5, support = 2.0
// ---------------------------------------------------------------------------

static double PilBicubicKernel(double X)
{
	// a = -0.5  →  (a+2)|x|³ - (a+3)|x|² + 1  for |x| < 1
	//              a|x|³ - 5a|x|² + 8a|x| - 4a for 1 ≤ |x| < 2
	if (X < 0.0) X = -X;
	if (X < 1.0) return (1.5 * X - 2.5) * X * X + 1.0;
	if (X < 2.0) return ((-0.5 * X + 2.5) * X - 4.0) * X + 2.0;
	return 0.0;
}

FResampleCoeffs PrecomputeBicubicCoeffs(int32 InSize, int32 OutSize)
{
	const double Scale       = static_cast<double>(InSize) / OutSize;
	const double FilterScale = FMath::Max(Scale, 1.0);
	const double Support     = 2.0 * FilterScale;  // bicubic support = 2.0 × filterscale
	const int32  KSize       = FMath::CeilToInt32(Support) * 2 + 1;

	FResampleCoeffs Coeffs;
	Coeffs.KSize = KSize;
	Coeffs.Bounds.SetNumZeroed(OutSize * 2);
	Coeffs.Weights.SetNumZeroed(OutSize * KSize);

	for (int32 OutIdx = 0; OutIdx < OutSize; ++OutIdx)
	{
		const double Center         = (OutIdx + 0.5) * Scale;
		const double InvFilterScale = 1.0 / FilterScale;

		int32 XMin = static_cast<int32>(Center - Support + 0.5);
		if (XMin < 0) XMin = 0;
		int32 XMax = static_cast<int32>(Center + Support + 0.5);
		if (XMax > InSize) XMax = InSize;
		const int32 Count = XMax - XMin;

		TArray<double> DWeights;
		DWeights.SetNumUninitialized(Count);
		double WSum = 0.0;
		for (int32 X = 0; X < Count; ++X)
		{
			const double W = PilBicubicKernel((X + XMin + 0.5 - Center) * InvFilterScale);
			DWeights[X] = W;
			WSum += W;
		}

		int32* WPtr = Coeffs.Weights.GetData() + OutIdx * KSize;
		if (WSum != 0.0)
		{
			for (int32 X = 0; X < Count; ++X)
			{
				// Bicubic weights can be negative — do NOT clamp to positive here.
				WPtr[X] = static_cast<int32>(DWeights[X] / WSum * (1 << PilPrecisionBits) + 0.5);
			}
		}

		Coeffs.Bounds[OutIdx * 2 + 0] = XMin;
		Coeffs.Bounds[OutIdx * 2 + 1] = Count;
	}

	return Coeffs;
}

// ---------------------------------------------------------------------------
// Shared two-pass separable resample (BGRA8 → BGRA8)
// ---------------------------------------------------------------------------

static uint8 PilClipToUint8(int32 Value)
{
	Value >>= PilPrecisionBits;
	return static_cast<uint8>(FMath::Clamp(Value, 0, 255));
}

void ResizePIL(
	const uint8* SrcPixels, int32 SrcW, int32 SrcH,
	uint8*       DstPixels, int32 DstW, int32 DstH,
	const FResampleCoeffs& HCoeffs,
	const FResampleCoeffs& VCoeffs)
{
	// Pass 1: Horizontal — (SrcW, SrcH) → (DstW, SrcH)
	TArray<uint8> Intermediate;
	Intermediate.SetNumUninitialized(DstW * SrcH * 4);

	for (int32 Y = 0; Y < SrcH; ++Y)
	{
		const uint8* SrcRow = SrcPixels + Y * SrcW * 4;
		uint8*       DstRow = Intermediate.GetData() + Y * DstW * 4;

		for (int32 X = 0; X < DstW; ++X)
		{
			const int32  XMin  = HCoeffs.Bounds[X * 2 + 0];
			const int32  Count = HCoeffs.Bounds[X * 2 + 1];
			const int32* K     = HCoeffs.Weights.GetData() + X * HCoeffs.KSize;

			const int32 RoundBias = 1 << (PilPrecisionBits - 1);
			int32 AccB = RoundBias, AccG = RoundBias, AccR = RoundBias, AccA = RoundBias;

			for (int32 I = 0; I < Count; ++I)
			{
				const uint8* Px = SrcRow + (XMin + I) * 4;
				AccB += Px[0] * K[I];
				AccG += Px[1] * K[I];
				AccR += Px[2] * K[I];
				AccA += Px[3] * K[I];
			}

			uint8* Out = DstRow + X * 4;
			Out[0] = PilClipToUint8(AccB);
			Out[1] = PilClipToUint8(AccG);
			Out[2] = PilClipToUint8(AccR);
			Out[3] = PilClipToUint8(AccA);
		}
	}

	// Pass 2: Vertical — (DstW, SrcH) → (DstW, DstH)
	const int32 RowStride = DstW * 4;

	for (int32 Y = 0; Y < DstH; ++Y)
	{
		const int32  YMin  = VCoeffs.Bounds[Y * 2 + 0];
		const int32  Count = VCoeffs.Bounds[Y * 2 + 1];
		const int32* K     = VCoeffs.Weights.GetData() + Y * VCoeffs.KSize;

		uint8* DstRow = DstPixels + Y * RowStride;

		for (int32 X = 0; X < DstW; ++X)
		{
			const int32 RoundBias = 1 << (PilPrecisionBits - 1);
			int32 AccB = RoundBias, AccG = RoundBias, AccR = RoundBias, AccA = RoundBias;

			for (int32 I = 0; I < Count; ++I)
			{
				const uint8* Px = Intermediate.GetData() + (YMin + I) * RowStride + X * 4;
				AccB += Px[0] * K[I];
				AccG += Px[1] * K[I];
				AccR += Px[2] * K[I];
				AccA += Px[3] * K[I];
			}

			uint8* Out = DstRow + X * 4;
			Out[0] = PilClipToUint8(AccB);
			Out[1] = PilClipToUint8(AccG);
			Out[2] = PilClipToUint8(AccR);
			Out[3] = PilClipToUint8(AccA);
		}
	}
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

void ResizePILBilinear(
	const uint8* SrcPixels, int32 SrcW, int32 SrcH,
	uint8*       DstPixels, int32 DstW, int32 DstH)
{
	ResizePIL(SrcPixels, SrcW, SrcH, DstPixels, DstW, DstH,
		PrecomputeBilinearCoeffs(SrcW, DstW),
		PrecomputeBilinearCoeffs(SrcH, DstH));
}

void ResizePILBicubic(
	const uint8* SrcPixels, int32 SrcW, int32 SrcH,
	uint8*       DstPixels, int32 DstW, int32 DstH)
{
	ResizePIL(SrcPixels, SrcW, SrcH, DstPixels, DstW, DstH,
		PrecomputeBicubicCoeffs(SrcW, DstW),
		PrecomputeBicubicCoeffs(SrcH, DstH));
}
