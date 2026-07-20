// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "HAL/Platform.h"
#include "Containers/Array.h"

/**
 * PIL-compatible separable image resize for BGRA8 uint8 images.
 *
 * Implements the same fixed-point two-pass algorithm as Pillow's
 * ImagingResampleHorizontal/Vertical (imaging.c), matching byte-for-byte
 * output for the bilinear and bicubic filter modes.
 *
 * Included by both Detectron2.cpp (bilinear) and SAM2.cpp (bicubic).
 */

// Fixed-point precision: 22 bits, matching Pillow's PRECISION_BITS.
static constexpr int32 PilPrecisionBits = 32 - 8 - 2;

/** Per-axis resample coefficients (precomputed for a single dimension). */
struct FResampleCoeffs
{
	TArray<int32> Bounds;   // [OutSize * 2]: (xmin, count) per output pixel
	TArray<int32> Weights;  // [OutSize * KSize]: fixed-point weights (may be negative)
	int32 KSize = 0;
};

FResampleCoeffs PrecomputeBilinearCoeffs(int32 InSize, int32 OutSize);
FResampleCoeffs PrecomputeBicubicCoeffs(int32 InSize, int32 OutSize);

void ResizePIL(
	const uint8* SrcPixels, int32 SrcW, int32 SrcH,
	uint8*       DstPixels, int32 DstW, int32 DstH,
	const FResampleCoeffs& HCoeffs,
	const FResampleCoeffs& VCoeffs);

void ResizePILBilinear(
	const uint8* SrcPixels, int32 SrcW, int32 SrcH,
	uint8*       DstPixels, int32 DstW, int32 DstH);

void ResizePILBicubic(
	const uint8* SrcPixels, int32 SrcW, int32 SrcH,
	uint8*       DstPixels, int32 DstW, int32 DstH);
