// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "ImageCore.h"

namespace UE::NNE::Segmentation
{

// --- Enums ---

enum class EDeviceType : uint8
{
	CPU,
	GPU,
	NPU
};

struct FRuntimePreference
{
	FString RuntimeName;
	EDeviceType DeviceType;
};

// --- Data types ---

struct FDetection
{
	FIntRect Box;                         // may be empty
	float Score = 0.0f;
	TArray<TArray<float>> Keypoints;      // N x [x, y, confidence] — may be empty
	FImage Mask;                          // Binary mask at original resolution — may be empty
};

struct FTracking : FDetection
{
	int32 Id = 0;
};

// --- Interfaces ---

class SEGMENTATION_API IDetector
{
public:
	virtual ~IDetector() = default;
	virtual TArray<FDetection> Detect(const FImage& Frame) = 0;
};

class SEGMENTATION_API ITracker
{
public:
	virtual ~ITracker() = default;
	virtual TArray<FTracking> Initialize(const FImage& Frame, TConstArrayView<FDetection> Detections) = 0;
	virtual TArray<FTracking> Track(const FImage& Frame) = 0;
};

} // namespace UE::NNE::Segmentation
