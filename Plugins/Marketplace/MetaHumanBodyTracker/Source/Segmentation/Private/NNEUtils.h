// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Segmentation.h"
#include "NNERuntimeRunSync.h"
#include "NNETypes.h"

class UNNEModelData;

namespace UE::NNE::Segmentation::NNEUtils
{

/** Compares two shape arrays element-wise. Returns true if they match exactly. */
bool ShapesEqual(
	TConstArrayView<UE::NNE::FTensorShape> A,
	TConstArrayView<UE::NNE::FTensorShape> B);

/**
 * Calls SetInputTensorShapes only if the shapes differ from CachedShapes.
 * Updates CachedShapes on success.
 */
UE::NNE::EResultStatus SetInputTensorShapesIfChanged(
	UE::NNE::IModelInstanceRunSync& Instance,
	TConstArrayView<UE::NNE::FTensorShape> NewShapes,
	TArray<UE::NNE::FTensorShape>& CachedShapes);

/**
 * Attempts to create an NNE model instance given a list of runtime preferences
 * Returns nullptr if it wasn't able to produce a model instance with any of the
 * preferences
 */
TSharedPtr<UE::NNE::IModelInstanceRunSync> TryCreateInstance(
	UNNEModelData* ModelData,
	TConstArrayView<FRuntimePreference> RuntimePreferences
);

/**
 * Attempts to create an NNE model instance for a given runtime name and device type.
 * Returns nullptr if the runtime doesn't exist, doesn't support the device type,
 * or can't create a model/instance from the given model data.
 */
TSharedPtr<UE::NNE::IModelInstanceRunSync> TryCreateInstance(
	const FString& RuntimeName,
	EDeviceType DeviceType,
	UNNEModelData* ModelData);

} // namespace UE::NNE::Segmentation::NNEUtils
