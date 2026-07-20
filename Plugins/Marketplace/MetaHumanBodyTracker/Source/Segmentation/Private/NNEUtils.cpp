// Copyright Epic Games, Inc. All Rights Reserved.

#include "NNEUtils.h"

#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeRunSync.h"
#include "NNETypes.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeNPU.h"

namespace UE::NNE::Segmentation::NNEUtils
{

bool ShapesEqual(
	TConstArrayView<UE::NNE::FTensorShape> A,
	TConstArrayView<UE::NNE::FTensorShape> B)
{
	if (A.Num() != B.Num())
	{
		return false;
	}
	for (int32 I = 0; I < A.Num(); ++I)
	{
		TConstArrayView<uint32> DA = A[I].GetData();
		TConstArrayView<uint32> DB = B[I].GetData();
		if (DA.Num() != DB.Num())
		{
			return false;
		}
		if (FMemory::Memcmp(DA.GetData(), DB.GetData(), DA.Num() * sizeof(uint32)) != 0)
		{
			return false;
		}
	}
	return true;
}

UE::NNE::EResultStatus SetInputTensorShapesIfChanged(
	UE::NNE::IModelInstanceRunSync& Instance,
	TConstArrayView<UE::NNE::FTensorShape> NewShapes,
	TArray<UE::NNE::FTensorShape>& CachedShapes)
{
	if (ShapesEqual(CachedShapes, NewShapes))
	{
		return UE::NNE::EResultStatus::Ok;
	}

	UE::NNE::EResultStatus Result = Instance.SetInputTensorShapes(NewShapes);
	if (Result == UE::NNE::EResultStatus::Ok)
	{
		CachedShapes = TArray<UE::NNE::FTensorShape>(NewShapes.GetData(), NewShapes.Num());
	}
	return Result;
}

TSharedPtr<UE::NNE::IModelInstanceRunSync> TryCreateInstance(
	UNNEModelData* ModelData,
	TConstArrayView<FRuntimePreference> RuntimePreferences
)
{
	for (const FRuntimePreference& Preference : RuntimePreferences)
	{
		TSharedPtr<UE::NNE::IModelInstanceRunSync> Result = TryCreateInstance(Preference.RuntimeName, Preference.DeviceType, ModelData);
		if (Result)
		{
			return Result;
		}
	}
	return nullptr;
}

TSharedPtr<UE::NNE::IModelInstanceRunSync> TryCreateInstance(
	const FString& RuntimeName,
	EDeviceType DeviceType,
	UNNEModelData* ModelData)
{
	switch (DeviceType)
	{
	case EDeviceType::CPU:
	{
		TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(RuntimeName);
		if (!Runtime.IsValid())
		{
			return nullptr;
		}
		if (Runtime->CanCreateModelCPU(ModelData) != UE::NNE::EResultStatus::Ok)
		{
			return nullptr;
		}
		TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(ModelData);
		if (!Model.IsValid())
		{
			return nullptr;
		}
		return Model->CreateModelInstanceCPU();
	}
	case EDeviceType::GPU:
	{
		TWeakInterfacePtr<INNERuntimeGPU> Runtime = UE::NNE::GetRuntime<INNERuntimeGPU>(RuntimeName);
		if (!Runtime.IsValid())
		{
			return nullptr;
		}
		if (Runtime->CanCreateModelGPU(ModelData) != UE::NNE::EResultStatus::Ok)
		{
			return nullptr;
		}
		TSharedPtr<UE::NNE::IModelGPU> Model = Runtime->CreateModelGPU(ModelData);
		if (!Model.IsValid())
		{
			return nullptr;
		}
		return Model->CreateModelInstanceGPU();
	}
	case EDeviceType::NPU:
	{
		TWeakInterfacePtr<INNERuntimeNPU> Runtime = UE::NNE::GetRuntime<INNERuntimeNPU>(RuntimeName);
		if (!Runtime.IsValid())
		{
			return nullptr;
		}
		if (Runtime->CanCreateModelNPU(ModelData) != UE::NNE::EResultStatus::Ok)
		{
			return nullptr;
		}
		TSharedPtr<UE::NNE::IModelNPU> Model = Runtime->CreateModelNPU(ModelData);
		if (!Model.IsValid())
		{
			return nullptr;
		}
		return Model->CreateModelInstanceNPU();
	}
	default:
		return nullptr;
	}
}

} // namespace UE::NNE::Segmentation::NNEUtils
