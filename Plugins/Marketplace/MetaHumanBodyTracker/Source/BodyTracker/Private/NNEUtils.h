// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "BodyTrackerModule.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeRDG.h"
#include "OfflineBodyTracker.h"
#include "RenderGraphUtils.h"
#include "RuntimePreference.h"
#include "Serialization/Archive.h"
#include "String/ParseTokens.h"

namespace UE::NNEUtils::Private
{
	template<class T>
	bool SetShape(TSharedPtr<T> ModelInstance)
	{
		TConstArrayView< UE::NNE::FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
		TArray<UE::NNE::FTensorShape> Shapes;
		for (const UE::NNE::FTensorDesc& Desc : InputDescs)
		{
			if (!Desc.GetShape().IsConcrete())
			{
				UE_LOG(LogBodyTracker, Warning, TEXT("Try to set shape on a model whoose tensors are not all concrete."));
				return false;
			}
			Shapes.Add(UE::NNE::FTensorShape::MakeFromSymbolic(Desc.GetShape()));
		}
		ModelInstance->SetInputTensorShapes(Shapes);
		return true;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> TryCreateInstance(
		UNNEModelData* ModelData,
		TConstArrayView<UE::BodyTracker::FRuntimePreference> RuntimePreferences,
		bool bSetShape
	);
	TSharedPtr<UE::NNE::IModelInstanceCPU> CreateCPUModelInstance(UNNEModelData& ModelData, const FString& RuntimeName, bool bSetShape);
	TSharedPtr<UE::NNE::IModelInstanceGPU> CreateGPUModelInstance(UNNEModelData& ModelData, const FString& RuntimeName, bool bSetShape);
	TSharedPtr<UE::NNE::IModelInstanceRDG> CreateRDGModelInstance(UNNEModelData& ModelData, const FString& RuntimeName, bool bSetShape);
	FRDGBufferDesc CreateRDGBufferDescForTensorDesc(uint32 ElemByteSize, uint64 SizeInByte, bool bIsInput);

	void ConvertBinding(FRDGBuilder& GraphBuilder,
		TConstArrayView<NNE::FTensorDesc> InTensorDescs,
		TConstArrayView<NNE::FTensorBindingCPU> InBindingsCPU,
		TArray<NNE::FTensorBindingRDG>& OutBindingsRDG,
		bool bIsInput);

	void UploadsBindingToGPU(FRDGBuilder& GraphBuilder,
		TConstArrayView<NNE::FTensorBindingCPU> InBindingsCPU,
		TConstArrayView<NNE::FTensorBindingRDG> InBindingsRDG);

	bool EnqueueRDG(
		FRDGBuilder& GraphBuilder,
		NNE::IModelInstanceRDG& ModelInstance,
		FReadbackManager& ReadbackMgr,
		TConstArrayView<NNE::FTensorBindingCPU> InInputBindings,
		TConstArrayView<NNE::FTensorBindingCPU> InOutputBindings);

	template<class T>
	UE::NNE::FTensorBindingCPU ToTensorBindingCPU(TArray64<T>& Array)
	{
		return UE::NNE::FTensorBindingCPU{ Array.GetData(), Array.NumBytes() };
	}

	template<class T>
	UE::NNE::FTensorBindingCPU ToTensorBindingCPU(TArray64<T>& Array, int64 Offset, int64 Count, int64 ElementSize)
	{
		return UE::NNE::FTensorBindingCPU{ Array.GetData() + Offset * ElementSize, Count * ElementSize * sizeof(typename TArray64<T>::ElementType)};
	}

	template<class T>
	UE::NNE::FTensorBindingCPU ToTensorBindingCPU(TArrayView64<T> Array)
	{
		return UE::NNE::FTensorBindingCPU{ Array.GetData(), Array.NumBytes() };
	}

	bool RunSync(UE::NNE::IModelInstanceRDG& ModelInstance, TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors);

} // namespace UE::NNEUtils::Private
