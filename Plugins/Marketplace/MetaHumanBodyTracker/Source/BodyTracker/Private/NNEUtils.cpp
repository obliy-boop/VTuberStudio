// Copyright Epic Games, Inc. All Rights Reserved.

#include "NNEUtils.h"

#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeNPU.h"
#include "NNERuntimeRDG.h"
#include "BodyTrackerModule.h"
#include "RenderGraphUtils.h"
#include "Serialization/Archive.h"
#include "String/ParseTokens.h"

namespace UE::NNEUtils::Private
{
	
	TSharedPtr<UE::NNE::IModelInstanceCPU> CreateCPUModelInstance(UNNEModelData& ModelData, const FString& RuntimeName, bool bSetShape)
	{
		check(!RuntimeName.IsEmpty());

		TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(RuntimeName);
		if (!Runtime.IsValid())
		{
			UE_LOG(LogBodyTracker, Log, TEXT("Could not create model instance. No CPU runtime '%s' found. Valid CPU runtimes are: "), *RuntimeName);
			for (const FString& Name : UE::NNE::GetAllRuntimeNames<INNERuntimeCPU>())
			{
				UE_LOG(LogBodyTracker, Log, TEXT("- %s"), *Name);
			}
			return {};
		}

		if (Runtime->CanCreateModelCPU(&ModelData) != INNERuntimeCPU::ECanCreateModelCPUStatus::Ok)
		{
			UE_LOG(LogBodyTracker, Log, TEXT("%s on CPU can not create model"), *RuntimeName);
			return {};
		}
		TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(&ModelData);
		if (!Model.IsValid())
		{
			UE_LOG(LogBodyTracker, Log, TEXT("Could not create model using %s on CPU"), *RuntimeName);
			return {};
		}

		TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance = Model->CreateModelInstanceCPU();
		if (!ModelInstance.IsValid())
		{
			UE_LOG(LogBodyTracker, Log, TEXT("Could not create model instance using %s on CPU"), *RuntimeName);
			return {};
		}
		if (bSetShape)
		{
			SetShape(ModelInstance);
		}
		return ModelInstance;
	}


	TSharedPtr<UE::NNE::IModelInstanceGPU> CreateGPUModelInstance(UNNEModelData& ModelData, const FString& RuntimeName, bool bSetShape)
	{
		check(!RuntimeName.IsEmpty());

		TWeakInterfacePtr<INNERuntimeGPU> Runtime = UE::NNE::GetRuntime<INNERuntimeGPU>(RuntimeName);
		if (!Runtime.IsValid())
		{
			UE_LOG(LogBodyTracker, Log, TEXT("Could not create model instance. No GPU runtime '%s' found. Valid GPU runtimes are: "), *RuntimeName);
			for (const FString& Name : UE::NNE::GetAllRuntimeNames<INNERuntimeGPU>())
			{
				UE_LOG(LogBodyTracker, Log, TEXT("- %s"), *Name);
			}
			return {};
		}

		if (Runtime->CanCreateModelGPU(&ModelData) != INNERuntimeGPU::ECanCreateModelGPUStatus::Ok)
		{
			UE_LOG(LogBodyTracker, Log, TEXT("%s on GPU can not create model"), *RuntimeName);
			return {};
		}
		TSharedPtr<UE::NNE::IModelGPU> Model = Runtime->CreateModelGPU(&ModelData);
		if (!Model.IsValid())
		{
			UE_LOG(LogBodyTracker, Log, TEXT("Could not create model using %s on GPU"), *RuntimeName);
			return {};
		}

		TSharedPtr<UE::NNE::IModelInstanceGPU> ModelInstance = Model->CreateModelInstanceGPU();
		if (!ModelInstance.IsValid())
		{
			UE_LOG(LogBodyTracker, Log, TEXT("Could not create model instance using %s on GPU"), *RuntimeName);
			return {};
		}
		if (bSetShape)
		{
			SetShape(ModelInstance);
		}
		return ModelInstance;
	}

	TSharedPtr<UE::NNE::IModelInstanceRDG> CreateRDGModelInstance(UNNEModelData& ModelData, const FString& RuntimeName, bool bSetShape)
	{
		check(!RuntimeName.IsEmpty());

		TWeakInterfacePtr<INNERuntimeRDG> Runtime = UE::NNE::GetRuntime<INNERuntimeRDG>(RuntimeName);
		if (!Runtime.IsValid())
		{
			UE_LOG(LogBodyTracker, Warning, TEXT("Could not create model instance. No RDG runtime '%s' found. Valid RDG runtimes are: "), *RuntimeName);
			for (const FString& Name : UE::NNE::GetAllRuntimeNames<INNERuntimeRDG>())
			{
				UE_LOG(LogBodyTracker, Warning, TEXT("- %s"), *Name);
			}
			return {};
		}

		if (Runtime->CanCreateModelRDG(&ModelData) != INNERuntimeRDG::ECanCreateModelRDGStatus::Ok)
		{
			UE_LOG(LogBodyTracker, Warning, TEXT("%s on RDG can not create model"), *RuntimeName);
			return {};
		}

		TSharedPtr<UE::NNE::IModelRDG> Model = Runtime->CreateModelRDG(&ModelData);
		if (!Model.IsValid())
		{
			UE_LOG(LogBodyTracker, Warning, TEXT("Could not create model using %s on RDG"), *RuntimeName);
			return {};
		}

		TSharedPtr<UE::NNE::IModelInstanceRDG> ModelInstance = Model->CreateModelInstanceRDG();
		if (!ModelInstance.IsValid())
		{
			UE_LOG(LogBodyTracker, Warning, TEXT("Could not create model instance using %s on RDG"), *RuntimeName);
			return {};
		}
		if (bSetShape)
		{
			SetShape(ModelInstance);
		}

		return ModelInstance;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> TryCreateInstance(
		const FString& RuntimeName,
		UE::BodyTracker::EDeviceType DeviceType,
		UNNEModelData* ModelData)
	{
		switch (DeviceType)
		{
		case UE::BodyTracker::EDeviceType::CPU:
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
		case UE::BodyTracker::EDeviceType::GPU:
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
		case UE::BodyTracker::EDeviceType::NPU:
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

	TSharedPtr<UE::NNE::IModelInstanceRunSync> TryCreateInstance(
		UNNEModelData* ModelData,
		TConstArrayView<UE::BodyTracker::FRuntimePreference> RuntimePreferences,
		bool bSetShape
	)
	{
		if (!ModelData)
		{
			return nullptr;
		}
		for (const UE::BodyTracker::FRuntimePreference& Preference : RuntimePreferences)
		{
			TSharedPtr<UE::NNE::IModelInstanceRunSync> Result = TryCreateInstance(Preference.RuntimeName, Preference.DeviceType, ModelData);
			if (Result)
			{
				if (bSetShape)
				{
					SetShape(Result);
				}
				return Result;
			}
		}
		return nullptr;
	}

	

	FRDGBufferDesc CreateRDGBufferDescForTensorDesc(uint32 ElemByteSize, uint64 SizeInByte, bool bIsInput)
	{
		uint64 MinSizeMultiple = FGenericPlatformMath::Max((uint64)64, (uint64)ElemByteSize);

		//Round up to the next multiple of MinSizeMultiple
		SizeInByte = FMath::DivideAndRoundUp(SizeInByte, MinSizeMultiple) * MinSizeMultiple;

		FRDGBufferDesc Desc = FRDGBufferDesc::CreateBufferDesc(ElemByteSize, SizeInByte / ElemByteSize);

		if (bIsInput)
		{
			Desc.Usage = EBufferUsageFlags(Desc.Usage | BUF_Static);
		}
		else
		{
			Desc.Usage = EBufferUsageFlags(Desc.Usage | BUF_SourceCopy | BUF_UnorderedAccess);
		}

		return Desc;
	}

	void ConvertBinding(FRDGBuilder& GraphBuilder,
		TConstArrayView<NNE::FTensorDesc> InTensorDescs,
		TConstArrayView<NNE::FTensorBindingCPU> InBindingsCPU,
		TArray<NNE::FTensorBindingRDG>& OutBindingsRDG,
		bool bIsInput)
	{
		check(IsInRenderingThread());
		check(InTensorDescs.Num() == InBindingsCPU.Num());

		for (int32 Idx = 0; Idx < InBindingsCPU.Num(); ++Idx)
		{
			const NNE::FTensorDesc& TensorDesc = InTensorDescs[Idx];
			const NNE::FTensorBindingCPU& BindingCPU = InBindingsCPU[Idx];
			const FRDGBufferDesc Desc = CreateRDGBufferDescForTensorDesc(TensorDesc.GetElementByteSize(), BindingCPU.SizeInBytes, bIsInput);

			OutBindingsRDG.Add({ GraphBuilder.CreateBuffer(Desc, *TensorDesc.GetName(), ERDGBufferFlags::None) });
		}
	}

	void UploadsBindingToGPU(FRDGBuilder& GraphBuilder,
		TConstArrayView<NNE::FTensorBindingCPU> InBindingsCPU,
		TConstArrayView<NNE::FTensorBindingRDG> InBindingsRDG)
	{
		check(IsInRenderingThread());
		check(InBindingsCPU.Num() == InBindingsRDG.Num());

		for (int32 Idx = 0; Idx < InBindingsCPU.Num(); ++Idx)
		{
			const NNE::FTensorBindingCPU& BindingCPU = InBindingsCPU[Idx];
			const NNE::FTensorBindingRDG& BindingRDG = InBindingsRDG[Idx];

			GraphBuilder.QueueBufferUpload(BindingRDG.Buffer, BindingCPU.Data, BindingCPU.SizeInBytes, ERDGInitialDataFlags::NoCopy);
		}
	}

	DECLARE_GPU_STAT_NAMED(FBodyTrackerRDG, TEXT("BodyTracker.EnqueueRDG"));

	bool EnqueueRDG(
		FRDGBuilder& GraphBuilder,
		NNE::IModelInstanceRDG& ModelInstance,
		FReadbackManager& ReadbackMgr,
		TConstArrayView<NNE::FTensorBindingCPU> InInputBindings,
		TConstArrayView<NNE::FTensorBindingCPU> InOutputBindings)
	{
		TArray<NNE::FTensorBindingRDG> InputBindingsRDG;
		TArray<NNE::FTensorBindingRDG> OutputBindingsRDG;

		ConvertBinding(GraphBuilder, ModelInstance.GetInputTensorDescs(), InInputBindings, InputBindingsRDG, true);
		ConvertBinding(GraphBuilder, ModelInstance.GetOutputTensorDescs(), InOutputBindings, OutputBindingsRDG, false);

		UploadsBindingToGPU(GraphBuilder, InInputBindings, InputBindingsRDG);

		bool Result;

		{
			RDG_EVENT_SCOPE_STAT(GraphBuilder, FBodyTrackerRDG, "EnqueueRDG");

			Result = ModelInstance.EnqueueRDG(GraphBuilder, InputBindingsRDG, OutputBindingsRDG) == NNE::IModelInstanceRDG::EEnqueueRDGStatus::Ok;
		}

		if (!ReadbackMgr.EnqueueReadbacks(GraphBuilder, OutputBindingsRDG, InOutputBindings))
		{
			UE_LOG(LogNNE, Error, TEXT("FModelQA:DownloadBindingToCPU() failed"));
			return false;
		}

		return Result;
	}

	bool RunSync(UE::NNE::IModelInstanceRDG& ModelInstance, TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors)
	{
		SCOPED_NAMED_EVENT_TEXT("BodyTracker::RunSunc", FColor::Magenta);

		FReadbackManager ReadbackManager;
		bool Result = false;

		ENQUEUE_RENDER_COMMAND(FModelInstanceRDG_Run)
			(
				[&ModelInstance, InInputTensors, InOutputTensors, &ReadbackManager, &Result](FRHICommandListImmediate& RHICmdList)
				{
					TOptional<ERHIPipeline> Pipeline = RHICmdList.GetPipeline();

					if (Pipeline == ERHIPipeline::None)
					{
						RHICmdList.SwitchPipeline(ERHIPipeline::Graphics);
					}

					FRDGBuilder	GraphBuilder(RHICmdList);

					Result = EnqueueRDG(GraphBuilder, ModelInstance, ReadbackManager, InInputTensors, InOutputTensors);

					GraphBuilder.Execute();
				}
				);

		// Wait for readbacks to be processed
		if (!ReadbackManager.Wait())
		{
			UE_LOG(LogBodyTracker, Error, TEXT("Failed to process all readbacks"));
			return false;
		}
		return Result;
	}

} // namespace UE::NNEUtils::Private
