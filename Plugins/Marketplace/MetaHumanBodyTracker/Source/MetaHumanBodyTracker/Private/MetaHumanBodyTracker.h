// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "MetaHumanBodyTrackerInterface.h"



class FMetaHumanBodyTracker : public IMetaHumanBodyTrackerInterface
{
public:

	virtual bool ExtendPipeline(const FBodyTrackerInputParams& InBodyTrackerInputParams, UE::MetaHuman::Pipeline::FPipeline& InOutPipeline, FBodyTrackerOutputParams& OutBodyTrackerOutputParams) const override;

	virtual TSubclassOf<AMetaHumanBodyDriverActorInterface> GetBodyDriverActorClass() const override;

	virtual FString GetBodyControlRigAssetPath() const override;

	virtual FString GetMetaHumanRetargeterAssetPath() const override;

	virtual USkeletalMesh* MakeMHSkeletalMesh(UObject* InOuter, const TArray<float>& InShape) const override;

	virtual USkeletalMesh* MakeSMPLSkeletalMesh(UObject* InOuter, const TArray<float>& InShape) const override;

#if WITH_EDITOR
	virtual void CustomizePerformanceDetails(IDetailLayoutBuilder& InDetailBuilder) const override;
#endif

	static TArray<TArray<uint8>> GetChunkDataArray(const TArray<uint8>& InPerformanceData);
	static void SetChunkDataArray(const TArray<TArray<uint8>>& InChunkDataArray, TArray<uint8>& InPerformanceData);
	static int32 GetChunkDataIndex(const TArray<TArray<uint8>>& InChunkDataArray, int32 InHeaderId);

	static const int32 SplitProcessingChunkHeader = 1;

	class FSplitProcessingChunk
	{
	public:

		TArray<float> AverageShape;

		float FocalLength = 0;
		float Pitch = 0;
		float Roll = 0;
		int32 BodyTrackingRasterWidth = 0;
		int32 BodyTrackingRasterHeight = 0;
		FVector2f CameraImgCenter = FVector2f(0.f, 0.f);

		FVector Origin = FVector::ZeroVector;

		bool Read(const TArray<uint8>& InChunkData);
		bool Write(TArray<uint8>& InChunkData);
	};
};
