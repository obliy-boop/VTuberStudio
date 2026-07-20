// Copyright Epic Games, Inc. All Rights Reserved.

#include "MetaHumanBodyTracker.h"
#include "Nodes/RealtimeBodyTrackerNode.h"
#include "Nodes/OfflineBodyTrackerNode.h"
#include "Nodes/OfflineCameraCalibrationNode.h"
#include "Nodes/DummyBodyTrackerNode.h"
#include "Nodes/BodyFaceAnimMergeNode.h"
#include "Nodes/NullBodyAnimationNode.h"
#include "Nodes/StoreFaceAnimationNode.h"
#include "Nodes/RestoreFaceAnimationNode.h"
#include "MetaHumanBodyDriverActor.h"
#include "MetaHumanPerformance.h"



class FBodyTrackerData : public IMetaHumanBodyTrackerInterface::FBodyTrackerDataInterface
{
public:

	TSharedPtr<UE::MetaHuman::Pipeline::FNode> ImageSrcNode;
	TSharedPtr<UE::MetaHuman::Pipeline::FOfflineCameraCalibrationNode> OfflineCameraCalibrationNode;
	TSharedPtr<UE::MetaHuman::Pipeline::FOfflineBodyDetectionNode> OfflineBodyDetectionNode;
	TSharedPtr<UE::MetaHuman::Pipeline::FOffline2DKeypointEstimationNode> Offline2DKeypointEstimationNode;
	TSharedPtr<UE::MetaHuman::Pipeline::FOfflinePoseEstimationNode> OfflinePoseEstimationNode;
	TSharedPtr<UE::MetaHuman::Pipeline::FStoreFaceAnimationNode> StoreFaceAnimationNode;
}; 



bool FMetaHumanBodyTracker::ExtendPipeline(const FBodyTrackerInputParams& InBodyTrackerInputParams, UE::MetaHuman::Pipeline::FPipeline& InOutPipeline, FBodyTrackerOutputParams& OutBodyTrackerOutputParams) const
{
	TSharedPtr<UE::MetaHuman::Pipeline::FNode> BodyTrackerNode;
	TSharedPtr<UE::MetaHuman::Pipeline::FNode> FaceAnimSrcNode = InBodyTrackerInputParams.FaceAnimSrcNode;

	if (InBodyTrackerInputParams.BodyTrackerMode == EMetaHumanBodyTrackerMode::Realtime)
	{
		TSharedPtr<UE::MetaHuman::Pipeline::FRealtimeBodyTrackerNode> RealtimeBodyTrackerNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FRealtimeBodyTrackerNode>("RealtimeBodyTracker");

		RealtimeBodyTrackerNode->ReportedAnimationQuality = EFrameAnimationQuality::PostFiltered;

		BodyTrackerNode = RealtimeBodyTrackerNode;

		OutBodyTrackerOutputParams.BodyTrackerFinalPipelineStage = 0;
	}
	else if (InBodyTrackerInputParams.BodyTrackerMode == EMetaHumanBodyTrackerMode::Offline)
	{
		// Retrieve stored state
		TSharedPtr<FBodyTrackerData> BodyTrackerData = StaticCastSharedPtr<FBodyTrackerData>(InBodyTrackerInputParams.BodyTrackerData);

		if (InBodyTrackerInputParams.PipelineStage == 0)
		{
			// Offline camera calibration tracker
			TSharedPtr<UE::MetaHuman::Pipeline::FOfflineCameraCalibrationNode> OfflineCameraCalibrationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FOfflineCameraCalibrationNode>("OfflineCameraCalibration");

			const FFrameRange& FrameRange = InBodyTrackerInputParams.PipelineFrameRanges[InBodyTrackerInputParams.PipelineFrameRangesIndex];
			OfflineCameraCalibrationNode->Stride = FMath::Max(1, (FrameRange.EndFrame - FrameRange.StartFrame + 1) / 30);

			InOutPipeline.MakeConnection(InBodyTrackerInputParams.ImageSrcNode, OfflineCameraCalibrationNode);

			// Body animation
			if (1) //InBodyTrackerInputParams.bSkipPreview)
			{
				// Null body animation generator node
				TSharedPtr<UE::MetaHuman::Pipeline::FNullBodyAnimationNode> NullBodyAnimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FNullBodyAnimationNode>("NullBodyAnimation");

				NullBodyAnimationNode->ReportedAnimationQuality = EFrameAnimationQuality::Preview;

				BodyTrackerNode = NullBodyAnimationNode;
			}
			else
			{
				check(false);
			}

			// Store facial animation calculated in stage 1 to be fed out in stage 4 - this way face is not seen until body is ready
			TSharedPtr<UE::MetaHuman::Pipeline::FStoreFaceAnimationNode> StoreFaceAnimationNode;

			if (FaceAnimSrcNode)
			{
				StoreFaceAnimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FStoreFaceAnimationNode>("StoreFaceAnimation");

				InOutPipeline.MakeConnection(FaceAnimSrcNode, StoreFaceAnimationNode);

				FaceAnimSrcNode = nullptr;
			}

			// Create stored state
			BodyTrackerData = MakeShared<FBodyTrackerData>();
			BodyTrackerData->ImageSrcNode = InBodyTrackerInputParams.ImageSrcNode;
			BodyTrackerData->OfflineCameraCalibrationNode = OfflineCameraCalibrationNode;
			BodyTrackerData->StoreFaceAnimationNode = StoreFaceAnimationNode;
		}
		else if (InBodyTrackerInputParams.PipelineStage == 1)
		{
			// Add image source node back in
			InOutPipeline.AddNode(BodyTrackerData->ImageSrcNode);

			// Offline tracker collects data but produces no output
			TSharedPtr<UE::MetaHuman::Pipeline::FOfflineBodyDetectionNode> OfflineBodyDetectionNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FOfflineBodyDetectionNode>("OfflineBodyDetection");
			
			if (InBodyTrackerInputParams.Performance)
			{
				OfflineBodyDetectionNode->DetectionScoreThreshold = InBodyTrackerInputParams.Performance->BodyDetectionConfidence;
				OfflineBodyDetectionNode->TrackingScoreThreshold = InBodyTrackerInputParams.Performance->BodyTrackingConfidence;
			}
			
			InOutPipeline.MakeConnection(BodyTrackerData->ImageSrcNode, OfflineBodyDetectionNode);

			BodyTrackerData->OfflineBodyDetectionNode = OfflineBodyDetectionNode;

			// Body animation
			if (1) // InBodyTrackerInputParams.bSkipPreview)
			{
				// Null body animation generator node
				TSharedPtr<UE::MetaHuman::Pipeline::FNullBodyAnimationNode> NullBodyAnimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FNullBodyAnimationNode>("NullBodyAnimation");

				NullBodyAnimationNode->ReportedAnimationQuality = EFrameAnimationQuality::Custom1;

				BodyTrackerNode = NullBodyAnimationNode;
			}
			else
			{
				check(false);
			}
		}
		else if (InBodyTrackerInputParams.PipelineStage == 2)
		{
			// Add image source node back in
			InOutPipeline.AddNode(BodyTrackerData->ImageSrcNode);

			// Offline tracker collects data but produces no output
			TSharedPtr<UE::MetaHuman::Pipeline::FOffline2DKeypointEstimationNode> Offline2DKeypointEstimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FOffline2DKeypointEstimationNode>("Offline2DKeypointEstimation");
			Offline2DKeypointEstimationNode->OfflineBodyDetectionNode = BodyTrackerData->OfflineBodyDetectionNode;

			InOutPipeline.MakeConnection(BodyTrackerData->ImageSrcNode, Offline2DKeypointEstimationNode);

			BodyTrackerData->Offline2DKeypointEstimationNode = Offline2DKeypointEstimationNode;

			// Body animation
			if (1) // InBodyTrackerInputParams.bSkipPreview)
			{
				// Null body animation generator node
				TSharedPtr<UE::MetaHuman::Pipeline::FNullBodyAnimationNode> NullBodyAnimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FNullBodyAnimationNode>("NullBodyAnimation");

				NullBodyAnimationNode->ReportedAnimationQuality = EFrameAnimationQuality::Custom2;

				BodyTrackerNode = NullBodyAnimationNode;
			}
			else
			{
				check(false);
			}
		}
		else if (InBodyTrackerInputParams.PipelineStage == 3)
		{
			// Add image source node back in
			InOutPipeline.AddNode(BodyTrackerData->ImageSrcNode);

			// Offline tracker collects data but produces no output
			TSharedPtr<UE::MetaHuman::Pipeline::FOfflinePoseEstimationNode> OfflinePoseEstimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FOfflinePoseEstimationNode>("OfflinePoseEstimation");
			OfflinePoseEstimationNode->OfflineCameraCalibrationNode = BodyTrackerData->OfflineCameraCalibrationNode;
			OfflinePoseEstimationNode->OfflineBodyDetectionNode = BodyTrackerData->OfflineBodyDetectionNode;
			OfflinePoseEstimationNode->Offline2DKeypointEstimationNode = BodyTrackerData->Offline2DKeypointEstimationNode;
			OfflinePoseEstimationNode->SourceFps = InBodyTrackerInputParams.Fps;
			OfflinePoseEstimationNode->BodyHeight = InBodyTrackerInputParams.bAutoBodyHeight ? 0.0f : InBodyTrackerInputParams.BodyHeight;

			InOutPipeline.MakeConnection(BodyTrackerData->ImageSrcNode, OfflinePoseEstimationNode);

			BodyTrackerData->OfflinePoseEstimationNode = OfflinePoseEstimationNode;

			// Body animation
			if (1) // InBodyTrackerInputParams.bSkipPreview)
			{
				// Null body animation generator node
				TSharedPtr<UE::MetaHuman::Pipeline::FNullBodyAnimationNode> NullBodyAnimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FNullBodyAnimationNode>("NullBodyAnimation");

				NullBodyAnimationNode->ReportedAnimationQuality = EFrameAnimationQuality::Final; // Final isn't the last stage!

				BodyTrackerNode = NullBodyAnimationNode;
			}
			else
			{
				check(false);
			}
		}
		else if (InBodyTrackerInputParams.PipelineStage == 4)
		{
			// Offline tracker finalize feeds out the data collected by the previous stages
			TSharedPtr<UE::MetaHuman::Pipeline::FOfflineBodyTrackerFinalizeNode> OfflineBodyTrackerFinalizeNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FOfflineBodyTrackerFinalizeNode>("OfflineBodyTrackerFinalize");

			OfflineBodyTrackerFinalizeNode->OfflineBodyTrackerNode = BodyTrackerData->OfflinePoseEstimationNode;
			OfflineBodyTrackerFinalizeNode->bEnableFootLocking = InBodyTrackerInputParams.bEnableFootLocking;

			// Is there existing body animation somewhere in the sequence? If so then the Performances's AdditionalBodyTrackerData will have
			// serialized info used in that solve - like camera params, shape - that we want to use for this solve in order
			// to maintain consistency. Extract that data and insert it into the nodes.
			for (const FFrameAnimationData& FrameData : InBodyTrackerInputParams.AnimationData)
			{
				if (!FrameData.RawBodyAnimationSMPLXShape.IsEmpty() && !FrameData.RawBodyAnimationSMPLXPose.IsEmpty() && InBodyTrackerInputParams.Performance)
				{
					const TArray<TArray<uint8>> ChunkDataArray = FMetaHumanBodyTracker::GetChunkDataArray(InBodyTrackerInputParams.Performance->AdditionalBodyTrackerData);

					const int32 SplitProcessingChunkIndex = FMetaHumanBodyTracker::GetChunkDataIndex(ChunkDataArray, FMetaHumanBodyTracker::SplitProcessingChunkHeader);

					if (SplitProcessingChunkIndex != -1)
					{
						FSplitProcessingChunk SplitProcessingChunk;

						if (SplitProcessingChunk.Read(ChunkDataArray[SplitProcessingChunkIndex]))
						{
							OfflineBodyTrackerFinalizeNode->AverageShape = SplitProcessingChunk.AverageShape;

							OfflineBodyTrackerFinalizeNode->OfflineBodyTrackerNode->OfflineCameraCalibrationNode->FocalLength = SplitProcessingChunk.FocalLength;
							OfflineBodyTrackerFinalizeNode->OfflineBodyTrackerNode->OfflineCameraCalibrationNode->Pitch = SplitProcessingChunk.Pitch;
							OfflineBodyTrackerFinalizeNode->OfflineBodyTrackerNode->OfflineCameraCalibrationNode->Roll = SplitProcessingChunk.Roll;
							OfflineBodyTrackerFinalizeNode->OfflineBodyTrackerNode->OfflineCameraCalibrationNode->BodyTrackingRasterWidth = SplitProcessingChunk.BodyTrackingRasterWidth;
							OfflineBodyTrackerFinalizeNode->OfflineBodyTrackerNode->OfflineCameraCalibrationNode->BodyTrackingRasterHeight = SplitProcessingChunk.BodyTrackingRasterHeight;
							OfflineBodyTrackerFinalizeNode->OfflineBodyTrackerNode->OfflineCameraCalibrationNode->CameraImgCenter = SplitProcessingChunk.CameraImgCenter;

							OfflineBodyTrackerFinalizeNode->bOriginSet = true;
							OfflineBodyTrackerFinalizeNode->Origin = SplitProcessingChunk.Origin;
						}
					}

					break;
				}
			}

			OfflineBodyTrackerFinalizeNode->Performance = InBodyTrackerInputParams.Performance;

			BodyTrackerNode = OfflineBodyTrackerFinalizeNode;

			if (BodyTrackerData->StoreFaceAnimationNode)
			{
				TSharedPtr<UE::MetaHuman::Pipeline::FRestoreFaceAnimationNode> RestoreFaceAnimationNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FRestoreFaceAnimationNode>("RestoreFaceAnimation");

				RestoreFaceAnimationNode->AnimationData = BodyTrackerData->StoreFaceAnimationNode->AnimationData;

				FaceAnimSrcNode = RestoreFaceAnimationNode;
			}
		}
		else
		{
			check(false);
		}

		OutBodyTrackerOutputParams.BodyTrackerData = BodyTrackerData;

		OutBodyTrackerOutputParams.BodyTrackerFinalPipelineStage = 4;
	}
	else
	{
		check(false);
		return false;
	}

	if (InBodyTrackerInputParams.ImageSrcNode)
	{
		InOutPipeline.MakeConnection(InBodyTrackerInputParams.ImageSrcNode, BodyTrackerNode);
	}

	if (FaceAnimSrcNode)
	{
		TSharedPtr<UE::MetaHuman::Pipeline::FBodyFaceAnimMergeNode> MergeNode = InOutPipeline.MakeNode<UE::MetaHuman::Pipeline::FBodyFaceAnimMergeNode>("BodyFaceMerge");

		InOutPipeline.MakeConnection(FaceAnimSrcNode, MergeNode, 0, 0);
		InOutPipeline.MakeConnection(BodyTrackerNode, MergeNode, 0, 1);

		OutBodyTrackerOutputParams.AnimationPinName = MergeNode->Name + ".Animation Out";
	}
	else
	{
		OutBodyTrackerOutputParams.AnimationPinName = BodyTrackerNode->Name + ".Animation Out";
	}

	return true;
}

TSubclassOf<AMetaHumanBodyDriverActorInterface> FMetaHumanBodyTracker::GetBodyDriverActorClass() const
{
	return AMetaHumanBodyDriverActor::StaticClass();
}

FString FMetaHumanBodyTracker::GetBodyControlRigAssetPath() const
{
	return "/" UE_PLUGIN_NAME "/MetaHuman_ControlRig_Simple.MetaHuman_ControlRig_Simple_C";
}

FString FMetaHumanBodyTracker::GetMetaHumanRetargeterAssetPath() const
{
	return "/" UE_PLUGIN_NAME "/RTG_MH_IKRig.RTG_MH_IKRig";
}

USkeletalMesh* FMetaHumanBodyTracker::MakeMHSkeletalMesh(UObject* InOuter, const TArray<float>& InShape) const
{
	FMetaHumanSMPLX SMPLX;

	SMPLX.Init();
	SMPLX.SetAccountForHeight(true);
	SMPLX.SetShape(InShape);

	USkeleton* MHSkeleton = SMPLX.GetMHSkeleton(InOuter);
	USkeleton* SMPLXSkeleton = SMPLX.CreateSMPLXSkeleton(InOuter);

	return SMPLX.CreateMHSkeletalMesh(InOuter, MHSkeleton, SMPLXSkeleton);
}

USkeletalMesh* FMetaHumanBodyTracker::MakeSMPLSkeletalMesh(UObject* InOuter, const TArray<float>& InShape) const
{
	FMetaHumanSMPLX SMPLX;

	SMPLX.Init();
	SMPLX.SetAccountForHeight(true);
	SMPLX.SetShape(InShape);

	USkeleton* SMPLXSkeleton = SMPLX.CreateSMPLXSkeleton(InOuter);

	return SMPLX.CreateSMPLXSkeletalMesh(InOuter, SMPLXSkeleton, true /* bIsSkinned */);
}

#if WITH_EDITOR
void FMetaHumanBodyTracker::CustomizePerformanceDetails(IDetailLayoutBuilder& InDetailBuilder) const
{
}
#endif

TArray<TArray<uint8>> FMetaHumanBodyTracker::GetChunkDataArray(const TArray<uint8>& InPerformanceData)
{
	FMemoryReader Reader(InPerformanceData);

	TArray<TArray<uint8>> ChunkDataArray;

	while (Reader.Tell() != Reader.TotalSize())
	{
		TArray<uint8> ChunkData;
		Reader << ChunkData;
		ChunkDataArray.Add(ChunkData);
	}

	return ChunkDataArray;
}

void FMetaHumanBodyTracker::SetChunkDataArray(const TArray<TArray<uint8>>& InChunkDataArray, TArray<uint8>& InPerformanceData)
{
	InPerformanceData.Reset();

	FMemoryWriter Writer(InPerformanceData);

	for (TArray<uint8> ChunkData : InChunkDataArray)
	{
		Writer << ChunkData;
	}
}

int32 FMetaHumanBodyTracker::GetChunkDataIndex(const TArray<TArray<uint8>>& InChunkDataArray, int32 InHeaderId)
{
	for (int32 Index = 0; Index < InChunkDataArray.Num(); ++Index)
	{
		FMemoryReader Reader(InChunkDataArray[Index]);

		int32 Header = -1;
		Reader << Header;

		if (Header == InHeaderId)
		{
			return Index;
		}
	}

	return -1;
}

bool FMetaHumanBodyTracker::FSplitProcessingChunk::Read(const TArray<uint8>& InChunkData)
{
	FMemoryReader Reader(InChunkData);

	int32 Header = -1;
	Reader << Header;

	if (Header != FMetaHumanBodyTracker::SplitProcessingChunkHeader)
	{
		return false;
	}

	Reader << AverageShape;

	Reader << FocalLength;
	Reader << Pitch;
	Reader << Roll;
	Reader << BodyTrackingRasterWidth;
	Reader << BodyTrackingRasterHeight;
	Reader << CameraImgCenter;

	Reader << Origin;

	return true;
}

bool FMetaHumanBodyTracker::FSplitProcessingChunk::Write(TArray<uint8>& InChunkData)
{
	FMemoryWriter Writer(InChunkData);

	int32 Header = FMetaHumanBodyTracker::SplitProcessingChunkHeader;
	Writer << Header;

	Writer << AverageShape;

	Writer << FocalLength;
	Writer << Pitch;
	Writer << Roll;
	Writer << BodyTrackingRasterWidth;
	Writer << BodyTrackingRasterHeight;
	Writer << CameraImgCenter;

	Writer << Origin;

	return true;
}
