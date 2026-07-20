// Copyright Epic Games, Inc. All Rights Reserved.

#include "Nodes/DummyBodyTrackerNode.h"

#include "MetaHumanBodyTrackerLog.h"
#include "MetaHumanSMPLX.h"

#include "Interfaces/IPluginManager.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"



namespace UE::MetaHuman::Pipeline
{

FDummyBodyTrackerNode::FDummyBodyTrackerNode(const FString& InName) : FNode("DummyBodyTracker", InName)
{
	Pins.Add(FPin("Animation Out", EPinDirection::Output, EPinType::Animation));
}

bool FDummyBodyTrackerNode::Start(const TSharedPtr<FPipelineData>& InPipelineData)
{
	Shape.Reset();
	Pose.Reset();
	Position.Reset();

	FString JsonPath = IPluginManager::Get().FindPlugin(UE_PLUGIN_NAME)->GetContentDir() + "/animation.json";
	FString JsonString;
	if (FFileHelper::LoadFileToString(JsonString, *JsonPath))
	{
		TSharedPtr<FJsonObject> JsonObject;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

		if (FJsonSerializer::Deserialize(Reader, JsonObject))
		{
			TArray<TSharedPtr<FJsonValue>> ShapeValues = JsonObject->GetArrayField(TEXT("shapeParameters"));

			Shape.SetNumUninitialized(ShapeValues.Num());
			for (int32 Index = 0; Index < ShapeValues.Num(); ++Index)
			{
				Shape[Index] = ShapeValues[Index]->AsNumber();
			}

			for (const FString& Field : { TEXT("bodyPose"), TEXT("leftHandPose"), TEXT("rightHandPose") })
			{
				TArray<TSharedPtr<FJsonValue>> FrameJointData = JsonObject->GetArrayField(Field);

				if (Pose.IsEmpty())
				{
					Pose.SetNumZeroed(FrameJointData.Num());
				}

				for (int32 Frame = 0; Frame < FrameJointData.Num(); ++Frame)
				{
					TArray<TSharedPtr<FJsonValue>> JointData = FrameJointData[Frame]->AsArray();

					for (int32 Joint = 0; Joint < JointData.Num(); ++Joint)
					{
						TArray<TSharedPtr<FJsonValue>> JointAngles = JointData[Joint]->AsArray();

						if (JointAngles.Num() == 3)
						{
							Pose[Frame].Add(JointAngles[0]->AsNumber());
							Pose[Frame].Add(JointAngles[1]->AsNumber());
							Pose[Frame].Add(JointAngles[2]->AsNumber());
						}
						else
						{
							check(false);
						}
					}

					if (Field == "bodyPose")
					{
						while (Pose[Frame].Num() < 25 * 3) // Left hand starts joint 25
						{
							Pose[Frame].Add(0);
						}
					}
				}
			}

			TArray<TSharedPtr<FJsonValue>> Translation = JsonObject->GetArrayField(TEXT("bodyTranslation"));

			Position.SetNumZeroed(Translation.Num());

			for (int32 Frame = 0; Frame < Translation.Num(); ++Frame)
			{
				TArray<TSharedPtr<FJsonValue>> FrameTranslationXYZ = Translation[Frame]->AsArray();

				if (FrameTranslationXYZ.Num() == 3)
				{
					Position[Frame] = FVector(FrameTranslationXYZ[0]->AsNumber(), FrameTranslationXYZ[1]->AsNumber(), FrameTranslationXYZ[2]->AsNumber());
				}
				else
				{
					check(false);
				}
			}
		}
	}

	return true;
}

bool FDummyBodyTrackerNode::Process(const TSharedPtr<FPipelineData>& InPipelineData)
{
	if (!SMPLX->IsInitialized())
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::FailedToInitialize);
		InPipelineData->SetErrorNodeMessage(TEXT("Failed to initialize"));
		return false;
	}

	if (Shape.IsEmpty() || Pose.IsEmpty() || Position.IsEmpty())
	{
		InPipelineData->SetErrorNodeCode(ErrorCode::BadData);
		InPipelineData->SetErrorNodeMessage(TEXT("Bad data"));
		return false;
	}

	FFrameAnimationData Output;

	Output.RawBodyAnimationSMPLXShape = Shape;
	Output.RawBodyAnimationSMPLXPose = Pose[InPipelineData->GetFrameNumber() % Pose.Num()];
	Output.RawBodyAnimationSMPLXTranslation = FVector(Position[InPipelineData->GetFrameNumber() % Position.Num()]);
	Output.AnimationQuality = EFrameAnimationQuality::PostFiltered;

	Output.AnimationData.Add("CTRL_expressions_jawOpen", (InPipelineData->GetFrameNumber() % 10) / 10.0);

	PrepareOutput(Output);

	InPipelineData->SetData<FFrameAnimationData>(Pins[0], MoveTemp(Output));

	FPlatformProcess::Sleep(1.0f / 60.0f);

	return true;
}

}
