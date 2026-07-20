// Copyright Epic Games, Inc. All Rights Reserved.

#include "MetaHumanBodyTrackerModule.h"
#include "MetaHumanBodyTracker.h"
#include "Features/IModularFeatures.h"



void FMetaHumanBodyTrackerModule::StartupModule()
{
	BodyTracker = MakeUnique<FMetaHumanBodyTracker>();
	IModularFeatures::Get().RegisterModularFeature(IMetaHumanBodyTrackerInterface::GetModularFeatureName(), BodyTracker.Get());

#if WITH_METADATA
	UEnum* Enum = StaticEnum<EMetaHumanBodyTrackerMode>();

	if (Enum)
	{
		int64 Index = Enum->GetIndexByValue(static_cast<int64>(EMetaHumanBodyTrackerMode::Realtime));
		Enum->SetMetaData(TEXT("Hidden"), TEXT("true"), Index);
	}
#endif
}

void FMetaHumanBodyTrackerModule::ShutdownModule()
{
	IModularFeatures::Get().UnregisterModularFeature(IMetaHumanBodyTrackerInterface::GetModularFeatureName(), BodyTracker.Get());
	BodyTracker.Reset();
}

IMPLEMENT_MODULE(FMetaHumanBodyTrackerModule, MetaHumanBodyTracker);
