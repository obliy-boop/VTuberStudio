// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Modules/ModuleManager.h"
#include "Pipeline/DataTreeTypes.h"

DECLARE_LOG_CATEGORY_EXTERN(LogBodyTracker, Log, All);

class FBodyTrackerModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
