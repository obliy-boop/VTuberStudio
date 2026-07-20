// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Modules/ModuleInterface.h"

/**
 * IModuleInterface for the MetaHumanBodyOptimizer plugin module.
 */
class FMetaHumanBodyOptimizerModule final : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
