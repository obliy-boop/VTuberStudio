// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "resourceloader/IMetaHumanFileResourceLoader.h"

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

class FMetaHumanResourceLoaderHub
{
public:
	/** Install a process-wide loader for the singleton */
	static void Install(IMetaHumanFileResourceLoader* InResourceLoader);

	/** Access the active loader. Never null. */
	static IMetaHumanFileResourceLoader& GetResourceLoader();

private:
	FMetaHumanResourceLoaderHub() = default;
	FMetaHumanResourceLoaderHub(const FMetaHumanResourceLoaderHub&) = delete;
	FMetaHumanResourceLoaderHub& operator=(const FMetaHumanResourceLoaderHub&) = delete;

	static IMetaHumanFileResourceLoader& FallbackFileLoader();
	
	static IMetaHumanFileResourceLoader* ResourceLoader;
};

CARBON_NAMESPACE_END(TITAN_NAMESPACE)