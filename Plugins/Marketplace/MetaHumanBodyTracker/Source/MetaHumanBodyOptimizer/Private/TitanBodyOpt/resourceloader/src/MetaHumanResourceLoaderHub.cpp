// Copyright Epic Games, Inc. All Rights Reserved.

#include "resourceloader/MetaHumanResourceLoaderHub.h"
#include "resourceloader/MetaHumanFileResourceLoader.h"

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

IMetaHumanFileResourceLoader* FMetaHumanResourceLoaderHub::ResourceLoader = nullptr;

void FMetaHumanResourceLoaderHub::Install(IMetaHumanFileResourceLoader* InResourceLoader)
{
	ResourceLoader = InResourceLoader;
}

IMetaHumanFileResourceLoader& FMetaHumanResourceLoaderHub::GetResourceLoader()
{
	return ResourceLoader ? *ResourceLoader : FallbackFileLoader();
}

IMetaHumanFileResourceLoader& FMetaHumanResourceLoaderHub::FallbackFileLoader()
{
	static FMetaHumanFileResourceLoader GFallback;
	return GFallback;
}

CARBON_NAMESPACE_END(TITAN_NAMESPACE)