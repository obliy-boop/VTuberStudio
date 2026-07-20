// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <cstddef>
#include <cstdint>
#include "carbon/common/Defs.h"

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

class IMetaHumanFileResourceLoader
{
public:
	virtual ~IMetaHumanFileResourceLoader() = default;

	/** Check if a path (UTF-8) exists. */
	virtual bool Exists(const char* InPathUtf8) const = 0;

	/** Read text file as raw UTF-8 bytes. Allocates a buffer that has to be freed */
	virtual bool ReadAllText(const char* InPathUtf8, char*& OutData, std::size_t& OutSize) const = 0;

	//* Read binary file as raw bytes. Allocates a buffer that has to be freed */
	virtual bool ReadAllBytes(const char* InPathUtf8, void*& OutData, std::size_t& OutSize) const = 0;

	/** Release any pointer previously returned via Read / Resolve functions */
	virtual void Free(const void* InPtr) const = 0;
	
	/** Constructs a valid path for data that could be loaded based on the resource loader type.
	 *  Allocates a buffer that has to be freed. Expects a null terminated base and appended input
	 */
	virtual bool ResolveDataLoadingPath(const char* InBasePathUtf8, const std::size_t& InBaseSize,
		const char* InAppendPathUtf8, std::size_t& InAppendSize, char*& OutPathUtf8, std::size_t& OutSize) const = 0;
};

CARBON_NAMESPACE_END(TITAN_NAMESPACE)