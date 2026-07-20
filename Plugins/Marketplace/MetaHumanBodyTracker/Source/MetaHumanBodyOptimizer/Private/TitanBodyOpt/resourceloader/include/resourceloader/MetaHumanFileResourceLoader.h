// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/common/Defs.h>

#include "IMetaHumanFileResourceLoader.h"
#include <string>
#include "pma/ScopedPtr.h"

namespace trio { class BoundedIOStream; }

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)
class JsonElement;

class FMetaHumanFileResourceLoader : public IMetaHumanFileResourceLoader
{
public:
	// IMetaHumanFileResourceLoader
	virtual bool Exists(const char* InPathUtf8) const override;
	virtual bool ReadAllText(const char* InPathUtf8, char*& OutData, std::size_t& OutSize) const override;
	virtual bool ReadAllBytes(const char* InPathUtf8, void*& OutData, std::size_t& OutSize) const override;
	virtual void Free(const void* InPtr) const override;
	virtual bool ResolveDataLoadingPath(const char* InBasePathUtf8, const std::size_t& InBaseSize,
		const char* InAppendPathUtf8, std::size_t& InAppendSize, char*& OutPathUtf8, std::size_t& OutSize) const override;

	/** Reads the json data based on static IResourceLoader implementation of ReadAllText override and returns json element */
    static TITAN_NAMESPACE::JsonElement GetJsonElementForFile(const std::string& InFilePath);

	/** Appends the paths based on the corresponding implementation of static IResourceLoader */
	static std::string GetResolvedPathForFile(const std::string& InBasePathUtf8, const std::string& InAppendPathUtf8);

	/** A convenience function that wraps around the overriden Exists function for static IResourceLoader */
	static bool ResourceExists(const std::string& InResourcePathUtf8);

	// Type-erased deleter that can call either FileStream::destroy or MemoryStream::destroy
    struct BoundedDestroy
    {
        using Fn = void (*)(trio::BoundedIOStream*);
        Fn fn { nullptr };

        void operator()(trio::BoundedIOStream* p)
        {
            if (p && fn)
                fn(p);
        }
    };

    // Unified smart pointer type that contains deleter
    using BoundedStreamPtr = pma::ScopedPtr<::trio::BoundedIOStream, BoundedDestroy>;

	/** A convenience function that returns File or Memory based IO stream for input binary file. Or nullptr if failed */
    static BoundedStreamPtr GetBoundedIOStreamFromFile(const std::string& InFilePath, bool bForceMemoryStream = false);

private:

	// Copies raw UTF-8 (InData/InSize) into std::string.
	static std::string ToStdString(const char* InData, std::size_t InSize);
};

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
