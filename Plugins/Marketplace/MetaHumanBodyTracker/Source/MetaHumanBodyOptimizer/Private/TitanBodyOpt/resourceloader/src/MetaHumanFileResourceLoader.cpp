// Copyright Epic Games, Inc. All Rights Reserved.

#include "resourceloader/MetaHumanFileResourceLoader.h"
#include "resourceloader/MetaHumanResourceLoaderHub.h"

#include "carbon/io/JsonIO.h"
#include <carbon/io/Utils.h>
#include "trio/streams/FileStream.h"
#include "trio/streams/MemoryStream.h"
#include <pma/resources/DefaultMemoryResource.h>

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <functional>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

bool FMetaHumanFileResourceLoader::Exists(const char* InPathUtf8) const
{
    return TITAN_NAMESPACE::FileExists(std::filesystem::path(InPathUtf8).string());
}

bool FMetaHumanFileResourceLoader::ReadAllText(const char* InPathUtf8, char*& OutData, std::size_t& OutSize) const
{
    // Initialize outputs
    OutData = nullptr;
    OutSize = 0;

    const std::string TextString = TITAN_NAMESPACE::ReadFile(InPathUtf8);

    // Treat empty file as success with empty output (common for text files)
    if (TextString.empty())
    {
        return true; // Success: empty file is valid
    }

    void* Buf = std::malloc(TextString.size());
    if (!Buf)
    {
        return false; // Allocation failure
    }

    std::memcpy(Buf, TextString.data(), TextString.size());
    OutData = static_cast<char*>(Buf);
    OutSize = TextString.size();

    return true;
}

bool FMetaHumanFileResourceLoader::ReadAllBytes(const char* InPathUtf8, void*& OutData, std::size_t& OutSize) const
{
    OutData = nullptr;
	OutSize = 0;
	const std::filesystem::path Path(InPathUtf8);
	
	std::ifstream Ifstream(Path, std::ios::binary | std::ios::ate);
	if (!Ifstream)
	{
		return false;
	}

	const std::streamoff End = Ifstream.tellg();

	if (End < 0 || static_cast<unsigned long long>(End) > std::numeric_limits<std::size_t>::max())
	{
		return false;
	}

	const std::size_t Size = static_cast<std::size_t>(End);
	Ifstream.seekg(0, std::ios::beg);

	void* Buf = std::malloc(Size);
	if (!Buf)
	{
		return false;
	}

	Ifstream.read(static_cast<char*>(Buf), static_cast<std::streamsize>(Size));
	if (!Ifstream || Ifstream.gcount() != static_cast<std::streamsize>(Size)) {
		std::free(Buf);
		return false;
	}

	OutData = Buf;
	OutSize = Size;
	return true;
}

void FMetaHumanFileResourceLoader::Free(const void* InPtr) const
{
    std::free(const_cast<void*>(InPtr));
}

bool FMetaHumanFileResourceLoader::ResolveDataLoadingPath(const char* InBasePathUtf8, const std::size_t& InBaseSize, const char* InAppendPathUtf8,
	std::size_t& InAppendSize, char*& OutPathUtf8, std::size_t& OutSize) const
{
	// TODO: Fix the logic for when either Base or Append paths are empty in case the other contains all the info needed
	if ((!InBasePathUtf8 && InBaseSize != 0) || (!InAppendPathUtf8 && InAppendSize != 0)) {
		return false;
	}

	std::filesystem::path Resolved;
	const std::filesystem::path BasePath(InBasePathUtf8);
	const std::filesystem::path AppendPath(InAppendPathUtf8);

	if (AppendPath.is_relative())
	{
		const std::string DataDescriptionDirectory = std::filesystem::absolute(BasePath).string();
		Resolved = DataDescriptionDirectory / AppendPath;
	}
	else
	{
		Resolved = AppendPath;
	}

	const std::u8string U8 = Resolved.u8string();
	const size_t payload = U8.size(); 
	OutPathUtf8 = static_cast<char*>(std::malloc(payload));
	if (!OutPathUtf8)
	{
		return false;
	}

	std::memcpy(OutPathUtf8, U8.data(), payload);

	OutSize = payload;
	return true;
}

TITAN_NAMESPACE::JsonElement FMetaHumanFileResourceLoader::GetJsonElementForFile(const std::string& InFilePath)
{
	const IMetaHumanFileResourceLoader& ResourceLoader = FMetaHumanResourceLoaderHub::GetResourceLoader();
	TITAN_NAMESPACE::JsonElement JSONElement;
	char* JSONData = nullptr;
	std::size_t JSONSize = 0;
	if (ResourceLoader.ReadAllText(InFilePath.c_str(), JSONData, JSONSize))
	{
		std::string JsonAsString = ToStdString(JSONData, JSONSize);
		JSONElement = TITAN_NAMESPACE::ReadJson(JsonAsString);
	}
	
	ResourceLoader.Free(JSONData);
	
	return JSONElement;
}

std::string FMetaHumanFileResourceLoader::ToStdString(const char* InData, std::size_t InSize)
{
	std::string OutString;
	if (InData && InSize > 0)
	{
		OutString.assign(InData, InData + InSize);
	}
	
	return OutString;
}

std::string FMetaHumanFileResourceLoader::GetResolvedPathForFile(const std::string& InBasePathUtf8, const std::string& InAppendPathUtf8)
{
	const IMetaHumanFileResourceLoader& ResourceLoader = FMetaHumanResourceLoaderHub::GetResourceLoader();
	
	const char* BasePtr = InBasePathUtf8.data();
	const std::size_t BaseSize = InBasePathUtf8.size();
	const char* AppendPtr = InAppendPathUtf8.data();
	std::size_t AppendSize = InAppendPathUtf8.size();
	char* ResultingString = nullptr;
	std::size_t ResultingSize = 0;

	ResourceLoader.ResolveDataLoadingPath(BasePtr, BaseSize, AppendPtr, AppendSize, ResultingString, ResultingSize);
	std::string Result = ToStdString(ResultingString, ResultingSize);
	ResourceLoader.Free(ResultingString);
	
	return  Result;
}

bool FMetaHumanFileResourceLoader::ResourceExists(const std::string& InResourcePathUtf8)
{
	const IMetaHumanFileResourceLoader& ResourceLoader = FMetaHumanResourceLoaderHub::GetResourceLoader();

	return ResourceLoader.Exists(InResourcePathUtf8.c_str());
}

namespace
{
void DestroyFile(::trio::BoundedIOStream* p)
{
    ::trio::FileStream::destroy(static_cast<::trio::FileStream*>(p));
}

void DestroyMemory(::trio::BoundedIOStream* p)
{
    ::trio::MemoryStream::destroy(static_cast<::trio::MemoryStream*>(p));
}
} // namespace

FMetaHumanFileResourceLoader::BoundedStreamPtr FMetaHumanFileResourceLoader::GetBoundedIOStreamFromFile(const std::string& InFilePath, bool bForceMemoryStream)
{
    if (TITAN_NAMESPACE::FileExists(std::filesystem::path(InFilePath).string()) && !bForceMemoryStream)
    {
        // Can use a file-based stream

        auto fileStream = ::trio::FileStream::create(InFilePath.c_str(), ::trio::AccessMode::Read, ::trio::OpenMode::Binary);
        return BoundedStreamPtr { fileStream, BoundedDestroy { &DestroyFile } };
    }
    else
    {
        const IMetaHumanFileResourceLoader& ResourceLoader = FMetaHumanResourceLoaderHub::GetResourceLoader();
		if (ResourceLoader.Exists(InFilePath.c_str()))
		{
			// No file access but can still get data
            void* DataBuffer = nullptr;
            std::size_t DataSize = 0;
            if (ResourceLoader.ReadAllBytes(InFilePath.c_str(), DataBuffer, DataSize))
            {
                auto MemoryStream = ::trio::MemoryStream::create(DataSize);
				if (!MemoryStream)
				{
                    ResourceLoader.Free(DataBuffer);
                    return nullptr;
				}

                MemoryStream->write(static_cast<const char*>(DataBuffer), DataSize);
                MemoryStream->seek(0);
                ResourceLoader.Free(DataBuffer);

				return BoundedStreamPtr { MemoryStream, BoundedDestroy { &DestroyMemory } };
            }
		}
    }

    return nullptr;
}

CARBON_NAMESPACE_END(TITAN_NAMESPACE)