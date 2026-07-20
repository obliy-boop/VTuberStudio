// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

namespace UE::BodyTracker
{
	enum class EDeviceType : uint8
	{
		CPU,
		GPU,
		NPU
	};

	struct FRuntimePreference
	{
		FString RuntimeName;
		EDeviceType DeviceType;
	};
} // namespace UE::BodyTracker
