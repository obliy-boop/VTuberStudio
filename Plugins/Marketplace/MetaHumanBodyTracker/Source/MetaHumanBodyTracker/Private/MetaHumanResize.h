// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Containers/Map.h"
#include "Containers/UnrealString.h"
#include "Templates/TupleFwd.h"
#include "Math/MathFwd.h"
#include "Containers/Array.h"



namespace UE::MetaHuman::BodyTracker
{

class FMetaHumanResize
{
public:

	static TMap<FString, TPair<FString, FVector>> Resize(float InBeta0, const TArray<FVector>& InGlobalSMPLJointPosition);
};

}
