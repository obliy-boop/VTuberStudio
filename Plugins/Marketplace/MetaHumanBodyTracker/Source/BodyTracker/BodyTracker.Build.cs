// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class BodyTracker : ModuleRules
{
	public BodyTracker(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "MetaHumanPipelineCore",
                "MetaHumanCoreTech",
                "Segmentation"
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "AssetRegistry",
                "Engine",
                "Json",
                "NNE",
                "RenderCore",
                "RHI",
                "ImageCore"
            }
        );
    }
}
