// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class MetaHuman : ModuleRules
{
	public MetaHuman(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"EnhancedInput",
			"AIModule",
			"StateTreeModule",
			"GameplayStateTreeModule",
			"UMG",
			"Slate"
		});

		PrivateDependencyModuleNames.AddRange(new string[] { });

		PublicIncludePaths.AddRange(new string[] {
			"MetaHuman",
			"MetaHuman/Variant_Platforming",
			"MetaHuman/Variant_Platforming/Animation",
			"MetaHuman/Variant_Combat",
			"MetaHuman/Variant_Combat/AI",
			"MetaHuman/Variant_Combat/Animation",
			"MetaHuman/Variant_Combat/Gameplay",
			"MetaHuman/Variant_Combat/Interfaces",
			"MetaHuman/Variant_Combat/UI",
			"MetaHuman/Variant_SideScrolling",
			"MetaHuman/Variant_SideScrolling/AI",
			"MetaHuman/Variant_SideScrolling/Gameplay",
			"MetaHuman/Variant_SideScrolling/Interfaces",
			"MetaHuman/Variant_SideScrolling/UI"
		});

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
	}
}
