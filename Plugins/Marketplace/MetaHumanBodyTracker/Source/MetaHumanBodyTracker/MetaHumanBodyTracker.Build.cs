// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class MetaHumanBodyTracker : ModuleRules
{
	public MetaHumanBodyTracker(ReadOnlyTargetRules Target) : base(Target)
	{
		PublicDependencyModuleNames.AddRange(new string[]
		{
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"Json", 
			"Projects",
			"BodyTracker",
			"Segmentation",
			"ImageCore",
			"NNE",
			"MeshDescription",
			"SkeletalMeshDescription",
			"StaticMeshDescription",
			"GeometryCore",
			"GeometryFramework",
			"IKRig",
			"MetaHumanBodyTrackerInterface",
			"MetaHumanPipelineCore",
			"MetaHumanCoreTech",
			"MetaHumanBodyOptimizer",
			"MetaHumanPerformance",
		});

		if (Target.Type == TargetType.Editor)
		{
			PrivateDependencyModuleNames.AddRange(new string[]
			{
				"UnrealEd",
			});
		}
	}
}
