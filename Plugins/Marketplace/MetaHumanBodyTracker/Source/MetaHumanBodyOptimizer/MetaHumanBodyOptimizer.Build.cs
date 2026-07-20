// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class MetaHumanBodyOptimizer : ModuleRules
{
	public MetaHumanBodyOptimizer(ReadOnlyTargetRules Target) : base(Target)
	{
		IWYUSupport = IWYUSupport.None;
		CppCompileWarningSettings.UndefinedIdentifierWarningLevel = WarningLevel.Off;
		CppCompileWarningSettings.ShadowVariableWarningLevel = WarningLevel.Warning;
		bUseUnity = false;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Projects",
			"Json",
			"Eigen",
			"simde",
			"RigLogicLib",
		});

		if (Target.Type == TargetType.Editor)
		{
			PrivateDependencyModuleNames.AddRange(new string[]
			{
				"UnrealEd",
			});
		}

		string TitanBodyOpt = Path.Combine(ModuleDirectory, "Private/TitanBodyOpt");

		PrivateIncludePaths.AddRange(new string[]
		{
			TitanBodyOpt,
			Path.Combine(TitanBodyOpt, "carbon/include"),
			Path.Combine(TitanBodyOpt, "nls/include"),
			Path.Combine(TitanBodyOpt, "bodypostprocessing/include"),
			Path.Combine(TitanBodyOpt, "resourceloader/include"),
			Path.Combine(TitanBodyOpt, "api"),
		});

		PrivateDefinitions.Add("TITAN_DYNAMIC_API");
		PrivateDefinitions.Add("EIGEN_MPL2_ONLY");
		PrivateDefinitions.Add("TITAN_NAMESPACE=mhbt::epic::nls");
		PrivateDefinitions.Add("TITAN_API_NAMESPACE=mhbt::titan::api");

		if (Target.IsInPlatformGroup(UnrealPlatformGroup.Desktop))
		{
			PrivateDefinitions.Add("CARBON_ENABLE_SSE=1");
			PrivateDefinitions.Add("CARBON_ENABLE_AVX=0");
		}

		bEnableExceptions = true;
		bDisableAutoRTFMInstrumentation = true;

	}
}
