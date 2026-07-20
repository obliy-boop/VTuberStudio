// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "BodyTrackerModule.h"
#include "Containers/ContainersFwd.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeRDG.h"
#include "Pipeline/DataTreeTypes.h"
#include "RuntimePreference.h"


namespace UE::BodyTracker
{
	using UE::MetaHuman::Pipeline::FUEImageDataType;

	class BODYTRACKER_API FCameraCalibration
	{
	public:
		static TUniquePtr<FCameraCalibration> Make(UNNEModelData* GeoCalib, TConstArrayView<FRuntimePreference> RuntimePreferences);

		void Run(const UE::MetaHuman::Pipeline::FUEImageDataType& Frame);

		static constexpr bool CalculateFields = false;

		TArray64<float> PreprocessedImage;
		TArray64<double> TmpImage;
		TArray64<float> SmoothedImage;
		TArray64<float> UpField;
		TArray64<float> UpConfidence;
		TArray64<float> LatitudeField;
		TArray64<float> LatitudeConfidence;

		float Fov;
		float Pitch;
		float Roll;


	private:

		FCameraCalibration(TSharedRef<UE::NNE::IModelInstanceRunSync> GeoCalib) : GeoCalib{ GeoCalib }
		{
			Output.Init(0, 3);
		}

		TArray<float> Fovs;
		TArray<float> Pitches;
		TArray<float> Rolls;

		bool bShapeSet = false;
		TArray64<float> Output;
		TSharedRef<UE::NNE::IModelInstanceRunSync> GeoCalib;
	};
}
