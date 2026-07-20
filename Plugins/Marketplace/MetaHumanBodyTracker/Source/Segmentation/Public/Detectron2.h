// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Segmentation.h"


class UNNEModelData;

namespace UE::NNE::Segmentation::Detectron2
{

enum class EKeypointMethod : uint8
{
	Bicubic,
	Fast
};

enum class EResizeMethod : uint8
{
	PILBilinear,  // Matches Python PIL BILINEAR exactly
	UETriangle    // UE native Triangle filter — faster but may diverge from Python
};

/** Parameters for Make(). All fields have defaults matching the Detectron2 reference config. */
struct FMakeParams
{
	/** How keypoint heatmaps are converted to coordinates.
	 *  Bicubic: per-ROI bicubic upsample, matches Detectron2 exactly (~0.6px median diff).
	 *  Fast: argmax on 56x56 heatmap with analytic mapping, ~1-2px less precise but batchable. */
	EKeypointMethod KeypointMethod = EKeypointMethod::Fast;

	/** How the input image is resized during preprocessing.
	 *  PILBilinear: separable bilinear matching Python PIL exactly.
	 *  UETriangle: UE's built-in Triangle filter, faster but may diverge. */
	EResizeMethod ResizeMethod = EResizeMethod::PILBilinear;

	/** Minimum confidence score to keep a detection. Lower values return more detections
	 *  including uncertain ones; higher values keep only confident ones. */
	float BoxScoreThresh = 0.05f;

	/** IoU threshold for final non-maximum suppression. Detections overlapping above this
	 *  threshold are suppressed, keeping only the highest-scoring one.
	 *  Lower values suppress more aggressively (fewer overlapping boxes). */
	float BoxNmsThresh = 0.5f;

	/** Maximum number of detections returned per image after NMS. */
	int32 BoxTopKPerImage = 100;

	/** Number of top-scoring region proposals kept per FPN level before NMS.
	 *  Higher values consider more candidates but increase NMS cost. */
	int32 RpnPreNmsTopK = 1000;

	/** IoU threshold for region proposal NMS. Controls how aggressively duplicate proposals
	 *  are pruned before the box/keypoint heads. */
	float RpnNmsThresh = 0.7f;

	/** Total number of region proposals kept after NMS across all FPN levels.
	 *  These proposals are passed to the ROI box and keypoint heads. */
	int32 RpnPostNmsTopK = 1000;

	/** Loaded Detectron2 backbone model data. If nullptr, loaded from plugin content. */
	UNNEModelData* BackboneRpnData = nullptr;

	/** Loaded Detectron2 box head model data. If nullptr, loaded from plugin content. */
	UNNEModelData* RoiHeadsBoxData = nullptr;

	/** Loaded Detectron2 keypoint head model data. If nullptr, loaded from plugin content. */
	UNNEModelData* RoiHeadsKeypointData = nullptr;
};
/**
* Tries to load the Detectron2 models that can be passed to the Make function
*/
SEGMENTATION_API bool TryLoadModelData(
	UNNEModelData*& BackboneRpnData,
	UNNEModelData*& RoiHeadsBoxData,
	UNNEModelData*& RoiHeadsKeypointData
);

/**
 * Creates a Detectron2-based detector (Keypoint R-CNN).
 *
 * Returns nullptr if no runtime succeeds.
 *
 * @param RuntimePreferences	Prioritized list of (RuntimeName, DeviceType) pairs to try.
 *								e.g. {{"NNERuntimeORTDml", GPU}, {"NNERuntimeORTCpu", CPU}}.
 * @param Params				Configuration parameters. All fields have sensible defaults.
 */
SEGMENTATION_API TUniquePtr<IDetector> Make(
	TConstArrayView<FRuntimePreference> RuntimePreferences,
	const FMakeParams& Params = {}
);

} // namespace UE::NNE::Segmentation::Detectron2
