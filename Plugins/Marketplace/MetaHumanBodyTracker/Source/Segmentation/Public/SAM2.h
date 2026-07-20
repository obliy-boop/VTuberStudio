// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Segmentation.h"

#include "Engine/DataTable.h"

class UNNEModelData;

namespace UE::NNE::Segmentation::SAM2
{

enum class EOverlapMode : uint8
{
	/** No overlap handling (SAM2 default). Each person's mask is independent. */
	None,

	/**
	 * Apply non-overlap constraints on mask logits BEFORE memory encoding.
	 * Each person's memory only contains their exclusive pixels.
	 * Most aggressive — may affect tracking quality.
	 * Matches SAM2 non_overlap_masks_for_mem_enc=true.
	 */
	BeforeMemory,

	/**
	 * Apply non-overlap constraints on final output masks only.
	 * Memory encoding uses raw overlapping logits — tracking unchanged.
	 * Least invasive. Matches SAM2 non_overlap_masks=true.
	 */
	OutputOnly,
};

enum class EResizeMethod : uint8
{
	/**
	 * PIL BICUBIC resize on uint8 input before float conversion.
	 * Matches the Python reference pipeline exactly (sam2/utils/misc.py).
	 * Required for byte-level validation against reference data.
	 */
	PILBicubic,

	/**
	 * UE native bicubic filter — faster but may diverge slightly from PIL BICUBIC
	 * due to kernel boundary handling differences. Acceptable for production use
	 * where exact Python parity is not required.
	 */
	UEBicubic,
};

/** Parameters for Make(). All fields have defaults matching the SAM2 reference config. */
struct FMakeParams
{
	/** How to handle overlapping masks in multi-person tracking.
	 *  None: each person's mask is independent (reference data default).
	 *  BeforeMemory: suppress overlap before memory encoding.
	 *  OutputOnly: suppress overlap on output only. */
	EOverlapMode OverlapMode = EOverlapMode::None;

	/** How the input image is resized during preprocessing.
	 *  PILBicubic: matches Python reference exactly (default, use for validation).
	 *  UEBicubic: faster UE native bicubic, may differ slightly at boundaries. */
	EResizeMethod ResizeMethod = EResizeMethod::PILBicubic;

	/** Logit threshold for converting decoder output to binary masks.
	 *  0.0 = sigmoid > 0.5 (SAM2 default). Lower values include more uncertain pixels
	 *  (larger masks); higher values require more confidence. */
	float MaskThreshold = 0.0f;

	/** Number of past frames kept in the spatial memory bank (1–7).
	 *  Higher values improve tracking through occlusions at the cost of more memory and
	 *  longer memory_attention inference. Default 7 matches the SAM2 training configuration
	 *  (TD-004). Cannot exceed 7 (limited by the maskmem_tpos_enc weight shape in the DataTable). */
	int32 NumMaskMem = 7;

	/** Loaded SAM2 image encoder model data. If nullptr, loaded from plugin content. */
	UNNEModelData* ImageEncoderData = nullptr;

	/** Loaded SAM2 mask decoder model data. If nullptr, loaded from plugin content. */
	UNNEModelData* MaskDecoderData = nullptr;

	/** Loaded SAM2 memory encoder model data. If nullptr, loaded from plugin content. */
	UNNEModelData* MemoryEncoderData = nullptr;

	/** Loaded SAM2 memory attention model data. If nullptr, loaded from plugin content. */
	UNNEModelData* MemoryAttentionData = nullptr;

	/** Loaded Weights table. If nullptr, loaded during Make */
	UDataTable* WeightsTable = nullptr;
};

/**
* Tries to load the SAM2 models and data tables that can be passed to the Make function
*/
SEGMENTATION_API bool TryLoadModelData(
	UNNEModelData*& OutImageEncoderData,
	UNNEModelData*& OutMaskDecoderData,
	UNNEModelData*& OutMemoryEncoderData,
	UNNEModelData*& OutMemoryAttentionData,
	UDataTable*& OutWeightsTable
);

/**
 * Creates a SAM2-based tracker (4-model NNE pipeline with memory bank).
 *
 * Loads the learned weights DataTable from /Segmentation/SAM2/sam2_weights.
 * Returns nullptr if any model or the weights asset cannot be loaded, or if no
 * runtime succeeds.
 *
 * @param RuntimePreferences  Prioritized list of (RuntimeName, DeviceType) pairs to try.
 *                            e.g. {{"NNERuntimeORTDml", GPU}, {"NNERuntimeORTCpu", CPU}}.
 * @param Params              Configuration parameters. All fields have sensible defaults.
 */
SEGMENTATION_API TUniquePtr<ITracker> Make(
	TConstArrayView<FRuntimePreference> RuntimePreferences,
	const FMakeParams& Params = {}
);

} // namespace UE::NNE::Segmentation::SAM2
