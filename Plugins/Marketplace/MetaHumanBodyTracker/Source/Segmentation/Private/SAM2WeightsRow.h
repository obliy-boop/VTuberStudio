// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Engine/DataTable.h"
#include "SAM2WeightsRow.generated.h"

/**
 * DataTable row type for SAM2 learned weights.
 *
 * Each row holds one weight array (identified by row name).
 * The 13 expected rows match the keys in sam2_weights.npz:
 *   pe_gaussian_matrix, not_a_point_embed, point_embed_0..3,
 *   no_mask_embed, maskmem_tpos_enc, no_mem_embed,
 *   obj_ptr_tpos_proj_weight, obj_ptr_tpos_proj_bias,
 *   no_obj_ptr, no_obj_embed_spatial
 *
 * Import: drag sam2_weights.csv into Content Browser at /Segmentation/SAM2/.
 * CSV format: ---,ArrayData header; each row key_name,"(f0,f1,...)"
 */
USTRUCT()
struct FSAM2WeightRow : public FTableRowBase
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, Category = "Data")
	TArray<float> ArrayData;
};
