// Copyright Epic Games, Inc. All Rights Reserved.

#include "SAM2.h"

#include "NNEModelData.h"
#include "NNEUtils.h"
#include "ResampleUtils.h"
#include "SAM2WeightsRow.h"
#include "HAL/IConsoleManager.h"
#include "Async/ParallelFor.h"
#include "Containers/StaticArray.h"

DEFINE_LOG_CATEGORY_STATIC(LogSAM2, Log, All);

namespace UE::NNE::Segmentation::SAM2::Private
{

// ---------------------------------------------------------------------------
// Model and weight asset paths (plugin content)
// ---------------------------------------------------------------------------

static const TCHAR* ImageEncoderAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/SAM2/sam2_image_encoder");
static const TCHAR* MaskDecoderAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/SAM2/sam2_mask_decoder");
static const TCHAR* MemoryEncoderAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/SAM2/sam2_memory_encoder");
static const TCHAR* MemoryAttentionAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/SAM2/sam2_memory_attention");
static const TCHAR* WeightsAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/SAM2/sam2_weights");

// ---------------------------------------------------------------------------
// Constants (from sam2.1_hiera_t.yaml and sam2_base.py defaults)
// ---------------------------------------------------------------------------

static constexpr int32 ImageSize           = 1024;
static constexpr int32 HiddenDim           = 256;
static constexpr int32 MemDim              = 64;
static constexpr int32 NumObjPtrTokens     = HiddenDim / MemDim;  // 4
static constexpr int32 MaxObjPtrTokens     = 64;                  // 16 frames × 4 tokens
static constexpr int32 NumMaskMemMax       = 7;                   // upper bound — matches maskmem_tpos_enc shape (7,1,1,64)
static constexpr int32 MaxObjPtrsInEncoder = 16;
static constexpr int32 NumMaskTokens       = 4;
static constexpr float NoObjScore          = -1024.0f;
static constexpr float SigmoidScale        = 20.0f;  // sigmoid_scale_for_mem_enc
static constexpr float SigmoidBias         = -10.0f; // sigmoid_bias_for_mem_enc

// ImageNet RGB normalization
static constexpr float PixelMeanRGB[3] = { 0.485f, 0.456f, 0.406f };
static constexpr float PixelStdRGB[3]  = { 0.229f, 0.224f, 0.225f };

// Expected element counts per weight array
static constexpr int32 WeightCounts[] = {
	256,    // pe_gaussian_matrix     (2,128)
	256,    // not_a_point_embed      (256,)
	256,    // point_embed_0          (256,)
	256,    // point_embed_1          (256,)
	256,    // point_embed_2          (256,)
	256,    // point_embed_3          (256,)
	256,    // no_mask_embed          (256,)
	448,    // maskmem_tpos_enc       (7,1,1,64)
	256,    // no_mem_embed           (1,1,256)
	16384,  // obj_ptr_tpos_proj_weight (64,256)
	64,     // obj_ptr_tpos_proj_bias  (64,)
	256,    // no_obj_ptr             (1,256)
	64,     // no_obj_embed_spatial   (1,64)
};

static const TCHAR* WeightKeys[] = {
	TEXT("pe_gaussian_matrix"),
	TEXT("not_a_point_embed"),
	TEXT("point_embed_0"),
	TEXT("point_embed_1"),
	TEXT("point_embed_2"),
	TEXT("point_embed_3"),
	TEXT("no_mask_embed"),
	TEXT("maskmem_tpos_enc"),
	TEXT("no_mem_embed"),
	TEXT("obj_ptr_tpos_proj_weight"),
	TEXT("obj_ptr_tpos_proj_bias"),
	TEXT("no_obj_ptr"),
	TEXT("no_obj_embed_spatial"),
};
static constexpr int32 NumWeightArrays = UE_ARRAY_COUNT(WeightKeys);
static_assert(UE_ARRAY_COUNT(WeightCounts) == NumWeightArrays, "WeightCounts/WeightKeys mismatch");

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/**
 * Plain-value copy of all 13 SAM2 learned weight arrays.
 * Copied from the DataTable at Make() time so inference is independent of UObject lifetime.
 */
struct FSAM2Weights
{
	TArray<float> PeGaussianMatrix;       // (2,128)     — random Fourier PE matrix
	TArray<float> NotAPointEmbed;         // (256,)      — padding token embedding
	TArray<float> PointEmbed0;            // (256,)      — background label embedding
	TArray<float> PointEmbed1;            // (256,)      — foreground label embedding
	TArray<float> PointEmbed2;            // (256,)      — box top-left corner embedding
	TArray<float> PointEmbed3;            // (256,)      — box bottom-right corner embedding
	TArray<float> NoMaskEmbed;            // (256,)      — dense embedding when no mask input
	TArray<float> MaskmemTposEnc;         // (7,1,1,64)  — temporal PE for 7 memory slots
	TArray<float> NoMemEmbed;             // (1,1,256)   — added to fpn2 on frame 0 (no memory)
	TArray<float> ObjPtrTposProjWeight;   // (64,256)    — project obj_ptr PE to MEM_DIM
	TArray<float> ObjPtrTposProjBias;     // (64,)       — projection bias
	TArray<float> NoObjPtr;              // (1,256)     — replacement pointer when not appearing
	TArray<float> NoObjEmbedSpatial;     // (1,64)      — spatial embed for non-appearing objects

	/**
	 * Loads all 13 weight arrays from the given DataTable.
	 * Returns false and logs an error if any row is missing or has unexpected element count.
	 */
	static bool FromDataTable(const UDataTable* Table, FSAM2Weights& OutWeights);
};

/** Per-person tracking state maintained across frames. */
struct FPersonMemoryState
{
	// Conditioning memory from frame 0 (always kept)
	TArray<float> CondMaskmemFeatures;             // (1,64,64,64)
	TArray<float> CondMaskmemPosEnc;               // (1,64,64,64)

	// Recent tracking memories, newest first (max NumMaskMem-1 = 6)
	TArray<TArray<float>> RecentFeatures;           // each (1,64,64,64)
	TArray<TArray<float>> RecentPosEncs;            // each (1,64,64,64)

	// Object pointer history with originating frame index (max MaxObjPtrsInEncoder = 16)
	TArray<int32>         ObjPtrFrameIndices;
	TArray<TArray<float>> ObjPtrs;                  // each (HiddenDim=256,)
};

/** Sparse + dense embeddings produced by the prompt encoder. */
struct FPromptEmbeddings
{
	TArray<float> Sparse;    // (1, NumTokens, HiddenDim)
	TArray<float> Dense;     // (1, HiddenDim, 64, 64)
	int32 NumTokens = 0;
};

/** Selected mask, IoU, and object pointer from the mask decoder. */
struct FMaskSelection
{
	TArray<float> Mask;      // (1, 1, 256, 256) — selected mask logits
	float Iou = 0.0f;
	TArray<float> ObjPtr;    // (HiddenDim=256,)
};

/** Assembled memory bank ready for memory_attention ONNX input. */
struct FMemoryBankResult
{
	TArray<float> Memory;     // (M, 1, MemDim)
	TArray<float> MemoryPos;  // (M, 1, MemDim)
	int32 M = 0;
};

struct FPerPersonData
{
	TArray<float> BestMask;       // (256*256,) logits
	TArray<float> GatedObjPtr;    // (256,)
	float         ObjScoreLogit;
	float         BestIou;        // IoU prediction for selected mask (matches reference "score")
};

// ---------------------------------------------------------------------------
// FSAM2Weights — DataTable loading
// ---------------------------------------------------------------------------

bool FSAM2Weights::FromDataTable(const UDataTable* Table, FSAM2Weights& OutWeights)
{
	TArray<float>* Members[] = {
		&OutWeights.PeGaussianMatrix,
		&OutWeights.NotAPointEmbed,
		&OutWeights.PointEmbed0,
		&OutWeights.PointEmbed1,
		&OutWeights.PointEmbed2,
		&OutWeights.PointEmbed3,
		&OutWeights.NoMaskEmbed,
		&OutWeights.MaskmemTposEnc,
		&OutWeights.NoMemEmbed,
		&OutWeights.ObjPtrTposProjWeight,
		&OutWeights.ObjPtrTposProjBias,
		&OutWeights.NoObjPtr,
		&OutWeights.NoObjEmbedSpatial,
	};
	static_assert(UE_ARRAY_COUNT(Members) == NumWeightArrays, "Members/WeightKeys mismatch");

	for (int32 I = 0; I < NumWeightArrays; ++I)
	{
		const FSAM2WeightRow* Row = Table->FindRow<FSAM2WeightRow>(
			FName(WeightKeys[I]), TEXT("FSAM2Weights::FromDataTable"));
		if (!Row)
		{
			UE_LOG(LogSAM2, Error, TEXT("SAM2 weights DataTable missing row '%s'"), WeightKeys[I]);
			return false;
		}
		if (Row->ArrayData.Num() != WeightCounts[I])
		{
			UE_LOG(LogSAM2, Error, TEXT("SAM2 weight '%s': expected %d elements, got %d"),
				WeightKeys[I], WeightCounts[I], Row->ArrayData.Num());
			return false;
		}
		*Members[I] = Row->ArrayData;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Stage 1: Preprocessing
// ---------------------------------------------------------------------------

/**
 * Preprocesses an RGB image for sam2_image_encoder.
 *
 * Matches SAM2's exact pipeline: PIL BICUBIC resize uint8 → 1024×1024,
 * convert to float32 / 255.0, ImageNet normalize, HWC → CHW, add batch dim.
 *
 * @param FrameRgb  Input image (any format; read as RGB).
 * @param OutTensor Output (1, 3, 1024, 1024) float32 tensor, row-major NCHW.
 * @param OutH      Original image height (for later mask postprocessing).
 * @param OutW      Original image width.
 */
static void PreprocessImage(
	TArray<float>& OutTensor,
	const FImage& FrameRgb,
	int32& OutH, int32& OutW,
	EResizeMethod ResizeMethod)
{
	SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::Preprocess", FColor::Yellow);

	OutH = FrameRgb.GetHeight();
	OutW = FrameRgb.GetWidth();

	// Convert to BGRA8 if not already (ResizePILBicubic operates on BGRA8)
	FImage Bgra8Frame;
	if (FrameRgb.Format != ERawImageFormat::BGRA8)
	{
		FrameRgb.CopyTo(Bgra8Frame, ERawImageFormat::BGRA8, EGammaSpace::Linear);
	}
	const FImage& Src = (FrameRgb.Format == ERawImageFormat::BGRA8) ? FrameRgb : Bgra8Frame;

	// Resize BGRA8 to 1024×1024
	FImage ResizedImage;
	ResizedImage.Init(ImageSize, ImageSize, ERawImageFormat::BGRA8, EGammaSpace::Linear);
	if (ResizeMethod == EResizeMethod::PILBicubic)
	{
		ResizePILBicubic(
			Src.RawData.GetData(), OutW, OutH,
			ResizedImage.RawData.GetData(), ImageSize, ImageSize);
	}
	else
	{
		FImageCore::ResizeImageAllocDest(Src, ResizedImage, ImageSize, ImageSize,
			ERawImageFormat::BGRA8, EGammaSpace::Linear,
			FImageCore::EResizeImageFilter::CubicSharp);
	}

	// Normalize and transpose HWC→CHW → (1, 3, 1024, 1024)
	// BGRA8 layout: [B, G, R, A, ...]. SAM2 uses RGB normalization.
	//   R = Src[2], G = Src[1], B = Src[0]
	//   Output channel 0 = R (PixelMeanRGB[0]=0.485, PixelStdRGB[0]=0.229)
	//   Output channel 1 = G (PixelMeanRGB[1]=0.456, PixelStdRGB[1]=0.224)
	//   Output channel 2 = B (PixelMeanRGB[2]=0.406, PixelStdRGB[2]=0.225)
	const int32 ChannelStride = ImageSize * ImageSize;
	OutTensor.SetNumUninitialized(3 * ChannelStride);
	float* DstR = OutTensor.GetData();
	float* DstG = OutTensor.GetData() + ChannelStride;
	float* DstB = OutTensor.GetData() + 2 * ChannelStride;

	const uint8* Pixels = ResizedImage.RawData.GetData();
	for (int32 Idx = 0; Idx < ChannelStride; ++Idx)
	{
		const uint8* Px = Pixels + Idx * 4;
		DstR[Idx] = (static_cast<float>(Px[2]) / 255.0f - PixelMeanRGB[0]) / PixelStdRGB[0];
		DstG[Idx] = (static_cast<float>(Px[1]) / 255.0f - PixelMeanRGB[1]) / PixelStdRGB[1];
		DstB[Idx] = (static_cast<float>(Px[0]) / 255.0f - PixelMeanRGB[2]) / PixelStdRGB[2];
	}
}

// ---------------------------------------------------------------------------
// Stage 2: Prompt Encoding
// ---------------------------------------------------------------------------

/**
 * Core Fourier positional encoding for N 2D coordinates already in [-1, 1].
 *
 * Port of sam2 PositionEmbeddingRandom._pe_encoding.
 *
 * @param Coords2D          Flat array of N×2 (x,y) coordinates in [-1, 1].
 * @param N                 Number of coordinate pairs.
 * @param PeGaussianMatrix  (2, 128) = 256 floats, row-major. Learned random matrix.
 * @returns                 (N, 256) float32 positional encoding.
 */
static TArray<float> FourierPeEncoding(
	TConstArrayView<float> Coords2D,
	int32 N,
	TConstArrayView<float> PeGaussianMatrix)
{
	// projected[n, j] = sum_k( Coords2D[n*2+k] * PeGaussianMatrix[k*128+j] ), j in 0..127
	// then *2π, then [sin, cos] concat → (N, 256)
	constexpr int32 PeDim = 128; // HiddenDim / 2
	TArray<float> Result;
	Result.SetNumUninitialized(N * HiddenDim);

	for (int32 Ni = 0; Ni < N; ++Ni)
	{
		const float Cx = Coords2D[Ni * 2 + 0];
		const float Cy = Coords2D[Ni * 2 + 1];

		for (int32 J = 0; J < PeDim; ++J)
		{
			// Matrix is (2, 128) row-major: row 0 = x weights, row 1 = y weights
			const float Proj = Cx * PeGaussianMatrix[0 * PeDim + J]
			                 + Cy * PeGaussianMatrix[1 * PeDim + J];
			const float Scaled = Proj * (2.0f * UE_PI);
			Result[Ni * HiddenDim + J]          = FMath::Sin(Scaled);  // sin half
			Result[Ni * HiddenDim + PeDim + J]  = FMath::Cos(Scaled);  // cos half
		}
	}
	return Result;
}

/**
 * Computes the dense positional encoding for the mask decoder feature grid.
 *
 * Port of sam2 PositionEmbeddingRandom.forward + get_dense_pe.
 * This is the image_pe input to sam2_mask_decoder — NOT the FPN pos_enc.
 * Result is constant for all frames; call once and cache.
 *
 * @param Weights  SAM2 weights (uses PeGaussianMatrix).
 * @param Size     Feature grid size (default 64 for 64×64).
 * @returns        (1, 256, Size, Size) float32.
 */
static TArray<float> GetDensePe(const FSAM2Weights& Weights, int32 Size = 64)
{
	// Regular grid (Size×Size, 2): coords[h*Size+w, {x,y}]
	// x = grid_x[w] = (w + 0.5) / Size → scaled to [-1,1]: 2*x-1
	// y = grid_y[h] = (h + 0.5) / Size → scaled to [-1,1]: 2*y-1
	// Stack: [x_val, y_val] per cell — matches np.meshgrid(grid_x, grid_y) + stack([xx,yy])
	const int32 NumCells = Size * Size;
	TArray<float> Coords;
	Coords.SetNumUninitialized(NumCells * 2);

	for (int32 H = 0; H < Size; ++H)
	{
		const float Y = 2.0f * ((H + 0.5f) / Size) - 1.0f;
		for (int32 W = 0; W < Size; ++W)
		{
			const float X = 2.0f * ((W + 0.5f) / Size) - 1.0f;
			Coords[(H * Size + W) * 2 + 0] = X;
			Coords[(H * Size + W) * 2 + 1] = Y;
		}
	}

	// FourierPeEncoding → (NumCells, 256) in HW-major order
	TArray<float> Pe = FourierPeEncoding(Coords, NumCells, Weights.PeGaussianMatrix);

	// Transpose (Size, Size, 256) → (1, 256, Size, Size)
	// pe[h*Size+w, c] → out[0, c, h, w] = out[c*Size*Size + h*Size + w]
	TArray<float> Out;
	Out.SetNumUninitialized(HiddenDim * NumCells);
	for (int32 HW = 0; HW < NumCells; ++HW)
	{
		for (int32 C = 0; C < HiddenDim; ++C)
		{
			Out[C * NumCells + HW] = Pe[HW * HiddenDim + C];
		}
	}
	return Out;  // (256, 64*64) = (256, 4096) — caller prepends batch dim conceptually
}

/**
 * Fourier positional encoding for coordinates in [0, 1024] space.
 *
 * Convenience wrapper: normalizes [0, 1024] → [0, 1] → [-1, 1],
 * then calls FourierPeEncoding.
 *
 * @param Coords2D  Flat N×2 (x,y) coordinates in [0, ImageSize=1024].
 * @param N         Number of coordinate pairs.
 * @returns         (N, 256) float32.
 */
static TArray<float> FourierEncodeCoords(
	TConstArrayView<float> Coords2D,
	int32 N,
	const FSAM2Weights& Weights)
{
	// Normalize [0, 1024] → [0, 1] → [-1, 1], then FourierPeEncoding
	TArray<float> Normalized;
	Normalized.SetNumUninitialized(N * 2);
	const float InvImageSize = 1.0f / static_cast<float>(ImageSize);
	for (int32 I = 0; I < N * 2; ++I)
	{
		Normalized[I] = 2.0f * (Coords2D[I] * InvImageSize) - 1.0f;
	}
	return FourierPeEncoding(Normalized, N, Weights.PeGaussianMatrix);
}

/**
 * Normalizes a box from original image coordinates to SAM2 internal space [0, 1024].
 *
 * Reference: sam2_video_predictor.py:215-217.
 *   coord_x = coord_x / video_W * image_size
 *   coord_y = coord_y / video_H * image_size
 *
 * @param Box4    (x1, y1, x2, y2) in original pixel coordinates.
 * @param VideoH  Original image height.
 * @param VideoW  Original image width.
 * @returns       (4,) box in [0, 1024] space.
 */
static FBox2f NormalizeBoxCoords(
	const FIntRect& InBox,
	int32 VideoH,
	int32 VideoW)
{
	FVector2f Scale{ static_cast<float>(ImageSize) / VideoW, static_cast<float>(ImageSize) / VideoH };
	FBox2f Result{ FVector2f(InBox.Min) * Scale, FVector2f(InBox.Max) * Scale };
	return Result;
}

/**
 * Encodes box and/or keypoint prompts for sam2_mask_decoder.
 *
 * Port of sam2 PromptEncoder + add_new_points_or_box concatenation logic.
 * Box corners (labels 2,3) are prepended before user points.
 * Always appends one not_a_point_embed padding token.
 *
 * Reference: sam2_video_predictor.py:209.
 *
 * @param VideoH, VideoW  Original image dimensions.
 * @param Weights         SAM2 weights.
 * @param Box4            (x1, y1, x2, y2) in original pixel space. May be empty.
 * @param Pts2D           Flat N×2 (x,y) in original pixel space. May be empty.
 * @param NumPoints       Number of point prompts.
 * @param Labels          Per-point label (1=foreground, 0=background). May be empty → all 1.
 * @returns               Sparse (1, T, 256) and Dense (1, 256, 64, 64) embeddings.
 *                        T = (2 if box) + NumPoints + 1 padding.
 *                        For box + 17 keypoints: T = 2 + 17 + 1 = 20.
 */
static FPromptEmbeddings EncodePrompt(
	int32 VideoH,
	int32 VideoW,
	const FSAM2Weights& Weights,
	const FIntRect& Box4,
	TConstArrayView<float> Pts2D,
	int32 NumPoints,
	TConstArrayView<int32> Labels)
{
	const bool bHasBox    = !Box4.IsEmpty();
	const bool bHasPoints = NumPoints > 0;

	// Count sparse tokens: 2 (box) + NumPoints + 1 (padding)
	const int32 BoxTokens = bHasBox ? 2 : 0;
	const int32 NumTokens = BoxTokens + NumPoints + 1;

	FPromptEmbeddings Result;
	Result.NumTokens = NumTokens;
	Result.Sparse.SetNumZeroed(NumTokens * HiddenDim);

	int32 WriteIdx = 0;

	// --- Box corners (labels 2, 3) ---
	if (bHasBox)
	{
		FBox2f NormBox = NormalizeBoxCoords(Box4, VideoH, VideoW);

		// +0.5 for SAM pixel center convention, then treat as 2 corner points
		NormBox.Min += FVector2f(0.5);
		NormBox.Max += FVector2f(0.5);

		TArray<float, TInlineAllocator<4>> Corners{ NormBox.Min.X, NormBox.Min.Y, NormBox.Max.X, NormBox.Max.Y };
		TArray<float> CornerPe = FourierEncodeCoords(Corners, 2, Weights);
		for (int32 D = 0; D < HiddenDim; ++D)
		{
			Result.Sparse[0 * HiddenDim + D] = CornerPe[0 * HiddenDim + D] + Weights.PointEmbed2[D];
			Result.Sparse[1 * HiddenDim + D] = CornerPe[1 * HiddenDim + D] + Weights.PointEmbed3[D];
		}
		WriteIdx = 2;
	}

	// --- User points ---
	if (bHasPoints)
	{
		const float ScaleX = static_cast<float>(ImageSize) / VideoW;
		const float ScaleY = static_cast<float>(ImageSize) / VideoH;

		TArray<float> PtsNorm;
		PtsNorm.SetNumUninitialized(NumPoints * 2);
		for (int32 P = 0; P < NumPoints; ++P)
		{
			PtsNorm[P * 2 + 0] = Pts2D[P * 2 + 0] * ScaleX + 0.5f;
			PtsNorm[P * 2 + 1] = Pts2D[P * 2 + 1] * ScaleY + 0.5f;
		}
		TArray<float> PtsPe = FourierEncodeCoords(PtsNorm, NumPoints, Weights);

		// Label-specific embed arrays: 0→PointEmbed0, 1→PointEmbed1 (default)
		const TArray<float>* LabelEmbeds[] = {
			&Weights.PointEmbed0,
			&Weights.PointEmbed1,
			&Weights.PointEmbed2,
			&Weights.PointEmbed3,
		};
		for (int32 P = 0; P < NumPoints; ++P)
		{
			const int32 Label     = Labels.IsEmpty() ? 1 : Labels[P];
			const int32 EmbedIdx  = FMath::Clamp(Label, 0, 3);
			const TArray<float>& Embed = *LabelEmbeds[EmbedIdx];
			float* Out = Result.Sparse.GetData() + (WriteIdx + P) * HiddenDim;
			for (int32 D = 0; D < HiddenDim; ++D)
			{
				Out[D] = PtsPe[P * HiddenDim + D] + Embed[D];
			}
		}
		WriteIdx += NumPoints;
	}

	// --- Padding token (not_a_point_embed, no PE) ---
	FMemory::Memcpy(
		Result.Sparse.GetData() + WriteIdx * HiddenDim,
		Weights.NotAPointEmbed.GetData(),
		HiddenDim * sizeof(float));

	// --- Dense: broadcast no_mask_embed (256,) → (1, 256, 64, 64) ---
	constexpr int32 FeatureSpatial = 64 * 64;
	Result.Dense.SetNumUninitialized(HiddenDim * FeatureSpatial);
	for (int32 C = 0; C < HiddenDim; ++C)
	{
		const float Val = Weights.NoMaskEmbed[C];
		for (int32 S = 0; S < FeatureSpatial; ++S)
		{
			Result.Dense[C * FeatureSpatial + S] = Val;
		}
	}

	return Result;
}

/**
 * Encodes the no-prompt embedding for tracking frames (frames 1+).
 *
 * On tracking frames SAM2 creates a dummy padding point (label -1) plus the
 * standard padding appended by _embed_points(pad=True). Both slots are filled
 * with not_a_point_embed.
 *
 * Reference: sam2_base.py:316-318, prompt_encoder.py:86-100.
 *
 * @returns  Sparse (1, 2, 256) and Dense (1, 256, 64, 64) embeddings.
 */
static FPromptEmbeddings EncodeTrackingPrompt(const FSAM2Weights& Weights)
{
	// Two not_a_point_embed tokens (dummy point + padding), no Fourier PE.
	FPromptEmbeddings Result;
	Result.NumTokens = 2;
	Result.Sparse.SetNumUninitialized(2 * HiddenDim);
	FMemory::Memcpy(Result.Sparse.GetData(),                   Weights.NotAPointEmbed.GetData(), HiddenDim * sizeof(float));
	FMemory::Memcpy(Result.Sparse.GetData() + HiddenDim,       Weights.NotAPointEmbed.GetData(), HiddenDim * sizeof(float));

	constexpr int32 FeatureSpatial = 64 * 64;
	Result.Dense.SetNumUninitialized(HiddenDim * FeatureSpatial);
	for (int32 C = 0; C < HiddenDim; ++C)
	{
		const float Val = Weights.NoMaskEmbed[C];
		for (int32 S = 0; S < FeatureSpatial; ++S)
		{
			Result.Dense[C * FeatureSpatial + S] = Val;
		}
	}

	return Result;
}

// ---------------------------------------------------------------------------
// Stage 3: Mask Selection + Postprocessing
// ---------------------------------------------------------------------------

/**
 * Selects the best mask, IoU, and object pointer from mask decoder outputs.
 *
 * Matches sam2_base.py:380-403 singlemask/multimask selection logic.
 * Also applies NO_OBJ_SCORE masking: when object_score_logits <= 0, all mask
 * logits are replaced with NoObjScore before selection (sam2_base.py:359-366).
 *
 * @param AllMasks           (1, 4, 256, 256) — all 4 mask candidates (logits).
 * @param AllIous            (1, 4) — IoU prediction per mask token.
 * @param AllObjPtrs         (1, 4, 256) — object pointer per mask token.
 * @param ObjectScoreLogit   Scalar — object presence logit.
 * @param bMultimaskOutput   true for tracking frames (best of tokens 1-3 by IoU),
 *                           false for frame 0 with box prompt (token 0).
 */
static FMaskSelection SelectMaskAndObjPtr(
	TConstArrayView<float> AllMasks,
	TConstArrayView<float> AllIous,
	TConstArrayView<float> AllObjPtrs,
	float ObjectScoreLogit,
	bool bMultimaskOutput)
{
	// (1, 4, 256, 256) logits — 4 tokens × 256×256 pixels
	constexpr int32 MaskPixels = 256 * 256;

	int32 BestIdx = 0;
	if (bMultimaskOutput)
	{
		// Pick best of tokens 1, 2, 3 by IoU prediction
		BestIdx = 1;
		for (int32 I = 2; I <= 3; ++I)
		{
			if (AllIous[I] > AllIous[BestIdx])
			{
				BestIdx = I;
			}
		}
	}

	FMaskSelection Sel;
	Sel.Iou = AllIous[BestIdx];

	// Extract selected mask (1, 1, 256, 256)
	Sel.Mask.SetNumUninitialized(MaskPixels);
	const float* SrcMask = AllMasks.GetData() + BestIdx * MaskPixels;
	if (ObjectScoreLogit <= 0.0f)
	{
		// Object not appearing: fill with NoObjScore
		for (int32 P = 0; P < MaskPixels; ++P) Sel.Mask[P] = NoObjScore;
	}
	else
	{
		FMemory::Memcpy(Sel.Mask.GetData(), SrcMask, MaskPixels * sizeof(float));
	}

	// Extract selected obj_ptr (256,) from (1, 4, 256)
	Sel.ObjPtr.SetNumUninitialized(HiddenDim);
	FMemory::Memcpy(
		Sel.ObjPtr.GetData(),
		AllObjPtrs.GetData() + BestIdx * HiddenDim,
		HiddenDim * sizeof(float));

	return Sel;
}


// Bilinear upsample of a single-channel float image (align_corners=False, matching PyTorch/PIL).
static void BilinearUpsampleFloat(
	TConstArrayView<float> Src, int32 SrcH, int32 SrcW,
	TArray<float>& Dst, int32 DstH, int32 DstW)
{
	Dst.SetNumUninitialized(DstH * DstW);
	const float ScaleY = static_cast<float>(SrcH) / DstH;
	const float ScaleX = static_cast<float>(SrcW) / DstW;

#if 1
	ParallelFor(DstH, [&](int DY)
		{
			const float SrcYF = (DY + 0.5f) * ScaleY - 0.5f;
			const int32 Y0 = FMath::Clamp(FMath::FloorToInt32(SrcYF), 0, SrcH - 1);
			const int32 Y1 = FMath::Clamp(Y0 + 1, 0, SrcH - 1);
			const float WY = FMath::Clamp(SrcYF - static_cast<float>(FMath::FloorToInt32(SrcYF)), 0.0f, 1.0f);

			for (int32 DX = 0; DX < DstW; ++DX)
			{
				const float SrcXF = (DX + 0.5f) * ScaleX - 0.5f;
				const int32 X0 = FMath::Clamp(FMath::FloorToInt32(SrcXF), 0, SrcW - 1);
				const int32 X1 = FMath::Clamp(X0 + 1, 0, SrcW - 1);
				const float WX = FMath::Clamp(SrcXF - static_cast<float>(FMath::FloorToInt32(SrcXF)), 0.0f, 1.0f);

				const float V00 = Src[Y0 * SrcW + X0];
				const float V01 = Src[Y0 * SrcW + X1];
				const float V10 = Src[Y1 * SrcW + X0];
				const float V11 = Src[Y1 * SrcW + X1];
				Dst[DY * DstW + DX] =
					V00 * (1.0f - WY) * (1.0f - WX) +
					V01 * (1.0f - WY) * WX +
					V10 * WY * (1.0f - WX) +
					V11 * WY * WX;
			}
		}, EParallelForFlags::None);
#else
	for (int32 DY = 0; DY < DstH; ++DY)
	{
		const float SrcYF = (DY + 0.5f) * ScaleY - 0.5f;
		const int32 Y0 = FMath::Clamp(FMath::FloorToInt32(SrcYF), 0, SrcH - 1);
		const int32 Y1 = FMath::Clamp(Y0 + 1, 0, SrcH - 1);
		const float WY = FMath::Clamp(SrcYF - static_cast<float>(FMath::FloorToInt32(SrcYF)), 0.0f, 1.0f);

		for (int32 DX = 0; DX < DstW; ++DX)
		{
			const float SrcXF = (DX + 0.5f) * ScaleX - 0.5f;
			const int32 X0 = FMath::Clamp(FMath::FloorToInt32(SrcXF), 0, SrcW - 1);
			const int32 X1 = FMath::Clamp(X0 + 1, 0, SrcW - 1);
			const float WX = FMath::Clamp(SrcXF - static_cast<float>(FMath::FloorToInt32(SrcXF)), 0.0f, 1.0f);

			const float V00 = Src[Y0 * SrcW + X0];
			const float V01 = Src[Y0 * SrcW + X1];
			const float V10 = Src[Y1 * SrcW + X0];
			const float V11 = Src[Y1 * SrcW + X1];
			Dst[DY * DstW + DX] =
				V00 * (1.0f - WY) * (1.0f - WX) +
				V01 * (1.0f - WY) * WX +
				V10 * WY * (1.0f - WX) +
				V11 * WY * WX;
		}
	}
#endif
}

/**
 * Removes stray pixels from a binary mask using 3×3 neighborhood filtering.
 *
 * Keeps a pixel only if it has at least MinNeighbors True neighbors in a 3×3
 * window (including itself). Removes single-pixel artifacts without shrinking
 * the main mask body.
 *
 * Port of sam2_processing.py clean_mask.
 *
 * @param Mask         Flat (H×W) boolean mask.
 * @param H, W         Mask dimensions.
 * @param MinNeighbors Minimum neighbors required to keep a pixel (default 4).
 *                     Actual threshold: count >= MinNeighbors + 1 (including self).
 * @returns            Flat (H×W) cleaned boolean mask.
 */
static void CleanMask(
	TConstArrayView<bool> InMask,
	TArray<bool>& OutMask,
	int32 H, int32 W,
	int32 MinNeighbors = 4)
{
	const int32 Threshold = MinNeighbors + 1;  // include self in 3×3 count
	OutMask.SetNumUninitialized(H * W);

#if 1
	ParallelFor(H, [&](int Y)
		{
			for (int32 X = 0; X < W; ++X)
			{
				if (!InMask[Y * W + X])
				{
					OutMask[Y * W + X] = false;
					continue;
				}
				int32 Count = 0;
				for (int32 DY = -1; DY <= 1; ++DY)
				{
					const int32 NY = Y + DY;
					if (NY < 0 || NY >= H) continue;
					for (int32 DX = -1; DX <= 1; ++DX)
					{
						const int32 NX = X + DX;
						if (NX < 0 || NX >= W) continue;
						if (InMask[NY * W + NX]) ++Count;
					}
				}
				OutMask[Y * W + X] = (Count >= Threshold);
			}
		}, EParallelForFlags::None);
#else
	for (int32 Y = 0; Y < H; ++Y)
	{
		for (int32 X = 0; X < W; ++X)
		{
			if (!InMask[Y * W + X])
			{
				OutMask[Y * W + X] = false;
				continue;
			}
			int32 Count = 0;
			for (int32 DY = -1; DY <= 1; ++DY)
			{
				const int32 NY = Y + DY;
				if (NY < 0 || NY >= H) continue;
				for (int32 DX = -1; DX <= 1; ++DX)
				{
					const int32 NX = X + DX;
					if (NX < 0 || NX >= W) continue;
					if (InMask[NY * W + NX]) ++Count;
				}
			}
			OutMask[Y * W + X] = (Count >= Threshold);
		}
	}
#endif
}



// ---------------------------------------------------------------------------
// Stage 4: Memory Encoder Preparation
// ---------------------------------------------------------------------------

/**
 * Prepares the high-resolution mask input for sam2_memory_encoder.
 *
 * Matches PyTorch's _encode_new_memory in sam2_base.py:700-712.
 * Logits are upsampled to model resolution FIRST, then sigmoid is applied
 * with scale/bias to produce the range [-10, +10].
 *
 * @param OutHiRes  Output (1, 1, 1024, 1024) float32 scaled sigmoid mask.
 * @param PredMask  (1, 1, 256, 256) selected mask logits (pre-sigmoid).
 */
static void PrepareMemoryEncoderInput(TArray<float>& OutHiRes, TConstArrayView<float> PredMask)
{
	// PredMask is (1, 1, 256, 256) — extract the (256, 256) slice
	constexpr int32 SrcSize = 256;
	TConstArrayView<float> Logits2D(PredMask.GetData(), SrcSize * SrcSize);

	BilinearUpsampleFloat(Logits2D, SrcSize, SrcSize, OutHiRes, ImageSize, ImageSize);

	// sigmoid(x) * SigmoidScale + SigmoidBias
	for (float& V : OutHiRes)
	{
		V = (1.0f / (1.0f + FMath::Exp(-V))) * SigmoidScale + SigmoidBias;
	}
}

/**
 * Gates the object pointer based on object presence score.
 *
 * Matches PyTorch's _forward_sam_heads in sam2_base.py:394-403.
 * When the object is not appearing (score <= 0), replaces the pointer with
 * the fixed no_obj_ptr to prevent noisy pointers from polluting the memory bank.
 *
 * @param ObjPtr           (256,) raw object pointer from mask decoder.
 * @param ObjScoreLogit    Scalar object presence logit.
 * @returns                (256,) gated object pointer.
 */
static TArray<float> GateObjPtr(
	TConstArrayView<float> ObjPtr,
	float ObjScoreLogit,
	const FSAM2Weights& Weights)
{
	const float IsObj    = ObjScoreLogit > 0.0f ? 1.0f : 0.0f;
	const float NotIsObj = 1.0f - IsObj;

	TArray<float> Result;
	Result.SetNumUninitialized(HiddenDim);
	for (int32 D = 0; D < HiddenDim; ++D)
	{
		Result[D] = IsObj * ObjPtr[D] + NotIsObj * Weights.NoObjPtr[D];
	}
	return Result;
}

/**
 * Adds the spatial no-object embedding to memory features when the object is not appearing.
 *
 * Matches PyTorch's _encode_new_memory in sam2_base.py.
 *
 * @param InOutFeatures  (1, 64, 64, 64) — modified in-place if object not appearing.
 * @param ObjScoreLogit  Scalar object presence logit.
 */
static void ApplyNoObjEmbedSpatial(
	TArray<float>& InOutFeatures,
	float ObjScoreLogit,
	const FSAM2Weights& Weights)
{
	if (ObjScoreLogit > 0.0f) return;

	// InOutFeatures: (1, MemDim, 64, 64) in NCHW = MemDim * 64 * 64 floats
	// NoObjEmbedSpatial: (1, MemDim) = MemDim floats
	// Broadcast: features[0, c, h, w] += NoObjEmbedSpatial[c]
	constexpr int32 SpatialArea = 64 * 64;
	for (int32 C = 0; C < MemDim; ++C)
	{
		const float Add = Weights.NoObjEmbedSpatial[C];
		float* Slice = InOutFeatures.GetData() + C * SpatialArea;
		for (int32 S = 0; S < SpatialArea; ++S)
		{
			Slice[S] += Add;
		}
	}
}

// ---------------------------------------------------------------------------
// Stage 5: Object Pointer Helpers
// ---------------------------------------------------------------------------

/**
 * Returns a view-compatible reshape of a 256-dim object pointer into 4 tokens of 64.
 *
 * The returned array is (4 * 64 = 256) floats — same data, just used as
 * NumObjPtrTokens × MemDim by the caller.
 *
 * @param ObjPtr  (256,) object pointer.
 * @returns       (256,) — same data; caller treats as (4, 64).
 */
static TArray<float> SplitObjPtr(TConstArrayView<float> ObjPtr)
{
	// (256,) → same data, caller treats as (NumObjPtrTokens=4, MemDim=64)
	return TArray<float>(ObjPtr.GetData(), ObjPtr.Num());
}

/**
 * 1D sine positional encoding (standard Transformer style).
 *
 * Port of sam2 get_1d_sine_pe (sam2/modeling/sam2_utils.py:64-74).
 *
 * @param Positions   (N,) float32 position values (may be fractional).
 * @param N           Number of positions.
 * @param Dim         Embedding dimension (must be even).
 * @param Temperature Frequency scaling base (default 10000).
 * @returns           (N, Dim) float32 positional encoding.
 */
static TArray<float> Get1DSinePe(
	TConstArrayView<float> Positions,
	int32 N,
	int32 Dim,
	float Temperature = 10000.0f)
{
	// Port of sam2 get_1d_sine_pe (sam2/modeling/sam2_utils.py).
	// pe_dim = Dim // 2 — each frequency pair (i, i+1) gets the same divisor.
	// dim_t[i] = Temperature^(2*(i//2)/pe_dim)  for i in 0..pe_dim-1
	// pos[n, i] = Positions[n] / dim_t[i]
	// result = [sin(pos[n, 0..pe_dim-1]), cos(pos[n, 0..pe_dim-1])] → (N, Dim)
	const int32 PeDim = Dim / 2;

	TArray<float> DimT;
	DimT.SetNumUninitialized(PeDim);
	for (int32 I = 0; I < PeDim; ++I)
	{
		const float Exp = 2.0f * static_cast<float>(I / 2) / static_cast<float>(PeDim);
		DimT[I] = FMath::Pow(Temperature, Exp);
	}

	TArray<float> Result;
	Result.SetNumUninitialized(N * Dim);
	for (int32 Ni = 0; Ni < N; ++Ni)
	{
		for (int32 I = 0; I < PeDim; ++I)
		{
			const float Val = Positions[Ni] / DimT[I];
			Result[Ni * Dim + I]          = FMath::Sin(Val);  // sin half
			Result[Ni * Dim + PeDim + I]  = FMath::Cos(Val);  // cos half
		}
	}
	return Result;
}

/**
 * Projects object pointer temporal PE from HiddenDim to MemDim via learned linear layer.
 *
 * @param SinePe  (N, HiddenDim=256) from Get1DSinePe.
 * @param N       Number of positions.
 * @returns       (N, MemDim=64) projected PE.
 */
static TArray<float> ApplyObjPtrTposProjection(
	TConstArrayView<float> SinePe,
	int32 N,
	const FSAM2Weights& Weights)
{
	// Linear: SinePe (N, HiddenDim=256) @ Weight.T (256, MemDim=64) + Bias (64,)
	// Weight stored as (MemDim, HiddenDim) = (64, 256) row-major.
	// result[n, j] = sum_k(SinePe[n,k] * Weight[j,k]) + Bias[j]
	TArray<float> Result;
	Result.SetNumUninitialized(N * MemDim);
	for (int32 Ni = 0; Ni < N; ++Ni)
	{
		const float* PeRow = SinePe.GetData() + Ni * HiddenDim;
		float* Out = Result.GetData() + Ni * MemDim;
		for (int32 J = 0; J < MemDim; ++J)
		{
			float Sum = Weights.ObjPtrTposProjBias[J];
			const float* WRow = Weights.ObjPtrTposProjWeight.GetData() + J * HiddenDim;
			for (int32 K = 0; K < HiddenDim; ++K)
			{
				Sum += PeRow[K] * WRow[K];
			}
			Out[J] = Sum;
		}
	}
	return Result;
}

// ---------------------------------------------------------------------------
// Stage 6: Memory Bank Assembly
// ---------------------------------------------------------------------------

/**
 * Assembles the memory bank for sam2_memory_attention on tracking frames (1+).
 *
 * Port of the memory assembly logic in sam2_base.py:522-640.
 * Combines spatial memory features (with temporal PE) and object pointer tokens
 * (with projected temporal PE), always padded to MaxObjPtrTokens = 64.
 *
 * @param SpatialFeatures    List of (1,64,64,64) maskmem_features, conditioning first.
 * @param SpatialPosEncs     Parallel list of (1,64,64,64) maskmem_pos_enc.
 * @param TPosIndices        Temporal position per frame: cond=0, most_recent=1, ...
 * @param ObjPtrFrameIndices Frame index for each obj_ptr (for distance computation).
 * @param ObjPtrs            List of (256,) object pointers.
 * @param CurrentFrameIdx    Index of the current frame being processed.
 * @param TotalFrameCount    Total number of frames in the sequence (caps distance normalization).
 * @returns                  memory and memory_pos both (M, 1, MemDim=64).
 *                           M = (num_spatial_frames × 64×64) + MaxObjPtrTokens.
 */
static FMemoryBankResult AssembleMemoryBank(
	const TArray<TArray<float>>& SpatialFeatures,
	const TArray<TArray<float>>& SpatialPosEncs,
	const TArray<int32>& TPosIndices,
	const TArray<int32>& ObjPtrFrameIndices,
	const TArray<TArray<float>>& ObjPtrs,
	int32 CurrentFrameIdx,
	int32 TotalFrameCount,
	const FSAM2Weights& Weights)
{
	constexpr int32 SpatialArea = 64 * 64;  // 4096 tokens per frame

	const int32 NumSpatialFrames = SpatialFeatures.Num();
	const int32 NumPtrs          = ObjPtrs.Num();
	const int32 TotalSpatialTokens = NumSpatialFrames * SpatialArea;

	// M = spatial tokens + always MaxObjPtrTokens (padded to bake num_k_exclude_rope)
	FMemoryBankResult Result;
	Result.M = TotalSpatialTokens + MaxObjPtrTokens;
	Result.Memory.SetNumZeroed(Result.M * MemDim);
	Result.MemoryPos.SetNumZeroed(Result.M * MemDim);

	// ---------- 1. Spatial frames ----------
	// feats: (1, 64, 64, 64) NCHW → flatten: feats_flat[hw, 0, c] = feats[0, c, h, w]
	// i.e. result[hw * MemDim + c] = feats[c * SpatialArea + hw]
	for (int32 F = 0; F < NumSpatialFrames; ++F)
	{
		const TArray<float>& Feats  = SpatialFeatures[F];
		const TArray<float>& PosEnc = SpatialPosEncs[F];
		const int32 TPos    = TPosIndices[F];
		const int32 TposIdx = NumMaskMemMax - TPos - 1;

		// maskmem_tpos_enc: (7,1,1,64) — enc for slot TposIdx is at TposIdx*64
		const float* TposEnc = Weights.MaskmemTposEnc.GetData() + TposIdx * MemDim;

		float* MemSlice    = Result.Memory.GetData()    + F * SpatialArea * MemDim;
		float* MemPosSlice = Result.MemoryPos.GetData() + F * SpatialArea * MemDim;

		for (int32 HW = 0; HW < SpatialArea; ++HW)
		{
			for (int32 C = 0; C < MemDim; ++C)
			{
				MemSlice[HW * MemDim + C]    = Feats[C * SpatialArea + HW];
				MemPosSlice[HW * MemDim + C] = PosEnc[C * SpatialArea + HW] + TposEnc[C];
			}
		}
	}

	// ---------- 2. Object pointer tokens ----------
	// Each ptr (256,) is split into NumObjPtrTokens=4 tokens of MemDim=64.
	// All 4 tokens share the same temporal PE.
	if (NumPtrs > 0)
	{
		// Compute normalized distances: |current - frame_idx| / max(min(total,16)-1, 1)
		const int32 MaxPtrs = FMath::Min(TotalFrameCount, MaxObjPtrsInEncoder);  // min(total, 16)
		const float DistNorm = static_cast<float>(FMath::Max(MaxPtrs - 1, 1));

		TArray<float> Distances;
		Distances.SetNumUninitialized(NumPtrs);
		for (int32 P = 0; P < NumPtrs; ++P)
		{
			Distances[P] = FMath::Abs(static_cast<float>(CurrentFrameIdx - ObjPtrFrameIndices[P])) / DistNorm;
		}

		// Temporal PE: Get1DSinePe → project → (NumPtrs, MemDim)
		TArray<float> SinePe  = Get1DSinePe(Distances, NumPtrs, HiddenDim);
		TArray<float> ProjPe  = ApplyObjPtrTposProjection(SinePe, NumPtrs, Weights);

		// Write pointer tokens + PE into the end of memory (after spatial tokens)
		// Tokens: ptr (NumObjPtrTokens=4, MemDim=64), all 4 share same PE
		const int32 PtrStart = TotalSpatialTokens;
		const int32 NumTokensToWrite = FMath::Min(NumPtrs * NumObjPtrTokens, MaxObjPtrTokens);

		for (int32 P = 0; P < NumPtrs && P * NumObjPtrTokens < MaxObjPtrTokens; ++P)
		{
			const float* PtrData  = ObjPtrs[P].GetData();  // (256,) = 4 × 64
			const float* PeProjP  = ProjPe.GetData() + P * MemDim;  // (64,)
			for (int32 T = 0; T < NumObjPtrTokens; ++T)
			{
				const int32 TokenIdx = P * NumObjPtrTokens + T;
				if (TokenIdx >= MaxObjPtrTokens) break;
				float* MemOut    = Result.Memory.GetData()    + (PtrStart + TokenIdx) * MemDim;
				float* MemPosOut = Result.MemoryPos.GetData() + (PtrStart + TokenIdx) * MemDim;
				FMemory::Memcpy(MemOut,    PtrData + T * MemDim, MemDim * sizeof(float));
				FMemory::Memcpy(MemPosOut, PeProjP,               MemDim * sizeof(float));
			}
		}
	}
	// Remaining ptr slots are already zero (SetNumZeroed above)

	return Result;
}

/**
 * Adds the no-memory embedding to the current frame features (frame 0 only).
 *
 * When no memory frames exist, SAM2 adds a learned no_mem_embed to the
 * current frame features before passing to the mask decoder. This replaces
 * the cross-attention that would occur on tracking frames.
 *
 * Reference: sam2_base.py directly_add_no_mem_embed config flag (TD-015).
 *
 * @param CurrFeatures  (1, HiddenDim=256, 64, 64) fpn2 from image encoder.
 * @returns             (1, 256, 64, 64) with no_mem_embed added.
 */
static void AddNoMemEmbed(
	TConstArrayView<float> CurrFeatures,
	const FSAM2Weights& Weights,
	TArray<float>& OutEmbeddings)
{
	// CurrFeatures: (1, HiddenDim=256, 64, 64) in NCHW
	// NoMemEmbed: (1, 1, 256) — reshape to (1, 256, 1, 1) and broadcast
	constexpr int32 SpatialArea = 64 * 64;
	OutEmbeddings.SetNumUninitialized(CurrFeatures.Num());
	// NoMemEmbed stored flat (256,) for our purposes — it's (1,1,256) row-major
	// so NoMemEmbed[c] is the value for channel c
	for (int32 C = 0; C < HiddenDim; ++C)
	{
		const float Add = Weights.NoMemEmbed[C];  // (1,1,256)[0,0,c]
		float* Dst = OutEmbeddings.GetData() + C * SpatialArea;
		const float* Src = CurrFeatures.GetData() + C * SpatialArea;
		for (int32 S = 0; S < SpatialArea; ++S)
		{
			Dst[S] = Src[S] + Add;
		}
	}
}

// ---------------------------------------------------------------------------
// Multi-object: Non-overlapping mask constraints
// ---------------------------------------------------------------------------

/**
 * Winner-take-all per-pixel overlap suppression for multi-person tracking.
 *
 * Matches SAM2's _apply_non_overlapping_constraints (sam2_base.py:891-909).
 * For each pixel, only the person with the highest logit keeps their value;
 * all others are clamped to min(-10.0, original). After sigmoid, -10.0 ≈ 4.5e-5.
 *
 * No-op when there is only one person.
 *
 * @param MaskLogits  List of N logit arrays (each 1,1,256,256, same shape).
 *                    Modified in-place: losers suppressed to -10.0.
 */
static void ApplyNonOverlappingConstraints(TArray<FPerPersonData>& PersonData)
{
	SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::ApplyNonOverlappingConstraints", FColor::Yellow);

	const int32 N = PersonData.Num();
	if (PersonData.Num() <= 1)
	{
		return;
	}

	const int32 NumPixels = PersonData[0].BestMask.Num();
	for (int32 P = 0; P < NumPixels; ++P)
	{
		// Find winner (highest logit) at this pixel
		int32 WinnerIdx = 0;
		for (int32 I = 1; I < N; ++I)
		{
			if (PersonData[I].BestMask[P] > PersonData[WinnerIdx].BestMask[P])
			{
				WinnerIdx = I;
			}
		}
		// Suppress losers
		for (int32 I = 0; I < N; ++I)
		{
			if (I != WinnerIdx)
			{
				PersonData[I].BestMask[P] = FMath::Min(-10.0f, PersonData[I].BestMask[P]);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Tracker class
// ---------------------------------------------------------------------------

class FSAM2Tracker : public ITracker
{
public:
	FSAM2Tracker(
		TSharedPtr<UE::NNE::IModelInstanceRunSync> InImageEncoder,
		TSharedPtr<UE::NNE::IModelInstanceRunSync> InMaskDecoder,
		TSharedPtr<UE::NNE::IModelInstanceRunSync> InMemoryEncoder,
		TSharedPtr<UE::NNE::IModelInstanceRunSync> InMemoryAttention,
		FSAM2Weights InWeights,
		EOverlapMode InOverlapMode,
		EResizeMethod InResizeMethod,
		float InMaskThreshold,
		int32 InNumMaskMem)
		: ImageEncoderInstance(MoveTemp(InImageEncoder))
		, MaskDecoderInstance(MoveTemp(InMaskDecoder))
		, MemoryEncoderInstance(MoveTemp(InMemoryEncoder))
		, MemoryAttentionInstance(MoveTemp(InMemoryAttention))
		, Weights(MoveTemp(InWeights))
		, OverlapMode(InOverlapMode)
		, ResizeMethod(InResizeMethod)
		, MaskThreshold(InMaskThreshold)
		, NumMaskMem(FMath::Clamp(InNumMaskMem, 1, NumMaskMemMax))
	{
		check(ImageEncoderInstance.IsValid());
		check(MaskDecoderInstance.IsValid());
		check(MemoryEncoderInstance.IsValid());
		check(MemoryAttentionInstance.IsValid());

		// Pre-compute constant image PE (used every frame, every person)
		ImagePe = GetDensePe(Weights);
	}

	virtual ~FSAM2Tracker() override = default;

	/**
	 * Initializes SAM2 tracking from frame 0 detections.
	 *
	 * Seeds one tracked person per detection using box + keypoint prompts.
	 * Runs the SAM2 image encoder once, then for each person:
	 *   encode_prompt → add_no_mem_embed → mask_decoder → select_mask
	 *   → gate_obj_ptr → memory_encoder → store conditioning memory.
	 *
	 * @param Frame       RGB frame (frame 0).
	 * @param Detections  Detectron2 detections (box + keypoints); one person per entry.
	 * @returns           One FTracking per detection with mask and IoU score.
	 */
	virtual TArray<FTracking> Initialize(
		const FImage& Frame,
		TConstArrayView<FDetection> Detections) override
	{
		return Step(Frame, Detections);
	}

	/**
	 * Tracks all initialized persons on a subsequent frame (1+).
	 *
	 * Runs image encoder once, then for each person:
	 *   assemble_memory_bank → memory_attention → encode_tracking_prompt
	 *   → mask_decoder → select_mask → gate_obj_ptr → memory_encoder → update state.
	 *
	 * @param Frame  RGB frame (frame 1+).
	 * @returns      One FTracking per tracked person with updated mask and IoU score.
	 */
	virtual TArray<FTracking> Track(const FImage& Frame) override { return Step(Frame, {}); }

private:
	TSharedPtr<UE::NNE::IModelInstanceRunSync> ImageEncoderInstance;
	TSharedPtr<UE::NNE::IModelInstanceRunSync> MaskDecoderInstance;
	TSharedPtr<UE::NNE::IModelInstanceRunSync> MemoryEncoderInstance;
	TSharedPtr<UE::NNE::IModelInstanceRunSync> MemoryAttentionInstance;

	FSAM2Weights Weights;
	TArray<float> ImagePe;               // (1,256,64,64) — constant Fourier PE, computed once
	TArray<FPersonMemoryState> PersonStates;
	int32 CurrentFrameIdx = 0;
	EOverlapMode OverlapMode;
	EResizeMethod ResizeMethod;
	float MaskThreshold;
	int32 NumMaskMem;                    // effective memory bank depth (1–NumMaskMemMax)

	// Cached input shapes for SetInputTensorShapesIfChanged
	TArray<UE::NNE::FTensorShape> CachedImageEncoderShapes;
	TArray<UE::NNE::FTensorShape> CachedMaskDecoderShapes;
	TArray<UE::NNE::FTensorShape> CachedMemoryEncoderShapes;
	TArray<UE::NNE::FTensorShape> CachedMemoryAttentionShapes;

	// Temporary Data
	TArray<float> CurrFlat;
	TArray<float> CurrPosFlat;
	TArray<float> ImageEmbeddings;

	TArray<float> PostProcessMasksUpsampled;
	TArray<bool> PostProcessMasksBinary;
	TArray<bool> BinaryMask;

	// Image preprocessing and encoder outputs
	TArray<float> ImageInput;   // (3, ImageSize, ImageSize)
	TArray<float> Fpn0;         // (1, HiddenDim, 256, 256)
	TArray<float> Fpn1;         // (1, HiddenDim, 128, 128)
	TArray<float> Fpn2;         // (1, HiddenDim, 64, 64)
	TArray<float> PosEnc2;      // (1, HiddenDim, 64, 64)
	TArray<float> Unused0;      // (1, HiddenDim, 256, 256) - unused encoder output slot
	TArray<float> Unused1;      // (1, HiddenDim, 128, 128) - unused encoder output slot

	// Per-person decode temporaries
	TArray<Private::FPerPersonData> PersonData;
	
	TArray<TArray<float>> SpatialFeatures, SpatialPosEncs;
	TArray<int32> TPosIndices;
	TArray<float> MemAttnOut;   // (SpatialArea, 1, HiddenDim)
	TArray<float> AllMasks;     // (1, NumMaskTokens, 256, 256)
	TArray<float> AllIous;      // (1, NumMaskTokens)
	TArray<float> AllPtrs;      // (1, NumMaskTokens, HiddenDim)
	TArray<float> ObjScore;     // (1,)
	TArray<float> HighResMasks; // (1, 1, ImageSize, ImageSize) — memory encoder input

	//Memory Encode
	TArray<float> MaskmemFeatures;
	TArray<float> MaskmemPosEnc;

	void PostprocessMasks(
		TConstArrayView<float> MaskLogits,
		TArray<bool>& OutMask,
		int32 OriginalH,
		int32 OriginalW,
		float Threshold);

	TArray<FTracking> Step(const FImage& Frame, TConstArrayView<FDetection> Detections);
};

TArray<FTracking> FSAM2Tracker::Step(
	const FImage& Frame,
	TConstArrayView<FDetection> Detections)
{
	using namespace UE::NNE;

	bool bInitialization = !Detections.IsEmpty();
	const int32 NPersons = bInitialization ? Detections.Num() : PersonStates.Num();
	if (NPersons == 0)
	{
		return {};
	}

	if (bInitialization)
	{
		CurrentFrameIdx = 0;
		PersonStates.SetNum(NPersons);
	}
	else
	{
		CurrentFrameIdx++;
	}
	const int32 FrameIdx = CurrentFrameIdx;
	// TotalFrameCount for distance normalization.
	// Python reference uses the actual total (20 frames → min(20,16)=16, DistNorm=15).
	// For any sequence ≥ MaxObjPtrsInEncoder frames, using MaxObjPtrsInEncoder gives the
	// same DistNorm (= MaxObjPtrsInEncoder - 1 = 15), matching the Python reference exactly.
	const int32 TotalFrameCount = MaxObjPtrsInEncoder;

	// --- Step 1: Preprocess image ---
	int32 OrigH = 0, OrigW = 0;
	PreprocessImage(ImageInput, Frame, OrigH, OrigW, ResizeMethod);

	// --- Step 2: Image encoder (1, 3, 1024, 1024) → fpn0, fpn1, fpn2, pos_enc2 ---
	// Outputs indexed 0=fpn0, 1=fpn1, 2=fpn2, 3=pos_enc0, 4=pos_enc1, 5=pos_enc2
	constexpr int32 Fpn0Vol = 1 * HiddenDim * 256 * 256;
	constexpr int32 Fpn1Vol = 1 * HiddenDim * 128 * 128;
	constexpr int32 Fpn2Vol = 1 * HiddenDim * 64 * 64;

	{
		SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::ImageEncoderSetShapes", FColor::Yellow);
		TArray<FTensorShape> InShapes;
		InShapes.Add(FTensorShape::Make({ 1, 3, static_cast<uint32>(ImageSize), static_cast<uint32>(ImageSize) }));
		verifyf(NNEUtils::SetInputTensorShapesIfChanged(*ImageEncoderInstance, InShapes, CachedImageEncoderShapes) == EResultStatus::Ok,
			TEXT("SAM2 ImageEncoder: SetInputTensorShapes failed"));
	}

	Fpn0.SetNumUninitialized(Fpn0Vol);
	Fpn1.SetNumUninitialized(Fpn1Vol);
	Fpn2.SetNumUninitialized(Fpn2Vol);
	PosEnc2.SetNumUninitialized(Fpn2Vol);
	Unused0.SetNumUninitialized(Fpn0Vol);
	Unused1.SetNumUninitialized(Fpn1Vol);
	{
		SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::ImageEncoder", FColor::Yellow);
		FTensorBindingCPU InBind = { ImageInput.GetData(), static_cast<uint64>(ImageInput.Num()) * sizeof(float) };
		TArray<FTensorBindingCPU> OutBinds = {
			{ Fpn0.GetData(),    static_cast<uint64>(Fpn0Vol) * sizeof(float) },
			{ Fpn1.GetData(),    static_cast<uint64>(Fpn1Vol) * sizeof(float) },
			{ Fpn2.GetData(),    static_cast<uint64>(Fpn2Vol) * sizeof(float) },
			{ Unused0.GetData(), static_cast<uint64>(Fpn0Vol) * sizeof(float) },
			{ Unused1.GetData(), static_cast<uint64>(Fpn1Vol) * sizeof(float) },
			{ PosEnc2.GetData(), static_cast<uint64>(Fpn2Vol) * sizeof(float) },
		};
		verifyf(ImageEncoderInstance->RunSync({ InBind }, OutBinds) == EResultStatus::Ok,
			TEXT("SAM2 ImageEncoder: RunSync failed"));
	}

	// Flatten fpn2/pos_enc2: (1, 256, 64, 64) → (4096, 1, 256)
	// feats_flat[hw, 0, c] = feats[0, c, hw/64, hw%64] = feats[c * 4096 + hw]
	constexpr int32 SpatialArea = 64 * 64;  // 4096
	constexpr int32 CurrVol = SpatialArea * 1 * HiddenDim;
	if (!bInitialization)
	{
		CurrFlat.SetNumUninitialized(CurrVol);
		CurrPosFlat.SetNumUninitialized(CurrVol);
		{
			SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::ImageEncoderFlatten", FColor::Yellow);
			for (int32 HW = 0; HW < SpatialArea; ++HW)
			{
				for (int32 C = 0; C < HiddenDim; ++C)
				{
					CurrFlat[HW * HiddenDim + C] = Fpn2[C * SpatialArea + HW];
					CurrPosFlat[HW * HiddenDim + C] = PosEnc2[C * SpatialArea + HW];
				}
			}
		}
	}

	// --- Step 3: Per-person decode ---
	PersonData.SetNum(NPersons);

	FPromptEmbeddings Prompt;
	if (bInitialization)
	{
		AddNoMemEmbed(Fpn2, Weights, ImageEmbeddings);
	}
	else
	{
		Prompt = EncodeTrackingPrompt(Weights);
	}

	constexpr int32 AllMasksVol = 1 * NumMaskTokens * 256 * 256;
	constexpr int32 AllIousVol = 1 * NumMaskTokens;
	constexpr int32 AllPtrsVol = 1 * NumMaskTokens * HiddenDim;
	constexpr int32 ObjScoreVol = 1 * 1;
	constexpr int32 MaskDecoderVol = AllMasksVol + AllIousVol + AllPtrsVol + ObjScoreVol;

	for (int32 PersonIdx = 0; PersonIdx < NPersons; ++PersonIdx)
	{
		SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::Decode", FColor::Yellow);
		if (bInitialization)
		{
			const FDetection& Det = Detections[PersonIdx];

			// Build keypoint xy array (17 × 2)
			TArray<float> KptXY;
			int32 NumKpts = 0;
			if (!Det.Keypoints.IsEmpty())
			{
				NumKpts = Det.Keypoints.Num();
				KptXY.SetNumUninitialized(NumKpts * 2);
				for (int32 K = 0; K < NumKpts; ++K)
				{
					KptXY[K * 2 + 0] = Det.Keypoints[K][0];  // x
					KptXY[K * 2 + 1] = Det.Keypoints[K][1];  // y
				}
			}
			Prompt = EncodePrompt(
				OrigH, OrigW, Weights,
				Det.Box,       // (x1,y1,x2,y2) may be empty
				KptXY,         // (NumKpts * 2,) may be empty
				NumKpts,
				{});           // default all-foreground labels
		}
		else
		{
			FPersonMemoryState& State = PersonStates[PersonIdx];

			// Assemble memory bank
			SpatialFeatures.Reset();
			SpatialPosEncs.Reset();
			TPosIndices.Reset();
			// Conditioning frame at t_pos=0
			SpatialFeatures.Add(State.CondMaskmemFeatures);
			SpatialPosEncs.Add(State.CondMaskmemPosEnc);
			TPosIndices.Add(0);

			// Recent frames, newest first, t_pos=1,2,...
			const int32 MaxRecent = NumMaskMem - 1;
			for (int32 R = 0; R < FMath::Min(State.RecentFeatures.Num(), MaxRecent); ++R)
			{
				SpatialFeatures.Add(State.RecentFeatures[R]);
				SpatialPosEncs.Add(State.RecentPosEncs[R]);
				TPosIndices.Add(R + 1);
			}

			FMemoryBankResult Bank = AssembleMemoryBank(
				SpatialFeatures, SpatialPosEncs, TPosIndices,
				State.ObjPtrFrameIndices, State.ObjPtrs,
				FrameIdx, TotalFrameCount, Weights);

			// Memory attention
			const int32 MVal = Bank.M;
			{
				TArray<FTensorShape> InShapes;
				InShapes.Add(FTensorShape::Make({ static_cast<uint32>(SpatialArea), 1, static_cast<uint32>(HiddenDim) }));  // curr
				InShapes.Add(FTensorShape::Make({ static_cast<uint32>(MVal), 1, static_cast<uint32>(MemDim) }));             // memory
				InShapes.Add(FTensorShape::Make({ static_cast<uint32>(SpatialArea), 1, static_cast<uint32>(HiddenDim) }));  // curr_pos
				InShapes.Add(FTensorShape::Make({ static_cast<uint32>(MVal), 1, static_cast<uint32>(MemDim) }));             // memory_pos
				verifyf(NNEUtils::SetInputTensorShapesIfChanged(*MemoryAttentionInstance, InShapes, CachedMemoryAttentionShapes) == EResultStatus::Ok,
					TEXT("SAM2 MemoryAttention: SetInputTensorShapes failed"));
			}

			MemAttnOut.SetNumUninitialized(CurrVol);
			{
				TArray<FTensorBindingCPU> InBinds = {
					{ CurrFlat.GetData(),     static_cast<uint64>(CurrVol) * sizeof(float) },
					{ Bank.Memory.GetData(),  static_cast<uint64>(Bank.Memory.Num()) * sizeof(float) },
					{ CurrPosFlat.GetData(),  static_cast<uint64>(CurrVol) * sizeof(float) },
					{ Bank.MemoryPos.GetData(), static_cast<uint64>(Bank.MemoryPos.Num()) * sizeof(float) },
				};
				TArray<FTensorBindingCPU> OutBinds = {
					{ MemAttnOut.GetData(), static_cast<uint64>(CurrVol) * sizeof(float) },
				};
				verifyf(MemoryAttentionInstance->RunSync(InBinds, OutBinds) == EResultStatus::Ok,
					TEXT("SAM2 MemoryAttention: RunSync failed"));
			}

			// Reshape output (4096, 1, 256) → (1, 256, 64, 64)
			ImageEmbeddings.SetNumUninitialized(Fpn2Vol);
			for (int32 HW = 0; HW < SpatialArea; ++HW)
			{
				for (int32 C = 0; C < HiddenDim; ++C)
				{
					ImageEmbeddings[C * SpatialArea + HW] = MemAttnOut[HW * HiddenDim + C];
				}
			}
		}

		// Mask decoder
		{
			TArray<FTensorShape> InShapes;
			InShapes.Add(FTensorShape::Make({ 1, static_cast<uint32>(HiddenDim), 64, 64 }));          // image_embeddings
			InShapes.Add(FTensorShape::Make({ 1, static_cast<uint32>(HiddenDim), 64, 64 }));          // image_pe
			InShapes.Add(FTensorShape::Make({ 1, static_cast<uint32>(Prompt.NumTokens), static_cast<uint32>(HiddenDim) }));  // sparse_prompts
			InShapes.Add(FTensorShape::Make({ 1, static_cast<uint32>(HiddenDim), 64, 64 }));          // dense_prompts
			InShapes.Add(FTensorShape::Make({ 1, static_cast<uint32>(HiddenDim), 256, 256 }));        // high_res_feat0
			InShapes.Add(FTensorShape::Make({ 1, static_cast<uint32>(HiddenDim), 128, 128 }));        // high_res_feat1
			verifyf(NNEUtils::SetInputTensorShapesIfChanged(*MaskDecoderInstance, InShapes, CachedMaskDecoderShapes) == EResultStatus::Ok,
				TEXT("SAM2 MaskDecoder: SetInputTensorShapes failed"));
		}

		AllMasks.SetNumUninitialized(AllMasksVol);
		AllIous.SetNumUninitialized(AllIousVol);
		AllPtrs.SetNumUninitialized(AllPtrsVol);
		ObjScore.SetNumUninitialized(ObjScoreVol);
		{
			TArray<FTensorBindingCPU> InBinds = {
				{ ImageEmbeddings.GetData(), static_cast<uint64>(Fpn2Vol) * sizeof(float) },
				{ ImagePe.GetData(),         static_cast<uint64>(Fpn2Vol) * sizeof(float) },
				{ Prompt.Sparse.GetData(),   static_cast<uint64>(Prompt.Sparse.Num()) * sizeof(float) },
				{ Prompt.Dense.GetData(),    static_cast<uint64>(Prompt.Dense.Num()) * sizeof(float) },
				{ Fpn0.GetData(),            static_cast<uint64>(Fpn0Vol) * sizeof(float) },
				{ Fpn1.GetData(),            static_cast<uint64>(Fpn1Vol) * sizeof(float) },
			};
			TArray<FTensorBindingCPU> OutBinds = {
				{ AllMasks.GetData(), static_cast<uint64>(AllMasksVol) * sizeof(float) },
				{ AllIous.GetData(),  static_cast<uint64>(AllIousVol) * sizeof(float) },
				{ AllPtrs.GetData(),  static_cast<uint64>(AllPtrsVol) * sizeof(float) },
				{ ObjScore.GetData(), static_cast<uint64>(ObjScoreVol) * sizeof(float) },
			};
			verifyf(MaskDecoderInstance->RunSync(InBinds, OutBinds) == EResultStatus::Ok,
				TEXT("SAM2 MaskDecoder: RunSync failed"));
		}

		FMaskSelection Sel = SelectMaskAndObjPtr(AllMasks, AllIous, AllPtrs, ObjScore[0], /*multimask=*/!bInitialization);
		PersonData[PersonIdx].BestMask = MoveTemp(Sel.Mask);
		PersonData[PersonIdx].GatedObjPtr = GateObjPtr(Sel.ObjPtr, ObjScore[0], Weights);
		PersonData[PersonIdx].ObjScoreLogit = ObjScore[0];
		PersonData[PersonIdx].BestIou = Sel.Iou;
	}

	// --- Apply BeforeMemory overlap constraint ---
	if (OverlapMode == EOverlapMode::BeforeMemory)
	{
		ApplyNonOverlappingConstraints(PersonData);
	}

	// --- Step 4: Memory encoding per person ---
	for (int32 PersonIdx = 0; PersonIdx < NPersons; ++PersonIdx)
	{
		SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::MemoryEncode", FColor::Yellow);

		FPerPersonData& PD = PersonData[PersonIdx];
		FPersonMemoryState& State = PersonStates[PersonIdx];

		PrepareMemoryEncoderInput(HighResMasks, PD.BestMask);
		constexpr int32 MemFeatVol = 1 * MemDim * 64 * 64;
		{
			TArray<FTensorShape> InShapes;
			InShapes.Add(FTensorShape::Make({ 1, static_cast<uint32>(HiddenDim), 64, 64 }));     // pix_feat
			InShapes.Add(FTensorShape::Make({ 1, 1, static_cast<uint32>(ImageSize), static_cast<uint32>(ImageSize) }));  // masks
			verifyf(NNEUtils::SetInputTensorShapesIfChanged(*MemoryEncoderInstance, InShapes, CachedMemoryEncoderShapes) == EResultStatus::Ok,
				TEXT("SAM2 MemoryEncoder: SetInputTensorShapes failed"));
		}

		MaskmemFeatures.SetNumUninitialized(MemFeatVol);
		MaskmemPosEnc.SetNumUninitialized(MemFeatVol);
		{
			TArray<FTensorBindingCPU> InBinds = {
				{ Fpn2.GetData(),        static_cast<uint64>(Fpn2Vol) * sizeof(float) },
				{ HighResMasks.GetData(), static_cast<uint64>(HighResMasks.Num()) * sizeof(float) },
			};
			TArray<FTensorBindingCPU> OutBinds = {
				{ MaskmemFeatures.GetData(), static_cast<uint64>(MemFeatVol) * sizeof(float) },
				{ MaskmemPosEnc.GetData(),   static_cast<uint64>(MemFeatVol) * sizeof(float) },
			};
			verifyf(MemoryEncoderInstance->RunSync(InBinds, OutBinds) == EResultStatus::Ok,
				TEXT("SAM2 MemoryEncoder: RunSync failed"));
		}

		ApplyNoObjEmbedSpatial(MaskmemFeatures, PD.ObjScoreLogit, Weights);

		if (bInitialization)
		{
			State.RecentFeatures.Empty();
			State.RecentPosEncs.Empty();
			State.ObjPtrFrameIndices = { 0 };
			State.ObjPtrs = { PD.GatedObjPtr };
			State.CondMaskmemFeatures = MoveTemp(MaskmemFeatures);
			State.CondMaskmemPosEnc = MoveTemp(MaskmemPosEnc);
		}
		else
		{
			// Update recent memories (newest first, max NumMaskMem-1)
			State.RecentFeatures.Insert(MoveTemp(MaskmemFeatures), 0);
			State.RecentPosEncs.Insert(MoveTemp(MaskmemPosEnc), 0);
			const int32 MaxRecent = NumMaskMem - 1;
			if (State.RecentFeatures.Num() > MaxRecent)
			{
				State.RecentFeatures.SetNum(MaxRecent);
				State.RecentPosEncs.SetNum(MaxRecent);
			}

			// Update obj_ptr history (keep last MaxObjPtrsInEncoder=16)
			State.ObjPtrFrameIndices.Add(FrameIdx);
			State.ObjPtrs.Add(MoveTemp(PD.GatedObjPtr));
			if (State.ObjPtrFrameIndices.Num() > MaxObjPtrsInEncoder)
			{
				State.ObjPtrFrameIndices.RemoveAt(0);
				State.ObjPtrs.RemoveAt(0);
			}
		}
	}

	// --- Output logits → binary masks ---
	if (OverlapMode == EOverlapMode::OutputOnly)
	{
		ApplyNonOverlappingConstraints(PersonData);
	}

	// --- Build output FTracking ---
	TArray<FTracking> Results;
	Results.SetNum(NPersons);
	for (int32 PersonIdx = 0; PersonIdx < NPersons; ++PersonIdx)
	{
		SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::BinaryMasks", FColor::Yellow);

		FTracking Out = {};
		Out.Id = PersonIdx;
		Out.Score = PersonData[PersonIdx].BestIou;

		// Create binary mask at original resolution
		PostprocessMasks(PersonData[PersonIdx].BestMask, BinaryMask, OrigH, OrigW, MaskThreshold);

		// Bounding box derived from mask (matches reference data format)
		{
			SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::BinaryMasks::BoundingBox", FColor::Cyan);
#if 1
			static constexpr int Parallelization = 32;
			const int WorkSize = FMath::DivideAndRoundDown(OrigH, Parallelization);
			TStaticArray<FIntRect, Parallelization> Bounds;
			ParallelFor(Parallelization, [&](int Index)
				{
					FIntRect Bound(OrigW, OrigH, 0, 0);
					const int StartIndex = WorkSize * Index;
					const int OnePatEndIndex = FMath::Min(StartIndex + WorkSize, OrigH);
					for (int32 Y = StartIndex; Y < OnePatEndIndex; ++Y)
					{
						for (int32 X = 0; X < OrigW; ++X)
						{
							if (BinaryMask[Y * OrigW + X])
							{
								Bound.Include(FInt32Point(X, Y));
							}
						}
					}
					Bounds[Index] = Bound;
				}, EParallelForFlags::None);
			FIntRect FinalBound = Bounds[0];
			for (int Index = 0; Index < Parallelization; Index++)
			{
				FinalBound.Union(Bounds[Index]);
			}
			if (FinalBound.Min.X < FinalBound.Max.X && FinalBound.Min.Y < FinalBound.Max.Y)
			{
				Out.Box = FinalBound;
			}
#else
			int32 X1 = OrigW, Y1 = OrigH, X2 = -1, Y2 = -1;
			for (int32 Y = 0; Y < OrigH; ++Y)
			{
				for (int32 X = 0; X < OrigW; ++X)
				{
					if (BinaryMask[Y * OrigW + X])
					{
						X1 = FMath::Min(X1, X);
						Y1 = FMath::Min(Y1, Y);
						X2 = FMath::Max(X2, X);
						Y2 = FMath::Max(Y2, Y);
					}
				}
			}
			if (X2 >= 0)
			{
				Out.Box = FIntRect{ X1, Y1, X2, Y2 };
			}
#endif
		}

		FImage ImageMask;
		ImageMask.Init(OrigW, OrigH, ERawImageFormat::G8, EGammaSpace::Linear);
		uint8* MaskData = ImageMask.RawData.GetData();
		for (int32 P = 0; P < OrigH * OrigW; ++P)
		{
			MaskData[P] = BinaryMask[P] ? 255u : 0u;
		}
		Out.Mask = MoveTemp(ImageMask);
		Results[PersonIdx] = Out;
	}
	return Results;
}

/**
 * Upsamples low-res mask logits, thresholds to binary, and cleans stray pixels.
 *
 * Port of sam2_processing.py postprocess_masks.
 *
 * @param MaskLogits  (1, 1, 256, 256) float32 logits from mask_decoder.
 * @param OutMask     Flat (OriginalH × OriginalW) boolean binary mask.
 * @param OriginalH   Target height (original video frame height).
 * @param OriginalW   Target width.
 * @param Threshold   Logit threshold (default 0.0 → sigmoid > 0.5).
 */
void FSAM2Tracker::PostprocessMasks(
	TConstArrayView<float> MaskLogits,
	TArray<bool>& OutMask,
	int32 OriginalH,
	int32 OriginalW,
	float Threshold)
{
	SCOPED_NAMED_EVENT_TEXT("FSAM2Tracker::BinaryMasks::PostProcessMasks", FColor::Cyan);

	// MaskLogits is (1, 1, 256, 256) — extract the (256, 256) slice
	constexpr int32 SrcSize = 256;
	TConstArrayView<float> Logits2D(MaskLogits.GetData(), SrcSize * SrcSize);

	BilinearUpsampleFloat(Logits2D, SrcSize, SrcSize, PostProcessMasksUpsampled, OriginalH, OriginalW);

	PostProcessMasksBinary.SetNumUninitialized(OriginalH * OriginalW);
#if 1
	static const int Parallelization = 32;
	const int WorkSize = FMath::DivideAndRoundUp(PostProcessMasksBinary.Num(), Parallelization);
	ParallelFor(Parallelization, [&](int Index)
		{
			const int StartIndex = WorkSize * Index;
			const int OnePastEndIndex = FMath::Min(StartIndex + WorkSize, PostProcessMasksBinary.Num());
			for (int32 I = StartIndex; I < OnePastEndIndex; I++)
			{
				PostProcessMasksBinary[I] = PostProcessMasksUpsampled[I] > Threshold;
			}
		}, EParallelForFlags::None);
#else
	for (int32 I = 0; I < PostProcessMasksBinary.Num(); ++I)
	{
		PostProcessMasksBinary[I] = Upsampled[I] > Threshold;
	}
#endif

	CleanMask(PostProcessMasksBinary, OutMask, OriginalH, OriginalW);
}

} // namespace UE::NNE::Segmentation::SAM2::Private

// ---------------------------------------------------------------------------
// Public factory
// ---------------------------------------------------------------------------

namespace UE::NNE::Segmentation::SAM2
{

bool TryLoadModelData(
	UNNEModelData*& OutImageEncoderData,
	UNNEModelData*& OutMaskDecoderData,
	UNNEModelData*& OutMemoryEncoderData,
	UNNEModelData*& OutMemoryAttentionData,
	UDataTable*& OutWeightsTable)
{
	OutImageEncoderData    = LoadObject<UNNEModelData>(nullptr, Private::ImageEncoderAssetPath);
	OutMaskDecoderData     = LoadObject<UNNEModelData>(nullptr, Private::MaskDecoderAssetPath);
	OutMemoryEncoderData   = LoadObject<UNNEModelData>(nullptr, Private::MemoryEncoderAssetPath);
	OutMemoryAttentionData = LoadObject<UNNEModelData>(nullptr, Private::MemoryAttentionAssetPath);
	OutWeightsTable        = LoadObject<UDataTable>(nullptr, Private::WeightsAssetPath);
	return OutImageEncoderData && OutMaskDecoderData && OutMemoryEncoderData && OutMemoryAttentionData && OutWeightsTable;
}

TUniquePtr<ITracker> Make(
	TConstArrayView<FRuntimePreference> RuntimePreferences,
	const FMakeParams& Params)
{
	using namespace Private;

	// Use caller-supplied model data or load from plugin content
	UNNEModelData* ImageEncoderData = Params.ImageEncoderData
		? Params.ImageEncoderData
		: LoadObject<UNNEModelData>(nullptr, ImageEncoderAssetPath);
	UNNEModelData* MaskDecoderData = Params.MaskDecoderData
		? Params.MaskDecoderData
		: LoadObject<UNNEModelData>(nullptr, MaskDecoderAssetPath);
	UNNEModelData* MemoryEncoderData = Params.MemoryEncoderData
		? Params.MemoryEncoderData
		: LoadObject<UNNEModelData>(nullptr, MemoryEncoderAssetPath);
	UNNEModelData* MemoryAttentionData = Params.MemoryAttentionData
		? Params.MemoryAttentionData
		: LoadObject<UNNEModelData>(nullptr, MemoryAttentionAssetPath);

	if (!ImageEncoderData || !MaskDecoderData || !MemoryEncoderData || !MemoryAttentionData)
	{
		UE_LOG(LogSAM2, Error, TEXT("Failed to load one or more SAM2 model assets. "
			"Ensure the 4 ONNX models are imported under /Segmentation/SAM2/."));
		return nullptr;
	}

	// Load and validate learned weights
	UDataTable* WeightsTable = Params.WeightsTable 
		? Params.WeightsTable
		: LoadObject<UDataTable>(nullptr, WeightsAssetPath);
	if (!WeightsTable)
	{
		UE_LOG(LogSAM2, Error, TEXT("Failed to load SAM2 weights DataTable at %s. "
			"Generate sam2_weights.csv and import it into /Segmentation/SAM2/."),
			WeightsAssetPath);
		return nullptr;
	}

	FSAM2Weights Weights;
	if (!FSAM2Weights::FromDataTable(WeightsTable, Weights))
	{
		UE_LOG(LogSAM2, Error, TEXT("SAM2 weights DataTable at %s is incomplete or corrupt."),
			WeightsAssetPath);
		return nullptr;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> ImageEncoder = NNEUtils::TryCreateInstance(ImageEncoderData, RuntimePreferences);
	if (!ImageEncoder.IsValid())
	{
		UE_LOG(LogSAM2, Verbose, TEXT("Failed to create ImageEncoder instance"));
		return nullptr;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> MaskDecoder = NNEUtils::TryCreateInstance(MaskDecoderData, RuntimePreferences);
	if (!MaskDecoder.IsValid())
	{
		UE_LOG(LogSAM2, Verbose, TEXT("Failed to create MaskDecoder instance"));
		return nullptr;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> MemoryEncoder = NNEUtils::TryCreateInstance(MemoryEncoderData, RuntimePreferences);
	if (!MemoryEncoder.IsValid())
	{
		UE_LOG(LogSAM2, Verbose, TEXT("Failed to create MemoryEncoder instance"));
		return nullptr;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> MemoryAttention = NNEUtils::TryCreateInstance(MemoryAttentionData, RuntimePreferences);
	if (!MemoryAttention.IsValid())
	{
		UE_LOG(LogSAM2, Verbose, TEXT("Failed to create MemoryAttention instance"));
		return nullptr;
	}

	return MakeUnique<FSAM2Tracker>(
		MoveTemp(ImageEncoder),
		MoveTemp(MaskDecoder),
		MoveTemp(MemoryEncoder),
		MoveTemp(MemoryAttention),
		MoveTemp(Weights),
		Params.OverlapMode,
		Params.ResizeMethod,
		Params.MaskThreshold,
		Params.NumMaskMem);
}

} // namespace UE::NNE::Segmentation::SAM2
