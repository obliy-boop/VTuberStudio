// Copyright Epic Games, Inc. All Rights Reserved.

#include "Detectron2.h"

#include "NNEModelData.h"
#include "NNEUtils.h"
#include "ResampleUtils.h"

DEFINE_LOG_CATEGORY_STATIC(LogDetectron2, Log, All);

namespace UE::NNE::Segmentation::Detectron2::Private
{
	
// Model asset paths (plugin content)
static const TCHAR* BackboneRpnAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/Detectron2/detectron2_backbone_rpn");
static const TCHAR* RoiHeadsBoxAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/Detectron2/detectron2_roi_heads_box");
static const TCHAR* RoiHeadsKeypointAssetPath =
	TEXT("/" UE_PLUGIN_NAME "/Detectron2/detectron2_roi_heads_keypoint_patched");

// ---------------------------------------------------------------------------
// Model-specific constants (Keypoint R-CNN X-101-32x8d-FPN-3x)
// ---------------------------------------------------------------------------

// Preprocessing
static constexpr float PixelMeanBGR[3] = { 103.530f, 116.280f, 123.675f };
static constexpr float PixelStdBGR[3]  = {  57.375f,  57.120f,  58.395f };
static constexpr int32 MinSize = 800;
static constexpr int32 MaxSize = 1333;
static constexpr int32 SizeDivisibility = 32;

// Anchor generation — 1 size per FPN level, 3 aspect ratios
static constexpr int32 NumFpnLevels = 5;
static constexpr int32 AnchorSizesPerLevel[NumFpnLevels] = { 32, 64, 128, 256, 512 };
static constexpr float AnchorAspectRatios[3] = { 0.5f, 1.0f, 2.0f };
static constexpr int32 NumAspectRatios = 3;
static constexpr int32 AnchorStrides[NumFpnLevels] = { 4, 8, 16, 32, 64 };
static constexpr float AnchorOffset = 0.0f;

// Box regression weights
static constexpr float RpnBoxRegWeights[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
static constexpr float BoxRegWeights[4] = { 10.0f, 10.0f, 5.0f, 5.0f };

// Scale clamp: log(1000/16) ≈ 4.135
static constexpr float ScaleClamp = 4.135166556742356f;

// Keypoint postprocessing
static constexpr int32 HeatmapSize = 56;
static constexpr int32 NumKeypoints = 17;

struct FPreprocessResult
{
	float ScaleX;    // new_w / original_w
	float ScaleY;    // new_h / original_h
	int32 PaddedH;
	int32 PaddedW;
};

// ---------------------------------------------------------------------------
// Static helper functions
// ---------------------------------------------------------------------------

static void GenerateAnchors(
	TConstArrayView<FIntPoint> FeatureSizes,
	TArray<TArray<float>>& OutAnchors)
{
	OutAnchors.SetNum(NumFpnLevels);

	for (int32 Level = 0; Level < NumFpnLevels; ++Level)
	{
		const int32 Stride = AnchorStrides[Level];
		const int32 Size = AnchorSizesPerLevel[Level];
		const int32 FH = FeatureSizes[Level].Y; // Height
		const int32 FW = FeatureSizes[Level].X; // Width
		const float Area = static_cast<float>(Size * Size);

		// Generate 3 cell anchors centered at origin
		float CellAnchors[NumAspectRatios * 4];
		for (int32 R = 0; R < NumAspectRatios; ++R)
		{
			const float W = FMath::Sqrt(Area / AnchorAspectRatios[R]);
			const float H = W * AnchorAspectRatios[R];
			CellAnchors[R * 4 + 0] = -W / 2.0f;
			CellAnchors[R * 4 + 1] = -H / 2.0f;
			CellAnchors[R * 4 + 2] =  W / 2.0f;
			CellAnchors[R * 4 + 3] =  H / 2.0f;
		}

		// Generate grid anchors: for each spatial position, shift each cell anchor
		const int32 NumAnchors = FH * FW * NumAspectRatios;
		OutAnchors[Level].SetNumUninitialized(NumAnchors * 4);
		float* Dst = OutAnchors[Level].GetData();

		for (int32 Y = 0; Y < FH; ++Y)
		{
			const float ShiftY = Y * Stride + AnchorOffset * Stride;
			for (int32 X = 0; X < FW; ++X)
			{
				const float ShiftX = X * Stride + AnchorOffset * Stride;
				for (int32 R = 0; R < NumAspectRatios; ++R)
				{
					*Dst++ = CellAnchors[R * 4 + 0] + ShiftX;
					*Dst++ = CellAnchors[R * 4 + 1] + ShiftY;
					*Dst++ = CellAnchors[R * 4 + 2] + ShiftX;
					*Dst++ = CellAnchors[R * 4 + 3] + ShiftY;
				}
			}
		}
	}
}

static void ApplyBoxDeltas(
	const float* Deltas,
	const float* Boxes,
	const float Weights[4],
	int32 NumBoxes,
	TArray<float>& OutBoxes)
{
	OutBoxes.SetNumUninitialized(NumBoxes * 4);
	float* Out = OutBoxes.GetData();

	const float Wx = Weights[0], Wy = Weights[1], Ww = Weights[2], Wh = Weights[3];

	for (int32 I = 0; I < NumBoxes; ++I)
	{
		const int32 Idx = I * 4;
		const float X1 = Boxes[Idx + 0];
		const float Y1 = Boxes[Idx + 1];
		const float X2 = Boxes[Idx + 2];
		const float Y2 = Boxes[Idx + 3];

		const float Width  = X2 - X1;
		const float Height = Y2 - Y1;
		const float CtrX = X1 + 0.5f * Width;
		const float CtrY = Y1 + 0.5f * Height;

		const float Dx = Deltas[Idx + 0] / Wx;
		const float Dy = Deltas[Idx + 1] / Wy;
		const float Dw = FMath::Min(Deltas[Idx + 2] / Ww, ScaleClamp);
		const float Dh = FMath::Min(Deltas[Idx + 3] / Wh, ScaleClamp);

		const float PredCtrX = Dx * Width + CtrX;
		const float PredCtrY = Dy * Height + CtrY;
		const float PredW = FMath::Exp(Dw) * Width;
		const float PredH = FMath::Exp(Dh) * Height;

		Out[Idx + 0] = PredCtrX - 0.5f * PredW;
		Out[Idx + 1] = PredCtrY - 0.5f * PredH;
		Out[Idx + 2] = PredCtrX + 0.5f * PredW;
		Out[Idx + 3] = PredCtrY + 0.5f * PredH;
	}
}

static void ClipBoxes(float* Boxes, int32 NumBoxes, int32 ImageH, int32 ImageW)
{
	const float MaxX = static_cast<float>(ImageW);
	const float MaxY = static_cast<float>(ImageH);

	for (int32 I = 0; I < NumBoxes; ++I)
	{
		const int32 Idx = I * 4;
		Boxes[Idx + 0] = FMath::Clamp(Boxes[Idx + 0], 0.0f, MaxX);
		Boxes[Idx + 1] = FMath::Clamp(Boxes[Idx + 1], 0.0f, MaxY);
		Boxes[Idx + 2] = FMath::Clamp(Boxes[Idx + 2], 0.0f, MaxX);
		Boxes[Idx + 3] = FMath::Clamp(Boxes[Idx + 3], 0.0f, MaxY);
	}
}

static void SoftmaxInPlace(float* Data, int32 Rows, int32 Cols)
{
	for (int32 R = 0; R < Rows; ++R)
	{
		float* Row = Data + R * Cols;

		// Find max for numerical stability
		float MaxVal = Row[0];
		for (int32 C = 1; C < Cols; ++C)
		{
			MaxVal = FMath::Max(MaxVal, Row[C]);
		}

		// Exp and sum
		float Sum = 0.0f;
		for (int32 C = 0; C < Cols; ++C)
		{
			Row[C] = FMath::Exp(Row[C] - MaxVal);
			Sum += Row[C];
		}

		// Normalize
		const float InvSum = 1.0f / Sum;
		for (int32 C = 0; C < Cols; ++C)
		{
			Row[C] *= InvSum;
		}
	}
}

static void Nms(
	const float* Boxes,
	const float* Scores,
	int32 NumBoxes,
	float IoUThreshold,
	TArray<int32>& OutIndices)
{
	OutIndices.Reset();

	if (NumBoxes == 0)
	{
		return;
	}

	// Sort indices by score descending
	TArray<int32> Order;
	Order.SetNumUninitialized(NumBoxes);
	for (int32 I = 0; I < NumBoxes; ++I)
	{
		Order[I] = I;
	}
	Order.Sort([Scores](int32 A, int32 B) { return Scores[A] > Scores[B]; });

	TArray<bool> Suppressed;
	Suppressed.SetNumZeroed(NumBoxes);

	for (int32 I = 0; I < NumBoxes; ++I)
	{
		const int32 Idx = Order[I];
		if (Suppressed[Idx])
		{
			continue;
		}

		OutIndices.Add(Idx);

		const float X1_I = Boxes[Idx * 4 + 0];
		const float Y1_I = Boxes[Idx * 4 + 1];
		const float X2_I = Boxes[Idx * 4 + 2];
		const float Y2_I = Boxes[Idx * 4 + 3];
		const float AreaI = (X2_I - X1_I) * (Y2_I - Y1_I);

		for (int32 J = I + 1; J < NumBoxes; ++J)
		{
			const int32 JIdx = Order[J];
			if (Suppressed[JIdx])
			{
				continue;
			}

			const float X1_J = Boxes[JIdx * 4 + 0];
			const float Y1_J = Boxes[JIdx * 4 + 1];
			const float X2_J = Boxes[JIdx * 4 + 2];
			const float Y2_J = Boxes[JIdx * 4 + 3];

			const float XX1 = FMath::Max(X1_I, X1_J);
			const float YY1 = FMath::Max(Y1_I, Y1_J);
			const float XX2 = FMath::Min(X2_I, X2_J);
			const float YY2 = FMath::Min(Y2_I, Y2_J);

			const float Inter = FMath::Max(0.0f, XX2 - XX1) * FMath::Max(0.0f, YY2 - YY1);
			const float AreaJ = (X2_J - X1_J) * (Y2_J - Y1_J);
			const float IoU = Inter / (AreaI + AreaJ - Inter + 1e-6f);

			if (IoU > IoUThreshold)
			{
				Suppressed[JIdx] = true;
			}
		}
	}
}

static void ToPoolerFormat(
	const float* Boxes,
	int32 NumBoxes,
	TArray<float>& OutPoolerBoxes)
{
	OutPoolerBoxes.SetNumUninitialized(NumBoxes * 5);
	float* Dst = OutPoolerBoxes.GetData();

	for (int32 I = 0; I < NumBoxes; ++I)
	{
		const int32 SrcIdx = I * 4;
		*Dst++ = 0.0f; // batch index
		*Dst++ = Boxes[SrcIdx + 0];
		*Dst++ = Boxes[SrcIdx + 1];
		*Dst++ = Boxes[SrcIdx + 2];
		*Dst++ = Boxes[SrcIdx + 3];
	}
}

/**
 * Transposes a (1, C, H, W) NCHW tensor to flat (H*W*C) HWC order.
 * Used to reshape RPN logits and deltas before processing.
 */
static void TransposeNCHWtoHWC(
	const float* Src,
	int32 C, int32 H, int32 W,
	TArray<float>& Out)
{
	const int32 Total = H * W * C;
	Out.SetNumUninitialized(Total);
	float* Dst = Out.GetData();

	for (int32 Y = 0; Y < H; ++Y)
	{
		for (int32 X = 0; X < W; ++X)
		{
			for (int32 Ch = 0; Ch < C; ++Ch)
			{
				// NCHW index: 0*C*H*W + Ch*H*W + Y*W + X
				*Dst++ = Src[Ch * H * W + Y * W + X];
			}
		}
	}
}

class FDetectron2 : public IDetector
{
public:
	FDetectron2(
		TSharedPtr<UE::NNE::IModelInstanceRunSync> InBackboneRpn,
		TSharedPtr<UE::NNE::IModelInstanceRunSync> InRoiHeadsBox,
		TSharedPtr<UE::NNE::IModelInstanceRunSync> InRoiHeadsKeypoint,
		EKeypointMethod InKeypointMethod,
		EResizeMethod InResizeMethod,
		float InBoxScoreThresh,
		float InBoxNmsThresh,
		int32 InBoxTopKPerImage,
		int32 InRpnPreNmsTopK,
		float InRpnNmsThresh,
		int32 InRpnPostNmsTopK)
		: BackboneRpnInstance(MoveTemp(InBackboneRpn))
		, RoiHeadsBoxInstance(MoveTemp(InRoiHeadsBox))
		, RoiHeadsKeypointInstance(MoveTemp(InRoiHeadsKeypoint))
		, KeypointMethod(InKeypointMethod)
		, ResizeMethod(InResizeMethod)
		, BoxScoreThresh(InBoxScoreThresh)
		, BoxNmsThresh(InBoxNmsThresh)
		, BoxTopKPerImage(InBoxTopKPerImage)
		, RpnPreNmsTopK(InRpnPreNmsTopK)
		, RpnNmsThresh(InRpnNmsThresh)
		, RpnPostNmsTopK(InRpnPostNmsTopK)
	{
		check(BackboneRpnInstance.IsValid());
		check(RoiHeadsBoxInstance.IsValid());
		check(RoiHeadsKeypointInstance.IsValid());
		check(InBoxScoreThresh >= 0.0f && InBoxScoreThresh <= 1.0f);
		check(InBoxNmsThresh > 0.0f && InBoxNmsThresh <= 1.0f);
		check(InBoxTopKPerImage > 0);
		check(InRpnPreNmsTopK > 0);
		check(InRpnNmsThresh > 0.0f && InRpnNmsThresh <= 1.0f);
		check(InRpnPostNmsTopK > 0);
	}

	virtual ~FDetectron2() override = default;

	virtual TArray<FDetection> Detect(const FImage& Frame) override
	{
		SCOPED_NAMED_EVENT_TEXT("FDetectron2::Detect", FColor::Orange);

		using namespace UE::NNE;

		// Validate input image
		const ERawImageFormat::Type Format = Frame.Format;
		const int32 NumChannels = ERawImageFormat::NumChannels(Format);

		checkf(NumChannels >= 3,
			TEXT("Detect requires at least 3 channels, got %d (format: %s)"),
			NumChannels, ERawImageFormat::GetName(Format));

		checkf(Format == ERawImageFormat::BGRA8
			|| Format == ERawImageFormat::BGRE8
			|| Format == ERawImageFormat::RGBA16
			|| Format == ERawImageFormat::RGBA16F
			|| Format == ERawImageFormat::RGBA32F,
			TEXT("Unsupported image format: %s. Expected a 3+ channel format (8/16/32 bit)."),
			ERawImageFormat::GetName(Format));

		checkf(Frame.GetWidth() > 0 && Frame.GetHeight() > 0,
			TEXT("Detect requires a non-empty image, got %lldx%lld"),
			Frame.GetWidth(), Frame.GetHeight());

		const int32 OrigH = Frame.GetHeight();
		const int32 OrigW = Frame.GetWidth();

		// =====================================================================
		// Stage 1: Preprocess
		// =====================================================================
		TArray<float> ImageTensor;
		const FPreprocessResult Preprocess = PreprocessImage(Frame, ImageTensor);
		const int32 PH = Preprocess.PaddedH;
		const int32 PW = Preprocess.PaddedW;

		// =====================================================================
		// Stage 2: Backbone + RPN inference
		// =====================================================================
		// Input: image (1, 3, PaddedH, PaddedW)
		// Outputs [15]: p2..p6, rpn_logits_p2..p6, rpn_deltas_p2..p6

		{
			TArray<FTensorShape> InputShapes;
			InputShapes.Add(FTensorShape::Make({1, 3, static_cast<uint32>(PH), static_cast<uint32>(PW)}));
			verifyf(NNEUtils::SetInputTensorShapesIfChanged(*BackboneRpnInstance, InputShapes, CachedBackboneRpnShapes) == EResultStatus::Ok,
				TEXT("BackboneRpn: SetInputTensorShapes failed"));
		}

		// Compute feature sizes per FPN level from padded image size
		TArray<FIntPoint> FeatureSizes;
		FeatureSizes.SetNum(NumFpnLevels);
		for (int32 L = 0; L < NumFpnLevels; ++L)
		{
			// p6 has same stride as p5 (64) in this model, but feature sizes come
			// from the actual output shapes. Compute from strides for p2-p5,
			// p6 matches p5 dimensions.
			const int32 Stride = AnchorStrides[L];
			FeatureSizes[L] = FIntPoint(
				(PW + Stride - 1) / Stride,  // Width (X)
				(PH + Stride - 1) / Stride);  // Height (Y)
		}

		// Allocate output buffers for backbone_rpn (15 outputs)
		// Some runtimes don't resolve output shapes from SetInputTensorShapes alone,
		// so we compute expected sizes from the known model structure:
		//   outputs[ 0.. 4] = FPN features p2..p6:  (1, 256, Hi, Wi)
		//   outputs[ 5.. 9] = RPN logits p2..p6:    (1, 3, Hi, Wi)
		//   outputs[10..14] = RPN deltas p2..p6:    (1, 12, Hi, Wi)
		static constexpr int32 FpnChannels = 256;
		static constexpr int32 RpnLogitChannels = NumAspectRatios;      // 3
		static constexpr int32 RpnDeltaChannels = NumAspectRatios * 4;  // 12

		TArray<TArray<float>> BrpnOutputs;
		BrpnOutputs.SetNum(15);
		TArray<FTensorBindingCPU> BrpnOutBindings;
		BrpnOutBindings.SetNum(15);

		for (int32 L = 0; L < NumFpnLevels; ++L)
		{
			const int32 H = FeatureSizes[L].Y;
			const int32 W = FeatureSizes[L].X;

			// FPN feature
			const int32 FpnVol = 1 * FpnChannels * H * W;
			BrpnOutputs[L].SetNumUninitialized(FpnVol);
			BrpnOutBindings[L] = { BrpnOutputs[L].GetData(), static_cast<uint64>(FpnVol) * sizeof(float) };

			// RPN logits
			const int32 LogitVol = 1 * RpnLogitChannels * H * W;
			BrpnOutputs[5 + L].SetNumUninitialized(LogitVol);
			BrpnOutBindings[5 + L] = { BrpnOutputs[5 + L].GetData(), static_cast<uint64>(LogitVol) * sizeof(float) };

			// RPN deltas
			const int32 DeltaVol = 1 * RpnDeltaChannels * H * W;
			BrpnOutputs[10 + L].SetNumUninitialized(DeltaVol);
			BrpnOutBindings[10 + L] = { BrpnOutputs[10 + L].GetData(), static_cast<uint64>(DeltaVol) * sizeof(float) };
		}

		// Run backbone_rpn
		{
			FTensorBindingCPU InputBinding = {
				ImageTensor.GetData(),
				static_cast<uint64>(ImageTensor.Num()) * sizeof(float)
			};
			verifyf(BackboneRpnInstance->RunSync({ InputBinding }, BrpnOutBindings) == EResultStatus::Ok,
				TEXT("BackboneRpn: RunSync failed"));
		}

		// FPN features: outputs[0..4] = p2, p3, p4, p5, p6
		// RPN logits:   outputs[5..9]  = rpn_logits_p2..p6, each (1, 3, Hi, Wi)
		// RPN deltas:   outputs[10..14] = rpn_deltas_p2..p6, each (1, 12, Hi, Wi)

		// =====================================================================
		// Stage 3: RPN postprocessing
		// =====================================================================
		// Transpose RPN outputs from NCHW to flat HWC for processing functions

		TArray<TArray<float>> RpnLogits;
		TArray<TArray<float>> RpnDeltas;
		RpnLogits.SetNum(NumFpnLevels);
		RpnDeltas.SetNum(NumFpnLevels);

		TArray<TArray<float>> Anchors;
		GenerateAnchors(FeatureSizes, Anchors);

		for (int32 L = 0; L < NumFpnLevels; ++L)
		{
			const int32 FH = FeatureSizes[L].Y;
			const int32 FW = FeatureSizes[L].X;

			// Logits: (1, 3, H, W) → flat (H*W*3) — one score per anchor
			TransposeNCHWtoHWC(BrpnOutputs[5 + L].GetData(), NumAspectRatios, FH, FW, RpnLogits[L]);

			// Deltas: (1, 12, H, W) → flat (H*W*12) = (H*W*3, 4)
			TransposeNCHWtoHWC(BrpnOutputs[10 + L].GetData(), NumAspectRatios * 4, FH, FW, RpnDeltas[L]);
		}

		TArray<float> Proposals;
		int32 NumProposals = 0;
		RpnPostprocess(RpnLogits, RpnDeltas, Anchors, FeatureSizes,
			PH, PW, Proposals, NumProposals);

		if (NumProposals == 0)
		{
			return {};
		}

		// =====================================================================
		// Stage 4: ROI box head inference
		// =====================================================================
		// Inputs: p2, p3, p4, p5 (NOT p6) + boxes (N, 5) in pooler format
		// Outputs: cls_scores (N, 2), box_deltas (N, 4)

		TArray<float> PoolerBoxes;
		ToPoolerFormat(Proposals.GetData(), NumProposals, PoolerBoxes);

		{
			// 5 inputs: p2, p3, p4, p5, boxes
			TArray<FTensorShape> BoxInputShapes;
			for (int32 L = 0; L < 4; ++L) // p2-p5 only
			{
				BoxInputShapes.Add(FTensorShape::Make({
					1, static_cast<uint32>(FpnChannels),
					static_cast<uint32>(FeatureSizes[L].Y),
					static_cast<uint32>(FeatureSizes[L].X)
				}));
			}
			BoxInputShapes.Add(FTensorShape::Make({
				static_cast<uint32>(NumProposals), 5
			}));
			verifyf(NNEUtils::SetInputTensorShapesIfChanged(*RoiHeadsBoxInstance, BoxInputShapes, CachedRoiHeadsBoxShapes) == EResultStatus::Ok,
				TEXT("RoiHeadsBox: SetInputTensorShapes failed"));
		}

		// Allocate outputs: cls_scores (N, 2) and box_deltas (N, 4)
		TArray<float> ClsScores, BoxDeltasOut;
		ClsScores.SetNumUninitialized(NumProposals * 2);
		BoxDeltasOut.SetNumUninitialized(NumProposals * 4);

		{
			TArray<FTensorBindingCPU> BoxInputBindings;
			for (int32 L = 0; L < 4; ++L) // p2-p5
			{
				BoxInputBindings.Add({
					BrpnOutputs[L].GetData(),
					static_cast<uint64>(BrpnOutputs[L].Num()) * sizeof(float)
				});
			}
			BoxInputBindings.Add({
				PoolerBoxes.GetData(),
				static_cast<uint64>(PoolerBoxes.Num()) * sizeof(float)
			});

			TArray<FTensorBindingCPU> BoxOutputBindings = {
				{ ClsScores.GetData(), static_cast<uint64>(ClsScores.Num()) * sizeof(float) },
				{ BoxDeltasOut.GetData(), static_cast<uint64>(BoxDeltasOut.Num()) * sizeof(float) }
			};

			verifyf(RoiHeadsBoxInstance->RunSync(BoxInputBindings, BoxOutputBindings) == EResultStatus::Ok,
				TEXT("RoiHeadsBox: RunSync failed"));
		}

		// =====================================================================
		// Stage 5: Box postprocessing
		// =====================================================================
		TArray<float> DetBoxes, DetScores;
		int32 NumDetections = 0;
		BoxPostprocess(ClsScores.GetData(), BoxDeltasOut.GetData(),
			Proposals.GetData(), NumProposals,
			PH, PW, DetBoxes, DetScores, NumDetections);

		if (NumDetections == 0)
		{
			return {};
		}

		// =====================================================================
		// Stage 6: ROI keypoint head inference
		// =====================================================================
		// Inputs: p2, p3, p4, p5 (NOT p6) + boxes (D, 5) in pooler format
		// Outputs: keypoint_heatmaps (D, 17, 56, 56)

		TArray<float> KpPoolerBoxes;
		ToPoolerFormat(DetBoxes.GetData(), NumDetections, KpPoolerBoxes);

		{
			TArray<FTensorShape> KpInputShapes;
			for (int32 L = 0; L < 4; ++L) // p2-p5 only
			{
				KpInputShapes.Add(FTensorShape::Make({
					1, static_cast<uint32>(FpnChannels),
					static_cast<uint32>(FeatureSizes[L].Y),
					static_cast<uint32>(FeatureSizes[L].X)
				}));
			}
			KpInputShapes.Add(FTensorShape::Make({
				static_cast<uint32>(NumDetections), 5
			}));
			verifyf(NNEUtils::SetInputTensorShapesIfChanged(*RoiHeadsKeypointInstance, KpInputShapes, CachedRoiHeadsKeypointShapes) == EResultStatus::Ok,
				TEXT("RoiHeadsKeypoint: SetInputTensorShapes failed"));
		}

		TArray<float> Heatmaps;
		Heatmaps.SetNumUninitialized(NumDetections * NumKeypoints * HeatmapSize * HeatmapSize);

		{
			TArray<FTensorBindingCPU> KpInputBindings;
			for (int32 L = 0; L < 4; ++L) // p2-p5
			{
				KpInputBindings.Add({
					BrpnOutputs[L].GetData(),
					static_cast<uint64>(BrpnOutputs[L].Num()) * sizeof(float)
				});
			}
			KpInputBindings.Add({
				KpPoolerBoxes.GetData(),
				static_cast<uint64>(KpPoolerBoxes.Num()) * sizeof(float)
			});

			FTensorBindingCPU KpOutputBinding = {
				Heatmaps.GetData(),
				static_cast<uint64>(Heatmaps.Num()) * sizeof(float)
			};

			verifyf(RoiHeadsKeypointInstance->RunSync(KpInputBindings, { KpOutputBinding }) == EResultStatus::Ok,
				TEXT("RoiHeadsKeypoint: RunSync failed"));
		}

		// =====================================================================
		// Stage 7: Keypoint postprocessing
		// =====================================================================
		TArray<float> KeypointsFlat;
		KeypointPostprocess(Heatmaps.GetData(), DetBoxes.GetData(), NumDetections, KeypointsFlat);

		// =====================================================================
		// Stage 8: Rescale to original coordinates
		// =====================================================================
		RescaleDetections(DetBoxes.GetData(), KeypointsFlat.GetData(), NumDetections,
			Preprocess.ScaleX, Preprocess.ScaleY, OrigH, OrigW);

		// =====================================================================
		// Assemble output
		// =====================================================================
		TArray<FDetection> Results;
		Results.Reserve(NumDetections);

		for (int32 D = 0; D < NumDetections; ++D)
		{
			FDetection Det;
			Det.Box = { FMath::RoundToInt(DetBoxes[D * 4 + 0]), 
					    FMath::RoundToInt(DetBoxes[D * 4 + 1]),
						FMath::RoundToInt(DetBoxes[D * 4 + 2]), 
						FMath::RoundToInt(DetBoxes[D * 4 + 3]) };
			Det.Score = DetScores[D];

			Det.Keypoints.SetNum(NumKeypoints);
			for (int32 K = 0; K < NumKeypoints; ++K)
			{
				const int32 Idx = (D * NumKeypoints + K) * 3;
				Det.Keypoints[K] = {
					KeypointsFlat[Idx + 0],
					KeypointsFlat[Idx + 1],
					KeypointsFlat[Idx + 2]
				};
			}

			Results.Add(MoveTemp(Det));
		}

		return Results;
	}

private:
	/**
	 * Preprocesses an input image for the Detectron2 backbone_rpn model.
	 *
	 * Performs in one pass: format conversion (any supported format → BGR float32),
	 * bilinear resize (shortest edge → 800, max edge ≤ 1333), per-channel normalization,
	 * HWC→CHW transpose, and zero-padding to a multiple of 32.
	 *
	 * @param Frame       Input image in any supported format (BGRA8, BGRE8, RGBA16, RGBA16F, RGBA32F).
	 * @param OutTensor   Output buffer, resized to hold (1 * 3 * PaddedH * PaddedW) floats in CHW layout.
	 * @return            Scale factors and padded dimensions needed for postprocessing.
	 */
	FPreprocessResult PreprocessImage(const FImage& Frame, TArray<float>& OutTensor)
	{
		const int32 OrigH = Frame.GetHeight();
		const int32 OrigW = Frame.GetWidth();

		// Compute resize scale: shortest edge → MinSize, max edge ≤ MaxSize
		float Scale = static_cast<float>(MinSize) / FMath::Min(OrigH, OrigW);
		if (FMath::Max(OrigH, OrigW) * Scale > MaxSize)
		{
			Scale = static_cast<float>(MaxSize) / FMath::Max(OrigH, OrigW);
		}
		const int32 NewH = FMath::RoundToInt32(OrigH * Scale);
		const int32 NewW = FMath::RoundToInt32(OrigW * Scale);

		// Resize to BGRA8 using the selected method
		FImage ResizedImage;
		if (ResizeMethod == EResizeMethod::PILBilinear)
		{
			// Ensure BGRA8 input for the PIL-compatible path
			FImage Bgra8Frame;
			if (Frame.Format != ERawImageFormat::BGRA8)
			{
				Frame.CopyTo(Bgra8Frame, ERawImageFormat::BGRA8, EGammaSpace::Linear);
			}
			const FImage& Src = (Frame.Format == ERawImageFormat::BGRA8) ? Frame : Bgra8Frame;

			ResizedImage.Init(NewW, NewH, ERawImageFormat::BGRA8, EGammaSpace::Linear);
			ResizePILBilinear(
				Src.RawData.GetData(), OrigW, OrigH,
				ResizedImage.RawData.GetData(), NewW, NewH);
		}
		else
		{
			FImageCore::ResizeImageAllocDest(Frame, ResizedImage, NewW, NewH,
				ERawImageFormat::BGRA8, EGammaSpace::Linear,
				FImageCore::EResizeImageFilter::Triangle);
		}

		// Pad dimensions to next multiple of SizeDivisibility
		const int32 PadH = (SizeDivisibility - NewH % SizeDivisibility) % SizeDivisibility;
		const int32 PadW = (SizeDivisibility - NewW % SizeDivisibility) % SizeDivisibility;
		const int32 PaddedH = NewH + PadH;
		const int32 PaddedW = NewW + PadW;

		// Allocate output: (1, 3, PaddedH, PaddedW) as flat array in CHW order
		const int32 ChannelStride = PaddedH * PaddedW;
		OutTensor.SetNumZeroed(3 * ChannelStride);

		// Normalize and transpose HWC→CHW in a single pass.
		// BGRA8 layout: [B, G, R, A, B, G, R, A, ...]
		// Output CHW layout: [B_channel...][G_channel...][R_channel...]
		const uint8* SrcPixels = ResizedImage.RawData.GetData();
		float* DstB = OutTensor.GetData();                     // Channel 0 = Blue
		float* DstG = OutTensor.GetData() + ChannelStride;     // Channel 1 = Green
		float* DstR = OutTensor.GetData() + 2 * ChannelStride; // Channel 2 = Red

		for (int32 Y = 0; Y < NewH; ++Y)
		{
			for (int32 X = 0; X < NewW; ++X)
			{
				const int32 SrcIdx = (Y * NewW + X) * 4; // BGRA8 = 4 bytes per pixel
				const int32 DstIdx = Y * PaddedW + X;

				DstB[DstIdx] = (static_cast<float>(SrcPixels[SrcIdx + 0]) - PixelMeanBGR[0]) / PixelStdBGR[0];
				DstG[DstIdx] = (static_cast<float>(SrcPixels[SrcIdx + 1]) - PixelMeanBGR[1]) / PixelStdBGR[1];
				DstR[DstIdx] = (static_cast<float>(SrcPixels[SrcIdx + 2]) - PixelMeanBGR[2]) / PixelStdBGR[2];
			}
		}
		// Padded pixels remain zero from SetNumZeroed

		FPreprocessResult Result;
		Result.ScaleX = static_cast<float>(NewW) / OrigW;
		Result.ScaleY = static_cast<float>(NewH) / OrigH;
		Result.PaddedH = PaddedH;
		Result.PaddedW = PaddedW;
		return Result;
	}

	/**
	 * Decodes RPN outputs into region proposals.
	 *
	 * For each FPN level: takes top-K by foreground score, decodes box deltas
	 * against anchors, clips to image bounds, then runs per-level NMS.
	 * Finally merges proposals across all levels and keeps the overall top-K.
	 */
	void RpnPostprocess(
		TConstArrayView<TArray<float>> RpnLogits,
		TConstArrayView<TArray<float>> RpnDeltas,
		TConstArrayView<TArray<float>> Anchors,
		TConstArrayView<FIntPoint> FeatureSizes,
		int32 ImageH,
		int32 ImageW,
		TArray<float>& OutProposals,
		int32& OutNumProposals) const
	{
		TArray<float> AllProposals;
		TArray<float> AllScores;

		for (int32 Level = 0; Level < NumFpnLevels; ++Level)
		{
			const int32 FH = FeatureSizes[Level].Y;
			const int32 FW = FeatureSizes[Level].X;
			const int32 NumAnchorsPerLevel = FH * FW * NumAspectRatios;

			// RPN logits come as (1, 3, H, W) — transpose to (H, W, 3) then flatten
			// The ONNX model output is already flat (NumAnchors, 2) or (NumAnchors,)
			// depending on model variant. We receive flat arrays here.
			// Logits: (NumAnchors,) objectness scores (foreground logit)
			const float* LevelLogits = RpnLogits[Level].GetData();

			// Deltas: (NumAnchors, 4)
			const float* LevelDeltas = RpnDeltas[Level].GetData();
			const float* LevelAnchors = Anchors[Level].GetData();

			// Top-K by score (before NMS)
			const int32 K = FMath::Min(RpnPreNmsTopK, NumAnchorsPerLevel);

			// Find top-K indices by score
			TArray<int32> Indices;
			Indices.SetNumUninitialized(NumAnchorsPerLevel);
			for (int32 I = 0; I < NumAnchorsPerLevel; ++I)
			{
				Indices[I] = I;
			}

			// Partial sort: move top-K to front
			if (K < NumAnchorsPerLevel)
			{
				Indices.Sort([LevelLogits](int32 A, int32 B)
				{
					return LevelLogits[A] > LevelLogits[B];
				});
			}

			// Decode deltas for top-K only
			TArray<float> TopKDeltas;
			TArray<float> TopKAnchors;
			TArray<float> TopKScores;
			TopKDeltas.SetNumUninitialized(K * 4);
			TopKAnchors.SetNumUninitialized(K * 4);
			TopKScores.SetNumUninitialized(K);

			for (int32 I = 0; I < K; ++I)
			{
				const int32 Idx = Indices[I];
				TopKScores[I] = LevelLogits[Idx];
				FMemory::Memcpy(&TopKDeltas[I * 4], &LevelDeltas[Idx * 4], 4 * sizeof(float));
				FMemory::Memcpy(&TopKAnchors[I * 4], &LevelAnchors[Idx * 4], 4 * sizeof(float));
			}

			// Apply box deltas
			TArray<float> DecodedBoxes;
			ApplyBoxDeltas(TopKDeltas.GetData(), TopKAnchors.GetData(),
				RpnBoxRegWeights, K, DecodedBoxes);

			// Clip to image boundaries
			ClipBoxes(DecodedBoxes.GetData(), K, ImageH, ImageW);

			// Per-level NMS
			TArray<int32> KeepIndices;
			Nms(DecodedBoxes.GetData(), TopKScores.GetData(), K, RpnNmsThresh, KeepIndices);

			// Gather kept proposals and scores
			for (int32 KeepIdx : KeepIndices)
			{
				AllProposals.Add(DecodedBoxes[KeepIdx * 4 + 0]);
				AllProposals.Add(DecodedBoxes[KeepIdx * 4 + 1]);
				AllProposals.Add(DecodedBoxes[KeepIdx * 4 + 2]);
				AllProposals.Add(DecodedBoxes[KeepIdx * 4 + 3]);
				AllScores.Add(TopKScores[KeepIdx]);
			}
		}

		// Merge all levels, sort by score, keep top RpnPostNmsTopK
		const int32 TotalProposals = AllScores.Num();
		TArray<int32> MergeOrder;
		MergeOrder.SetNumUninitialized(TotalProposals);
		for (int32 I = 0; I < TotalProposals; ++I)
		{
			MergeOrder[I] = I;
		}
		const float* MergeScores = AllScores.GetData();
		MergeOrder.Sort([MergeScores](int32 A, int32 B)
		{
			return MergeScores[A] > MergeScores[B];
		});

		OutNumProposals = FMath::Min(TotalProposals, RpnPostNmsTopK);
		OutProposals.SetNumUninitialized(OutNumProposals * 4);

		for (int32 I = 0; I < OutNumProposals; ++I)
		{
			const int32 Idx = MergeOrder[I];
			FMemory::Memcpy(&OutProposals[I * 4], &AllProposals[Idx * 4], 4 * sizeof(float));
		}
	}

	/**
	 * Decodes ROI box head outputs into final detections.
	 *
	 * Applies softmax to classification scores, decodes box deltas against proposals,
	 * clips to image bounds, filters by foreground score threshold, and runs NMS.
	 */
	void BoxPostprocess(
		const float* ClsScores,
		const float* BoxDeltas,
		const float* Proposals,
		int32 NumProposals,
		int32 ImageH,
		int32 ImageW,
		TArray<float>& OutBoxes,
		TArray<float>& OutScores,
		int32& OutNumDetections) const
	{
		OutBoxes.Reset();
		OutScores.Reset();
		OutNumDetections = 0;

		if (NumProposals == 0)
		{
			return;
		}

		// Softmax on classification scores (N, 2): col 0 = person, col 1 = background
		TArray<float> Probs;
		Probs.SetNumUninitialized(NumProposals * 2);
		FMemory::Memcpy(Probs.GetData(), ClsScores, NumProposals * 2 * sizeof(float));
		SoftmaxInPlace(Probs.GetData(), NumProposals, 2);

		// Filter by validity and score threshold
		TArray<int32> ValidIndices;
		ValidIndices.Reserve(NumProposals);

		for (int32 I = 0; I < NumProposals; ++I)
		{
			const float Score = Probs[I * 2]; // Person class score
			const bool bFiniteScore = FMath::IsFinite(Score);
			const bool bFiniteDeltas = FMath::IsFinite(BoxDeltas[I * 4 + 0])
				&& FMath::IsFinite(BoxDeltas[I * 4 + 1])
				&& FMath::IsFinite(BoxDeltas[I * 4 + 2])
				&& FMath::IsFinite(BoxDeltas[I * 4 + 3]);

			if (bFiniteScore && bFiniteDeltas && Score > BoxScoreThresh)
			{
				ValidIndices.Add(I);
			}
		}

		if (ValidIndices.Num() == 0)
		{
			return;
		}

		// Gather valid proposals and deltas
		const int32 NumValid = ValidIndices.Num();
		TArray<float> ValidDeltas, ValidProposals, ValidScores;
		ValidDeltas.SetNumUninitialized(NumValid * 4);
		ValidProposals.SetNumUninitialized(NumValid * 4);
		ValidScores.SetNumUninitialized(NumValid);

		for (int32 I = 0; I < NumValid; ++I)
		{
			const int32 Idx = ValidIndices[I];
			ValidScores[I] = Probs[Idx * 2];
			FMemory::Memcpy(&ValidDeltas[I * 4], &BoxDeltas[Idx * 4], 4 * sizeof(float));
			FMemory::Memcpy(&ValidProposals[I * 4], &Proposals[Idx * 4], 4 * sizeof(float));
		}

		// Decode box deltas
		TArray<float> DecodedBoxes;
		ApplyBoxDeltas(ValidDeltas.GetData(), ValidProposals.GetData(),
			BoxRegWeights, NumValid, DecodedBoxes);

		// Clip to image boundaries
		ClipBoxes(DecodedBoxes.GetData(), NumValid, ImageH, ImageW);

		// NMS
		TArray<int32> NmsIndices;
		Nms(DecodedBoxes.GetData(), ValidScores.GetData(), NumValid, BoxNmsThresh, NmsIndices);

		// Top-K
		OutNumDetections = FMath::Min(NmsIndices.Num(), BoxTopKPerImage);
		OutBoxes.SetNumUninitialized(OutNumDetections * 4);
		OutScores.SetNumUninitialized(OutNumDetections);

		for (int32 I = 0; I < OutNumDetections; ++I)
		{
			const int32 Idx = NmsIndices[I];
			OutScores[I] = ValidScores[Idx];
			FMemory::Memcpy(&OutBoxes[I * 4], &DecodedBoxes[Idx * 4], 4 * sizeof(float));
		}
	}

	/**
	 * Converts keypoint heatmaps to (x, y, confidence) coordinates.
	 *
	 * Fast method: argmax on 56x56 heatmap + analytic mapping to box coordinates.
	 * Bicubic method: per-ROI bicubic upsample to box pixel size, then argmax.
	 */
	void KeypointPostprocess(
		const float* Heatmaps,
		const float* Boxes,
		int32 NumDetections,
		TArray<float>& OutKeypoints) const
	{
		if (NumDetections == 0)
		{
			OutKeypoints.Reset();
			return;
		}

		OutKeypoints.SetNumUninitialized(NumDetections * NumKeypoints * 3);
		float* Out = OutKeypoints.GetData();

		const int32 HeatmapArea = HeatmapSize * HeatmapSize;

		if (KeypointMethod == EKeypointMethod::Fast)
		{
			for (int32 D = 0; D < NumDetections; ++D)
			{
				const float BoxX1 = Boxes[D * 4 + 0];
				const float BoxY1 = Boxes[D * 4 + 1];
				const float BoxW = FMath::Max(Boxes[D * 4 + 2] - BoxX1, 1.0f);
				const float BoxH = FMath::Max(Boxes[D * 4 + 3] - BoxY1, 1.0f);

				for (int32 K = 0; K < NumKeypoints; ++K)
				{
					const float* Hmap = Heatmaps + (D * NumKeypoints + K) * HeatmapArea;

					// Argmax on flat heatmap
					int32 MaxIdx = 0;
					float MaxVal = Hmap[0];
					for (int32 P = 1; P < HeatmapArea; ++P)
					{
						if (Hmap[P] > MaxVal)
						{
							MaxVal = Hmap[P];
							MaxIdx = P;
						}
					}

					const int32 XHmap = MaxIdx % HeatmapSize;
					const int32 YHmap = MaxIdx / HeatmapSize;

					// Map to image coordinates
					const float X = (XHmap + 0.5f) / HeatmapSize * BoxW + BoxX1;
					const float Y = (YHmap + 0.5f) / HeatmapSize * BoxH + BoxY1;

					// Confidence: sigmoid of peak logit
					const float Confidence = 1.0f / (1.0f + FMath::Exp(-MaxVal));

					const int32 OutIdx = (D * NumKeypoints + K) * 3;
					Out[OutIdx + 0] = X;
					Out[OutIdx + 1] = Y;
					Out[OutIdx + 2] = Confidence;
				}
			}
		}
		else // Bicubic
		{
			for (int32 D = 0; D < NumDetections; ++D)
			{
				const float BoxX1 = Boxes[D * 4 + 0];
				const float BoxY1 = Boxes[D * 4 + 1];
				const float BoxW = FMath::Max(Boxes[D * 4 + 2] - BoxX1, 1.0f);
				const float BoxH = FMath::Max(Boxes[D * 4 + 3] - BoxY1, 1.0f);
				const int32 WCeil = FMath::CeilToInt32(BoxW);
				const int32 HCeil = FMath::CeilToInt32(BoxH);
				const float WCorrection = BoxW / WCeil;
				const float HCorrection = BoxH / HCeil;

				for (int32 K = 0; K < NumKeypoints; ++K)
				{
					const float* Hmap = Heatmaps + (D * NumKeypoints + K) * HeatmapArea;

					// Bicubic upsample 56x56 → WCeil x HCeil using FImage
					FImage HmapImage(HeatmapSize, HeatmapSize, ERawImageFormat::R32F);
					FMemory::Memcpy(HmapImage.RawData.GetData(), Hmap, HeatmapArea * sizeof(float));

					FImage UpsampledImage;
					FImageCore::ResizeImageAllocDest(HmapImage, UpsampledImage, WCeil, HCeil,
						ERawImageFormat::R32F, EGammaSpace::Linear,
						FImageCore::EResizeImageFilter::CubicSharp);

					// Argmax on upsampled map
					const float* Upsampled = reinterpret_cast<const float*>(UpsampledImage.RawData.GetData());
					const int32 UpsampledSize = WCeil * HCeil;
					int32 MaxIdx = 0;
					float MaxVal = Upsampled[0];
					for (int32 P = 1; P < UpsampledSize; ++P)
					{
						if (Upsampled[P] > MaxVal)
						{
							MaxVal = Upsampled[P];
							MaxIdx = P;
						}
					}

					const int32 XInt = MaxIdx % WCeil;
					const int32 YInt = MaxIdx / WCeil;

					// Map to image coordinates with correction factor
					const float X = (XInt + 0.5f) * WCorrection + BoxX1;
					const float Y = (YInt + 0.5f) * HCorrection + BoxY1;

					// Confidence: sigmoid of peak logit from original 56x56 heatmap
					float OrigMax = Hmap[0];
					for (int32 P = 1; P < HeatmapArea; ++P)
					{
						OrigMax = FMath::Max(OrigMax, Hmap[P]);
					}
					const float Confidence = 1.0f / (1.0f + FMath::Exp(-OrigMax));

					const int32 OutIdx = (D * NumKeypoints + K) * 3;
					Out[OutIdx + 0] = X;
					Out[OutIdx + 1] = Y;
					Out[OutIdx + 2] = Confidence;
				}
			}
		}
	}

	/**
	 * Maps detection boxes and keypoints from padded/resized coordinates
	 * back to original image space.
	 */
	static void RescaleDetections(
		float* Boxes,
		float* Keypoints,
		int32 NumDetections,
		float ScaleX,
		float ScaleY,
		int32 OrigH,
		int32 OrigW)
	{
		if (NumDetections == 0)
		{
			return;
		}

		const float InvScaleX = 1.0f / ScaleX;
		const float InvScaleY = 1.0f / ScaleY;

		for (int32 D = 0; D < NumDetections; ++D)
		{
			// Rescale boxes
			Boxes[D * 4 + 0] *= InvScaleX;
			Boxes[D * 4 + 1] *= InvScaleY;
			Boxes[D * 4 + 2] *= InvScaleX;
			Boxes[D * 4 + 3] *= InvScaleY;

			// Rescale keypoints (x and y only, confidence unchanged)
			for (int32 K = 0; K < NumKeypoints; ++K)
			{
				const int32 Idx = (D * NumKeypoints + K) * 3;
				Keypoints[Idx + 0] *= InvScaleX;
				Keypoints[Idx + 1] *= InvScaleY;
			}
		}

		// Clip boxes to original image bounds
		ClipBoxes(Boxes, NumDetections, OrigH, OrigW);
	}

	// NNE model instances
	TSharedPtr<UE::NNE::IModelInstanceRunSync> BackboneRpnInstance;
	TSharedPtr<UE::NNE::IModelInstanceRunSync> RoiHeadsBoxInstance;
	TSharedPtr<UE::NNE::IModelInstanceRunSync> RoiHeadsKeypointInstance;

	// Cached input shapes to avoid redundant SetInputTensorShapes calls
	TArray<UE::NNE::FTensorShape> CachedBackboneRpnShapes;
	TArray<UE::NNE::FTensorShape> CachedRoiHeadsBoxShapes;
	TArray<UE::NNE::FTensorShape> CachedRoiHeadsKeypointShapes;

	// Algorithm choice
	EKeypointMethod KeypointMethod;
	EResizeMethod ResizeMethod;

	// Thresholds & filtering
	float BoxScoreThresh;
	float BoxNmsThresh;
	int32 BoxTopKPerImage;
	int32 RpnPreNmsTopK;
	float RpnNmsThresh;
	int32 RpnPostNmsTopK;
};

} // namespace UE::NNE::Segmentation::Detectron2::Private

namespace UE::NNE::Segmentation::Detectron2
{

bool TryLoadModelData(
	UNNEModelData*& BackboneRpnData,
	UNNEModelData*& RoiHeadsBoxData,
	UNNEModelData*& RoiHeadsKeypointData
)
{
	BackboneRpnData = LoadObject<UNNEModelData>(nullptr, Private::BackboneRpnAssetPath);
	RoiHeadsBoxData = LoadObject<UNNEModelData>(nullptr, Private::RoiHeadsBoxAssetPath);
	RoiHeadsKeypointData = LoadObject<UNNEModelData>(nullptr, Private::RoiHeadsKeypointAssetPath);
	return BackboneRpnData && RoiHeadsBoxData && RoiHeadsKeypointData;
}

TUniquePtr<IDetector> Make(
	TConstArrayView<FRuntimePreference> RuntimePreferences,
	const FMakeParams& Params)
{
	UNNEModelData* BackboneRpnData = Params.BackboneRpnData
		? Params.BackboneRpnData
		: LoadObject<UNNEModelData>(nullptr, Private::BackboneRpnAssetPath);
	UNNEModelData* RoiHeadsBoxData = Params.RoiHeadsBoxData
		? Params.RoiHeadsBoxData
		: LoadObject<UNNEModelData>(nullptr, Private::RoiHeadsBoxAssetPath);
	UNNEModelData* RoiHeadsKeypointData = Params.RoiHeadsKeypointData
		? Params.RoiHeadsKeypointData
		: LoadObject<UNNEModelData>(nullptr, Private::RoiHeadsKeypointAssetPath);

	if (!BackboneRpnData || !RoiHeadsBoxData || !RoiHeadsKeypointData)
	{
		UE_LOG(LogDetectron2, Error,
			TEXT("Failed to load Detectron2 model assets. "
				 "BackboneRpn=%s, RoiHeadsBox=%s, RoiHeadsKeypoint=%s"),
			BackboneRpnData ? TEXT("OK") : TEXT("MISSING"),
			RoiHeadsBoxData ? TEXT("OK") : TEXT("MISSING"),
			RoiHeadsKeypointData ? TEXT("OK") : TEXT("MISSING"));
		return nullptr;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> BackboneRpn = NNEUtils::TryCreateInstance(BackboneRpnData, RuntimePreferences);
	if (!BackboneRpn.IsValid())
	{
		UE_LOG(LogDetectron2, Warning, TEXT("Failed to create backbone_rpn instance"));
		return nullptr;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> RoiHeadsBox = NNEUtils::TryCreateInstance(RoiHeadsBoxData, RuntimePreferences);
	if (!RoiHeadsBox.IsValid())
	{
		UE_LOG(LogDetectron2, Warning, TEXT("Failed to create roi_heads_box instance"));
		return nullptr;
	}

	TSharedPtr<UE::NNE::IModelInstanceRunSync> RoiHeadsKeypoint = NNEUtils::TryCreateInstance(RoiHeadsKeypointData, RuntimePreferences);
	if (!RoiHeadsKeypoint.IsValid())
	{
		UE_LOG(LogDetectron2, Warning, TEXT("Failed to create roi_heads_keypoint instance"));
		return nullptr;
	}

	return MakeUnique<Private::FDetectron2>(
		MoveTemp(BackboneRpn),
		MoveTemp(RoiHeadsBox),
		MoveTemp(RoiHeadsKeypoint),
		Params.KeypointMethod,
		Params.ResizeMethod,
		Params.BoxScoreThresh,
		Params.BoxNmsThresh,
		Params.BoxTopKPerImage,
		Params.RpnPreNmsTopK,
		Params.RpnNmsThresh,
		Params.RpnPostNmsTopK);
}

} // namespace UE::NNE::Segmentation::Detectron2
