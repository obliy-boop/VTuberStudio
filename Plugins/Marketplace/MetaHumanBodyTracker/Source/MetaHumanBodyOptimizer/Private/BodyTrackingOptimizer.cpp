// Copyright Epic Games, Inc. All Rights Reserved.

#include "BodyTrackingOptimizer.h"
#include "MetaHumanBodyOptimizerLog.h"
#include "Logging/StructuredLog.h"
#include "MetaHumanBodyTrackerAPI.h"
#include "Interfaces/IPluginManager.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#include <vector>

// SMPLX layout
static constexpr int32 GPoseFloats = 165;
// Hue outputs 10 betas per frame, but the SMPL-X model's blend shape space is larger (typically 400).
// The optimizer pads each frame's betas to the model's full count before calling the Titan API.
static constexpr int32 GInputBetasPerFrame = 10;
static constexpr int32 GTranslationFloats = 3;

// Lower-body joint IDs (aligned with Stage1Objective)
static constexpr int32 GNumLowerBodyJoints = 6;
static constexpr int32 GLowerBodyJointIds[GNumLowerBodyJoints] = { 1, 4, 7, 2, 5, 8 };
static constexpr int32 GExpectedR6dFloats = GNumLowerBodyJoints * 6;

// Keypoint layout (OpenPose BODY_25)
static constexpr int32 GKeypointSlots = 25;
static constexpr int32 GKeypointPackedFloats = GKeypointSlots * 3;
static constexpr int32 GBboxFloats = 4;
static constexpr int32 GContactLogitFloats = 6;

// Default optimization weights — these are the values used for Stage-1 optimization.
// Adjust here to change optimization behavior globally.
static constexpr float GWeightKeypointPosition = 0.5f;
static constexpr float GWeightContactHeight = 10.0f;
static constexpr float GWeightBelowFloor = 100.0f;
static constexpr int32 GMaxIterations = 1500;
static constexpr float GLbfgsAbsDeltaStoppingCriteria = 1e-4f;

static constexpr float GWeightKeypointEdge = 5.0f;
static constexpr float GWeightSmoothness = 0.1f;
static constexpr float GWeightSmoothnessJoint3D = 0.01f;
static constexpr float GWeightPrior = 2.0f;
static constexpr float GWeightGmm = 0.0005f;
static constexpr float GWeightContact = 11.11f;

// no-footlocking values (where different from the foot-locking config weights above)
static constexpr float GNoFootlockingWeightKeypointEdge = 0.0f;
static constexpr float GNoFootlockingWeightSmoothness = 0.25f;
static constexpr float GNoFootlockingWeightSmoothnessJoint3D = 0.0f;
static constexpr float GNoFootlockingWeightPrior = 0.0f;
static constexpr float GNoFootlockingWeightGmm = 0.0f;
static constexpr float GNoFootlockingWeightContact = 1.11f;

struct FOptimizerFrameResult
{
	TArray<float> TranslationDelta;
	TArray<float> LowerBodyR6D;
};

static void Rotation6dToMatrix(const float R6d[6], float R[9])
{
	auto Dot = [](const float* A, const float* B) { return A[0]*B[0] + A[1]*B[1] + A[2]*B[2]; };

	const float A1[3] = { R6d[0], R6d[1], R6d[2] };
	const float A2[3] = { R6d[3], R6d[4], R6d[5] };

	const float N1 = FMath::Sqrt(FMath::Max(1e-20f, Dot(A1, A1)));
	const float B1[3] = { A1[0]/N1, A1[1]/N1, A1[2]/N1 };

	const float Proj = Dot(B1, A2);
	const float A2o[3] = { A2[0]-Proj*B1[0], A2[1]-Proj*B1[1], A2[2]-Proj*B1[2] };
	const float N2 = FMath::Sqrt(FMath::Max(1e-20f, Dot(A2o, A2o)));
	const float B2[3] = { A2o[0]/N2, A2o[1]/N2, A2o[2]/N2 };

	const float B3[3] = { B1[1]*B2[2]-B1[2]*B2[1], B1[2]*B2[0]-B1[0]*B2[2], B1[0]*B2[1]-B1[1]*B2[0] };

	R[0]=B1[0]; R[1]=B2[0]; R[2]=B3[0];
	R[3]=B1[1]; R[4]=B2[1]; R[5]=B3[1];
	R[6]=B1[2]; R[7]=B2[2]; R[8]=B3[2];
}

static void RotationMatrixToAxisAngle(const float R[9], float AA[3])
{
	float CosTheta = FMath::Clamp(0.5f * (R[0] + R[4] + R[8] - 1.0f), -1.0f, 1.0f);
	const float Theta = FMath::Acos(CosTheta);
	if (Theta < 1e-8f) { AA[0] = AA[1] = AA[2] = 0.0f; return; }
	const float SinTheta = FMath::Sin(Theta);
	if (FMath::Abs(SinTheta) < 1e-8f) { AA[0] = AA[1] = AA[2] = 0.0f; return; }
	const float K = Theta / (2.0f * SinTheta);
	AA[0] = K * (R[7] - R[5]);
	AA[1] = K * (R[2] - R[6]);
	AA[2] = K * (R[3] - R[1]);
}

struct FBodyTrackingOptimizerImpl
{
	TITAN_API_NAMESPACE::SMPLXStaticData StaticData;
	TITAN_API_NAMESPACE::CameraParameters Camera;
	TArray<float> PerFrameRcw;
	TArray<float> PerFrameTcw;
	TArray<FOptimizerFrameResult> FrameResults;
	bool bInitialized = false;
	bool bSucceeded = false;
	bool bEnableFootLocking = true;

	template <typename T>
	static void CopyToStdVector(const T* InData, int64 InCount, std::vector<T>& OutVector)
	{
		OutVector.assign(InData, InData + InCount);
	}

	static void CopyInt32ToStdVectorInt(const TArray<int32>& InSource, std::vector<int>& OutVector)
	{
		OutVector.clear();
		OutVector.reserve(static_cast<size_t>(InSource.Num()));
		for (int32 V : InSource)
		{
			OutVector.push_back(static_cast<int>(V));
		}
	}

	TITAN_API_NAMESPACE::OptimizationConfig BuildConfig(float InFps, bool bInEnableFootlocking)
	{
		TITAN_API_NAMESPACE::OptimizationConfig Cfg{};

		Cfg.fps = InFps;
		Cfg.maxIterations = GMaxIterations;
		Cfg.useSparseJacobian = true;
		Cfg.numThreads = -1;
		Cfg.lbfgsAbsDeltaStoppingCriterion = GLbfgsAbsDeltaStoppingCriteria;
		Cfg.keypointPositionWeight = GWeightKeypointPosition;
		Cfg.contactHeightWeight = GWeightContactHeight;
		Cfg.belowFloorWeight = GWeightBelowFloor;


		if (bInEnableFootlocking)
		{
			Cfg.optimizationVariables = {}; // will optimize all parameters
			Cfg.keypointEdgeWeight = GWeightKeypointEdge;
			Cfg.smoothnessWeight = GWeightSmoothness;
			Cfg.smoothnessJoint3DWeight = GWeightSmoothnessJoint3D;
			Cfg.priorWeight = GWeightPrior;
			Cfg.gmmWeight = GWeightGmm;
			Cfg.contactWeight = GWeightContact;
		}
		else
		{
			Cfg.optimizationVariables = { "transl", "cam_height_offset", "cam_init_r6d" }; // do not optimize cam_scale or smplx_pose_r6d_lower_body
			Cfg.keypointEdgeWeight = GNoFootlockingWeightKeypointEdge;
			Cfg.smoothnessWeight = GNoFootlockingWeightSmoothness;
			Cfg.smoothnessJoint3DWeight = GNoFootlockingWeightSmoothnessJoint3D;
			Cfg.priorWeight = GNoFootlockingWeightPrior;
			Cfg.gmmWeight = GNoFootlockingWeightGmm;
			Cfg.contactWeight = GNoFootlockingWeightContact;
		}

		return Cfg;
	}

	TITAN_API_NAMESPACE::FrameKeypointData BuildFrame(const FBodyTrackingOptimizerFrameData& InFrameData, int32 InFrameIdx, int32 InModelNumBetas, int64 InExpectedBboxElems, int64 InExpectedLogitElems)
	{
		TITAN_API_NAMESPACE::FrameKeypointData Frame{};
		Frame.frameIndex = InFrameIdx;
		Frame.personId = 0;

		// Pose
		const int64 PoseBase = static_cast<int64>(InFrameIdx) * GPoseFloats;
		Frame.smplxPose165.resize(GPoseFloats);
		for (int32 i = 0; i < GPoseFloats; ++i)
		{
			Frame.smplxPose165[i] = InFrameData.Pose[PoseBase + i];
		}

		// Shape
		const int64 ShapeBase = static_cast<int64>(InFrameIdx) * GInputBetasPerFrame;
		Frame.smplxBetas10.resize(GInputBetasPerFrame);
		for (int32 i = 0; i < GInputBetasPerFrame; ++i)
		{
			Frame.smplxBetas10[i] = InFrameData.Shape[ShapeBase + i];
		}
		if (InModelNumBetas > 0 && Frame.smplxBetas10.size() < static_cast<size_t>(InModelNumBetas))
		{
			Frame.smplxBetas10.resize(static_cast<size_t>(InModelNumBetas), 0.f);
		}

		// Translation
		const int64 TransBase = static_cast<int64>(InFrameIdx) * GTranslationFloats;
		Frame.smplxTrans3.resize(GTranslationFloats);
		for (int32 i = 0; i < GTranslationFloats; ++i)
		{
			Frame.smplxTrans3[i] = InFrameData.Translation[TransBase + i];
		}

		// Keypoints
		Frame.keypointPositions2D.reserve(GKeypointSlots * 2);
		Frame.keypointWeights.reserve(GKeypointSlots);
		Frame.keypointIds.reserve(GKeypointSlots);
		const int64 KBase = static_cast<int64>(InFrameIdx) * GKeypointPackedFloats;
		for (int32 Ki = 0; Ki < GKeypointSlots; ++Ki)
		{
			const int64 O = KBase + Ki * 3;
			Frame.keypointPositions2D.push_back(InFrameData.Keypoints[O + 0]);
			Frame.keypointPositions2D.push_back(InFrameData.Keypoints[O + 1]);
			Frame.keypointWeights.push_back(InFrameData.Keypoints[O + 2]);
			Frame.keypointIds.push_back(Ki);
		}

		// TODO: Investigate whether zeroing BODY_25 slots 1 and 8 is still needed.
		Frame.keypointPositions2D[1 * 2 + 0] = 0.f;
		Frame.keypointPositions2D[1 * 2 + 1] = 0.f;
		Frame.keypointWeights[1] = 0.f;
		Frame.keypointPositions2D[8 * 2 + 0] = 0.f;
		Frame.keypointPositions2D[8 * 2 + 1] = 0.f;
		Frame.keypointWeights[8] = 0.f;

		// Bounding box
		if (InFrameData.BoundingBoxes.Num() == InExpectedBboxElems)
		{
			const int64 BbBase = static_cast<int64>(InFrameIdx) * GBboxFloats;
			Frame.bbox4.resize(GBboxFloats);
			for (int32 i = 0; i < GBboxFloats; ++i)
			{
				Frame.bbox4[i] = static_cast<float>(InFrameData.BoundingBoxes[BbBase + i]);
			}
		}

		// Contact logits
		if (InFrameData.StaticContactLogits.Num() == InExpectedLogitElems)
		{
			const int64 LBase = static_cast<int64>(InFrameIdx) * GContactLogitFloats;
			Frame.staticConfLogits.resize(GContactLogitFloats);
			for (int32 i = 0; i < GContactLogitFloats; ++i)
			{
				Frame.staticConfLogits[i] = InFrameData.StaticContactLogits[LBase + i];
			}
		}

		// Per-frame camera
		Frame.cameraRcw9.assign(PerFrameRcw.GetData(), PerFrameRcw.GetData() + PerFrameRcw.Num());
		Frame.cameraTcw3.assign(PerFrameTcw.GetData(), PerFrameTcw.GetData() + PerFrameTcw.Num());

		// Frame validity
		if (ensureMsgf(InFrameData.ValidFrame.IsValidIndex(InFrameIdx),
			TEXT("Out of range frame index InFrameIdx=%d"),
			InFrameIdx))
		{
			Frame.valid = InFrameData.ValidFrame[InFrameIdx];
		}

		return Frame;
	}
};

FBodyTrackingOptimizer::FBodyTrackingOptimizer() : Impl(MakeUnique<FBodyTrackingOptimizerImpl>())
{
	
}

FBodyTrackingOptimizer::~FBodyTrackingOptimizer() = default;

bool FBodyTrackingOptimizer::Initialize(TConstArrayView<float> InVerts, TConstArrayView<uint32> InFaces, TConstArrayView<float> InJointReg, TConstArrayView<float> InWeights, TConstArrayView<float> InBlendShapes)
{
	if (InVerts.IsEmpty())
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: SMPL-X vertices array is empty.");
		return false;
	}

	// Hardcoded SMPL-X topology data (kinematic tree, extra joints, landmarks)
	static constexpr int32 KinematicParents[] = {
		-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
		20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21,
		46, 47, 21, 49, 50, 21, 52, 53
	};

	static constexpr int32 ExtraJointIndices[] = {
		9120, 9929, 9448, 616, 6, 5770, 5780, 8846, 8463, 8474, 8635, 5361, 4933, 5058, 5169, 5286, 8079, 7669, 7794, 7905, 8022
	};

	static constexpr int32 LandmarkFaceIndices[] = {
		11837, 9244, 2969, 19771, 19709, 9294, 19837, 2365, 19823, 10721, 19555, 1180, 9214, 19732, 2337, 12916, 19735, 11392,
		11701, 12311, 13108, 9201, 19663, 1653, 19470, 11881, 16133, 11552, 268, 9167, 11211, 12998, 12965, 2457, 19700, 11469,
		885, 917, 11498, 11541, 19731, 2556, 13021, 12993, 9044, 19564, 509, 917, 11524, 19598, 13069
	};

	static constexpr float LandmarkBaryCoords[] = {
		0.45895788f, 0.3616237f, 0.17941843f, 0.53697538f, 0.4573279f, 0.00569675f, 0.22336893f, 0.03614211f, 0.74048895f,
		0.34821984f, 0.20582011f, 0.44596004f, 0.20469385f, 0.35875219f, 0.43655396f, 0.20469385f, 0.43655396f, 0.35875219f,
		0.44596004f, 0.20582011f, 0.34821984f, 0.03614211f, 0.22336893f, 0.74048895f, 0.53697538f, 0.00569675f, 0.4573279f,
		0.17941843f, 0.3616237f, 0.45895788f, 0.0f, 0.97400022f, 0.02599978f, 0.80864847f, 0.01935011f, 0.17200142f,
		0.00399201f, 0.45968971f, 0.5363183f, 0.86735135f, 0.12561925f, 0.00702941f, 0.68098348f, 0.23452331f, 0.08449321f,
		0.60946208f, 0.16802914f, 0.22250876f, 0.00481753f, 0.69911349f, 0.29606897f, 0.22250876f, 0.16802914f, 0.60946208f,
		0.68098348f, 0.08449321f, 0.23452331f, 0.1223315f, 0.17113392f, 0.70653456f, 0.50954217f, 0.09005367f, 0.40040416f,
		0.32565704f, 0.04174463f, 0.63259834f, 0.28644502f, 0.23422983f, 0.47932515f, 0.57774907f, 0.04045029f, 0.38180062f,
		0.01527767f, 0.54909837f, 0.43562394f, 0.47932515f, 0.23422983f, 0.28644502f, 0.32565704f, 0.63259834f, 0.04174463f,
		0.40040416f, 0.09005367f, 0.50954217f, 0.1223315f, 0.70653456f, 0.17113392f, 0.01527767f, 0.43562394f, 0.54909837f,
		0.57773387f, 0.38140291f, 0.04086319f, 0.47117928f, 0.50572842f, 0.02309232f, 0.13617527f, 0.70188785f, 0.16193686f,
		0.1317897f, 0.08670817f, 0.78150213f, 0.00028422f, 0.81925392f, 0.18046187f, 0.13178617f, 0.78151947f, 0.08669433f,
		0.13617527f, 0.16193686f, 0.70188785f, 0.47117928f, 0.02309232f, 0.50572842f, 0.09441352f, 0.9030233f, 0.00256319f,
		0.04700059f, 0.94237906f, 0.01062036f, 0.00010762f, 0.03357921f, 0.96631318f, 0.04700059f, 0.01062036f, 0.94237906f,
		0.00256319f, 0.9030233f, 0.09441352f, 0.0747745f, 0.33583617f, 0.58938932f, 0.03817926f, 0.00537378f, 0.95644695f,
		0.97835124f, 0.02164876f, 0.0f, 0.00537378f, 0.03817926f, 0.95644695f, 0.80937332f, -0.0f, 0.19062671f,
		0.39263141f, 0.57777333f, 0.02959524f, 0.02153543f, 0.97845107f, 1.35e-05f, 0.02959524f, 0.57777333f, 0.39263141f
	};

	Impl->StaticData = TITAN_API_NAMESPACE::SMPLXStaticData{};
	FBodyTrackingOptimizerImpl::CopyToStdVector(InVerts.GetData(), InVerts.Num(), Impl->StaticData.vertices);
	Impl->StaticData.faces.assign(InFaces.GetData(), InFaces.GetData() + InFaces.Num());
	FBodyTrackingOptimizerImpl::CopyToStdVector(InJointReg.GetData(), InJointReg.Num(), Impl->StaticData.jointRegressor);
	FBodyTrackingOptimizerImpl::CopyToStdVector(InWeights.GetData(), InWeights.Num(), Impl->StaticData.weights);
	FBodyTrackingOptimizerImpl::CopyToStdVector(InBlendShapes.GetData(), InBlendShapes.Num(), Impl->StaticData.blendShapes);
	FBodyTrackingOptimizerImpl::CopyInt32ToStdVectorInt(TArray<int32>(KinematicParents, UE_ARRAY_COUNT(KinematicParents)), Impl->StaticData.kintreeParents);
	FBodyTrackingOptimizerImpl::CopyInt32ToStdVectorInt(TArray<int32>(ExtraJointIndices, UE_ARRAY_COUNT(ExtraJointIndices)), Impl->StaticData.extraJointsIdxs);
	FBodyTrackingOptimizerImpl::CopyInt32ToStdVectorInt(TArray<int32>(LandmarkFaceIndices, UE_ARRAY_COUNT(LandmarkFaceIndices)), Impl->StaticData.landmarkFacesIdx);
	FBodyTrackingOptimizerImpl::CopyToStdVector(LandmarkBaryCoords, UE_ARRAY_COUNT(LandmarkBaryCoords), Impl->StaticData.landmarkBaryCoords);

	// Load GMM prior data from plugin content
	{
		const FString GmmPath = FPaths::Combine(
			IPluginManager::Get().FindPlugin(UE_PLUGIN_NAME)->GetContentDir(),
			TEXT("gmmPriorExport.json"));

		TSharedPtr<FJsonObject> GmmJson;
		{
			FString JsonString;
			if (FFileHelper::LoadFileToString(JsonString, *GmmPath))
			{
				TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
				FJsonSerializer::Deserialize(Reader, GmmJson);
			}
		}
		if (GmmJson.IsValid())
		{
			Impl->StaticData.gmmNumGaussians = GmmJson->GetIntegerField(TEXT("num_gaussians"));
			Impl->StaticData.gmmDim = GmmJson->GetIntegerField(TEXT("dim"));

			const TArray<TSharedPtr<FJsonValue>>* MeansArr = nullptr;
			if (GmmJson->TryGetArrayField(TEXT("means"), MeansArr))
			{
				Impl->StaticData.gmmMeans.reserve(MeansArr->Num());
				for (const auto& V : *MeansArr)
				{
					Impl->StaticData.gmmMeans.push_back(static_cast<float>(V->AsNumber()));
				}
			}

			const TArray<TSharedPtr<FJsonValue>>* PrecArr = nullptr;
			if (GmmJson->TryGetArrayField(TEXT("precisions"), PrecArr))
			{
				Impl->StaticData.gmmPrecisions.reserve(PrecArr->Num());
				for (const auto& V : *PrecArr)
				{
					Impl->StaticData.gmmPrecisions.push_back(static_cast<float>(V->AsNumber()));
				}
			}

			const TArray<TSharedPtr<FJsonValue>>* WeightsArr = nullptr;
			if (GmmJson->TryGetArrayField(TEXT("log_nll_weights"), WeightsArr))
			{
				Impl->StaticData.gmmLogNllWeights.reserve(WeightsArr->Num());
				for (const auto& V : *WeightsArr)
				{
					Impl->StaticData.gmmLogNllWeights.push_back(static_cast<float>(V->AsNumber()));
				}
			}
		}
		else
		{
			UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "GMM prior not found at {Path}, optimization will run without GMM regularization.", GmmPath);
		}
	}

	Impl->bInitialized = true;
	return true;
}

void FBodyTrackingOptimizer::SetCamera(float InFocalLength, FVector2f InCenter, float InPitchRad, float InRollRad, const FVector& InCameraWorldPosition)
{
	Impl->Camera = TITAN_API_NAMESPACE::CameraParameters{};
	Impl->Camera.intrinsics = { InFocalLength, 0.f, InCenter.X, 0.f, InFocalLength, InCenter.Y, 0.f, 0.f, 1.f };

	// Build Rcw (world-to-camera rotation) from pitch and roll
	const float Cp = FMath::Cos(-InPitchRad), Sp = FMath::Sin(-InPitchRad);
	const float Cr = FMath::Cos(InRollRad), Sr = FMath::Sin(InRollRad);
	const float Rcw[9] = { Cr, -Sp * Sr, Cp * Sr, 0.f, -Cp, -Sp, Sr, Sp * Cr, -Cp * Cr };
	Impl->PerFrameRcw = { Rcw[0], Rcw[1], Rcw[2], Rcw[3], Rcw[4], Rcw[5], Rcw[6], Rcw[7], Rcw[8] };

	// Compute Tcw = -Rcw * CameraWorldPosition
	const float Cx = static_cast<float>(InCameraWorldPosition.X);
	const float Cy = static_cast<float>(InCameraWorldPosition.Y);
	const float Cz = static_cast<float>(InCameraWorldPosition.Z);
	Impl->PerFrameTcw = {
		-(Rcw[0] * Cx + Rcw[1] * Cy + Rcw[2] * Cz),
		-(Rcw[3] * Cx + Rcw[4] * Cy + Rcw[5] * Cz),
		-(Rcw[6] * Cx + Rcw[7] * Cy + Rcw[8] * Cz)
	};
}

bool FBodyTrackingOptimizer::Run(const FBodyTrackingOptimizerFrameData& InFrameData)
{
	Impl->bSucceeded = false;
	Impl->FrameResults.Reset();

	if (!Impl->bInitialized)
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: not initialized.");
		return false;
	}

	const int32 NumFrames = InFrameData.NumFrames;
	if (NumFrames <= 0
		|| (InFrameData.Pose.Num() / GPoseFloats) != NumFrames
		|| (InFrameData.Translation.Num() / GTranslationFloats) != NumFrames
		|| (InFrameData.Shape.Num() / GInputBetasPerFrame) != NumFrames
		|| (InFrameData.Keypoints.Num() / GKeypointPackedFloats) != NumFrames)
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: buffer length mismatch (NumFrames={NumFrames}).", NumFrames);
		return false;
	}

	TITAN_API_NAMESPACE::MetaHumanBodyTrackerAPI Api;

	if (!Api.InitializeModel(Impl->StaticData))
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: InitializeModel failed.");
		return false;
	}

	if (!Api.SetOptimizationConfig(Impl->BuildConfig(InFrameData.Fps, Impl->bEnableFootLocking)))
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: SetOptimizationConfig failed.");
		return false;
	}

	// Derive beta count from static data (blendShapes encodes shape + pose basis).
	const int32 NumVerts = static_cast<int32>(Impl->StaticData.vertices.size()) / 3;
	const int32 ModelNumBetas = (NumVerts > 0) ? static_cast<int32>(Impl->StaticData.blendShapes.size()) / (NumVerts * 3) - 9 * 54 : GInputBetasPerFrame;
	const int64 ExpectedBboxElems = static_cast<int64>(NumFrames) * GBboxFloats;
	const int64 ExpectedLogitElems = static_cast<int64>(NumFrames) * GContactLogitFloats;

	std::vector<TITAN_API_NAMESPACE::FrameKeypointData> Payload;
	Payload.reserve(static_cast<size_t>(NumFrames));
	for (int32 i = 0; i < NumFrames; ++i)
	{
		Payload.push_back(Impl->BuildFrame(InFrameData, i, ModelNumBetas, ExpectedBboxElems, ExpectedLogitElems));
	}

	if (!Api.SetFrameData(Payload, Impl->Camera, 0))
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: SetFrameData failed.");
		return false;
	}

	if (!Api.OptimizeSequence())
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: OptimizeSequence failed.");
		return false;
	}

	TITAN_API_NAMESPACE::SequenceOptimizationResult TitanResult{};
	if (!Api.GetResults(TitanResult))
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: GetResults failed.");
		return false;
	}

	Impl->FrameResults.Reserve(static_cast<int32>(TitanResult.frameResults.size()));
	for (const TITAN_API_NAMESPACE::FrameOptimizationResult& Fr : TitanResult.frameResults)
	{
		FOptimizerFrameResult Out;
		Out.TranslationDelta.Append(Fr.translationDelta.data(), Fr.translationDelta.size());
		Out.LowerBodyR6D.Append(Fr.lowerBodyR6D.data(), Fr.lowerBodyR6D.size());
		Impl->FrameResults.Add(MoveTemp(Out));
	}
	Impl->bSucceeded = TitanResult.success;
	return Impl->bSucceeded;
}

void FBodyTrackingOptimizer::EnableFootlocking(bool bInEnableFootLocking)
{
	Impl->bEnableFootLocking = bInEnableFootLocking;
}


bool FBodyTrackingOptimizer::ApplyResult(TArray64<float>& InOutPose, TArray64<float>& InOutTranslation, int32 InNumFrames) const
{
	if (!Impl->bSucceeded || Impl->FrameResults.Num() != InNumFrames)
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: ApplyResult skipped (succeeded={Succeeded}, results={Results}, expected={Expected}).", Impl->bSucceeded, Impl->FrameResults.Num(), InNumFrames);
		return false;
	}

	if (InOutPose.Num() != static_cast<int64>(InNumFrames) * GPoseFloats
		|| InOutTranslation.Num() != static_cast<int64>(InNumFrames) * GTranslationFloats)
	{
		UE_LOGFMT(LogMetaHumanBodyOptimizer, Warning, "BodyTrackingOptimizer: ApplyResult buffer size mismatch.");
		return false;
	}

	for (int32 FrameIdx = 0; FrameIdx < InNumFrames; ++FrameIdx)
	{
		const FOptimizerFrameResult& Fr = Impl->FrameResults[FrameIdx];

		if (Fr.TranslationDelta.Num() == GTranslationFloats)
		{
			const int64 Tb = static_cast<int64>(FrameIdx) * GTranslationFloats;
			InOutTranslation[Tb + 0] += Fr.TranslationDelta[0];
			InOutTranslation[Tb + 1] += Fr.TranslationDelta[1];
			InOutTranslation[Tb + 2] += Fr.TranslationDelta[2];
		}

		if (Fr.LowerBodyR6D.Num() == GExpectedR6dFloats)
		{
			const int64 PoseBase = static_cast<int64>(FrameIdx) * GPoseFloats;
			for (int32 J = 0; J < GNumLowerBodyJoints; ++J)
			{
				float Rm[9], AA[3];
				Rotation6dToMatrix(&Fr.LowerBodyR6D[J * 6], Rm);
				RotationMatrixToAxisAngle(Rm, AA);
				const int64 Ja = PoseBase + GLowerBodyJointIds[J] * 3;
				InOutPose[Ja + 0] = AA[0];
				InOutPose[Ja + 1] = AA[1];
				InOutPose[Ja + 2] = AA[2];
			}
		}
	}

	return true;
}
