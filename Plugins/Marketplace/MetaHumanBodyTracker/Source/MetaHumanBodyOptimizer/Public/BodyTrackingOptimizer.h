// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Math/Vector2D.h"
#include "Math/Vector.h"

/**
 * Raw frame data passed into the optimizer. Views into caller-owned flat arrays.
 */
struct FBodyTrackingOptimizerFrameData
{
	TConstArrayView64<float> Pose;
	TConstArrayView64<float> Translation;
	TConstArrayView64<float> Shape;
	TConstArrayView64<float> Keypoints;
	TConstArrayView64<float> BoundingBoxes;
	TConstArrayView64<float> StaticContactLogits;
	TConstArrayView64<bool> ValidFrame;
	int32 NumFrames = 0;
	float Fps = 30.0f;
};

/**
 * Self-contained Stage-1 body sequence optimizer (MetaHumanBodyTrackerAPI / Titan).
 *
 * Usage:
 *   FBodyTrackingOptimizer Optimizer;
 *   Optimizer.Initialize(Verts, Faces, JointReg, Weights, BlendShapes);
 *   Optimizer.SetCamera(...);
 *   if (Optimizer.Run(FrameData)) { Optimizer.ApplyResult(Pose, Translation, NumFrames); }
 */
class METAHUMANBODYOPTIMIZER_API FBodyTrackingOptimizer
{
public:
	FBodyTrackingOptimizer();
	~FBodyTrackingOptimizer();

	FBodyTrackingOptimizer(const FBodyTrackingOptimizer&) = delete;
	FBodyTrackingOptimizer& operator=(const FBodyTrackingOptimizer&) = delete;

	/** Initialize with raw SMPL-X model arrays. Topology data (kinematic tree, landmarks) is loaded internally. */
	bool Initialize(TConstArrayView<float> InVerts, TConstArrayView<uint32> InFaces, TConstArrayView<float> InJointReg, TConstArrayView<float> InWeights, TConstArrayView<float> InBlendShapes);

	/** Set camera intrinsics and camera world position for the optimizer. Tcw is derived internally from Rcw and the world position. */
	void SetCamera(float InFocalLength, FVector2f InCameraImgCenter, float InPitchRadians, float InRollRadians, const FVector& InCameraWorldPosition);

	/** Set whether or not foot-locking should be used (by default, footlocking is used)*/
	void EnableFootlocking(bool bInEnableFootLocking);

	/** Build frame data, run optimization, and store results. Returns true on success. */
	bool Run(const FBodyTrackingOptimizerFrameData& InFrameData);

	/** Apply optimization results back to the pose and translation arrays. */
	bool ApplyResult(TArray64<float>& InOutPose, TArray64<float>& InOutTranslation, int32 InNumFrames) const;

private:
	TUniquePtr<struct FBodyTrackingOptimizerImpl> Impl;
};
