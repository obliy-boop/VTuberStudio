// Copyright Epic Games, Inc. All Rights Reserved.

#include "MetaHumanBodyDriverActor.h"

#include "SkeletalMeshAttributes.h"
#include "Rendering/SkeletalMeshModel.h"
#include "DynamicMesh/MeshNormals.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Engine/SkeletalMesh.h"
#include "Animation/Skeleton.h"
#include "Engine/StaticMesh.h"
#include "Features/IModularFeatures.h"
#include "Components/SkeletalMeshComponent.h"

#if WITH_EDITOR
#include "FileHelpers.h"
#endif



namespace
{

TAutoConsoleVariable<FString> CVarSaveDebugAssets
{
	TEXT("mh.BodyTracker.SaveDebugAssets"),
	"/Game",
	TEXT("Saves assets for debugging the body animation to the given folder, eg /Game/dump"),
	ECVF_Default
};

TAutoConsoleVariable<int32> CVarShowDebugAssets
{
	TEXT("mh.BodyTracker.ShowDebugAssets"),
	0,
	TEXT("Displays assets for debugging the body animation. Value is a mask of:\n\n"
		 "1 = SMPL results on a dynamic mesh (green)\n"
		 "2 = SMPL results on a skinned skeletal mesh (red)\n"
		 "4 = SMPL results on a stick-figure skeletal mesh (blue)\n"
		 "8 = MH results on the MH archetype (yellow)\n"
		 "16 = MH results on a MH proportioned to the actor driven by a control rig (black)\n"
		 "32 = The origin point sphere\n"
		 "64 = The groundplane"
	),
	ECVF_Default
};

TAutoConsoleVariable<float> CVarDebugAssetsSpacing
{
	TEXT("mh.BodyTracker.DebugAssetsSpacing"),
	50.0,
	TEXT("Sets the spacing between debugging assets"),
	ECVF_Default
};

}

AMetaHumanBodyDriverActor::AMetaHumanBodyDriverActor()
{
	SetRootComponent(CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent")));

	OriginComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("OriginComponent"));
	OriginComponent->SetupAttachment(GetRootComponent());
	OriginComponent->SetRelativeScale3D(FVector(0.1, 0.1, 0.1));

	GroundplaneComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("GroundplaneComponent"));
	GroundplaneComponent->SetupAttachment(GetRootComponent());
	GroundplaneComponent->SetRelativeScale3D(FVector(8, 8, 8));

	SMPLXDynamicMeshComponent = CreateDefaultSubobject<UDynamicMeshComponent>(TEXT("SMPLXDynamicMeshComponent"));
	SMPLXDynamicMeshComponent->SetupAttachment(GetRootComponent());

	SMPLXPoseableMeshSkinnedComponent = CreateDefaultSubobject<UPoseableMeshComponent>(TEXT("SMPLXPoseableMeshSkinnedComponent"));
	SMPLXPoseableMeshSkinnedComponent->SetupAttachment(GetRootComponent());
	SMPLXPoseableMeshSkinnedComponent->SetBoundsScale(10000.0f);
	SMPLXPoseableMeshSkinnedComponent->bNeverDistanceCull = true;

	SMPLXPoseableMeshBareComponent = CreateDefaultSubobject<UPoseableMeshComponent>(TEXT("SMPLXPoseableMeshBareComponent"));
	SMPLXPoseableMeshBareComponent->SetupAttachment(GetRootComponent());
	SMPLXPoseableMeshBareComponent->SetBoundsScale(10000.0f);
	SMPLXPoseableMeshBareComponent->bNeverDistanceCull = true;

	MHPoseableMeshArchetypeBareComponent = CreateDefaultSubobject<UPoseableMeshComponent>(TEXT("MHPoseableMeshArchetypeBareComponent"));
	MHPoseableMeshArchetypeBareComponent->SetupAttachment(GetRootComponent());
	MHPoseableMeshArchetypeBareComponent->SetBoundsScale(10000.0f);
	MHPoseableMeshArchetypeBareComponent->bNeverDistanceCull = true;

	MHPoseableMeshSizedBareComponent = CreateDefaultSubobject<UPoseableMeshComponent>(TEXT("MHPoseableMeshSizedBareComponent"));
	MHPoseableMeshSizedBareComponent->SetupAttachment(GetRootComponent());
	MHPoseableMeshSizedBareComponent->SetBoundsScale(10000.0f);
	MHPoseableMeshSizedBareComponent->bNeverDistanceCull = true;
}

void AMetaHumanBodyDriverActor::PostInitializeComponents()
{
	Super::PostInitializeComponents();

	CachedSMPLXShape = FMetaHumanSMPLX::InvalidShape();

	CVarSaveDebugAssets.AsVariable()->SetOnChangedCallback(FConsoleVariableDelegate::CreateUObject(this, &AMetaHumanBodyDriverActor::SaveDebugAssetsCVarChanged));
	CVarShowDebugAssets.AsVariable()->SetOnChangedCallback(FConsoleVariableDelegate::CreateUObject(this, &AMetaHumanBodyDriverActor::ShowDebugAssetsCVarChanged));
	CVarDebugAssetsSpacing.AsVariable()->SetOnChangedCallback(FConsoleVariableDelegate::CreateUObject(this, &AMetaHumanBodyDriverActor::ShowDebugAssetsCVarChanged));

	OriginComponent->SetStaticMesh(LoadObject<UStaticMesh>(this, TEXT("/Engine/BasicShapes/Sphere.Sphere")));
	GroundplaneComponent->SetStaticMesh(LoadObject<UStaticMesh>(this, TEXT("/Engine/BasicShapes/Plane")));

	UMaterial* Material = LoadObject<UMaterial>(this, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
	Material = DuplicateObject<UMaterial>(Material, this);
	Material->SetUsageByFlag(EMaterialUsage::MATUSAGE_SkeletalMesh, true);

	UMaterialInstanceDynamic* DynamicMaterial = UMaterialInstanceDynamic::Create(Material, this);
	DynamicMaterial->SetVectorParameterValue("Color", FLinearColor::Green);
	SMPLXDynamicMeshComponent->SetMaterial(0, DynamicMaterial);

	SMPLX.Init();
	SMPLXSkeleton = SMPLX.CreateSMPLXSkeleton(this);
	MHSkeleton = SMPLX.GetMHSkeleton(this);

	DynamicMaterial = UMaterialInstanceDynamic::Create(Material, this);
	DynamicMaterial->SetVectorParameterValue("Color", FLinearColor::Red);
	SMPLXPoseableMeshSkinnedComponent->SetMaterial(0, DynamicMaterial);

	DynamicMaterial = UMaterialInstanceDynamic::Create(Material, this);
	DynamicMaterial->SetVectorParameterValue("Color", FLinearColor::Blue);
	SMPLXPoseableMeshBareComponent->SetMaterial(0, DynamicMaterial);

	DynamicMaterial = UMaterialInstanceDynamic::Create(Material, this);
	DynamicMaterial->SetVectorParameterValue("Color", FLinearColor::Yellow);
	MHPoseableMeshArchetypeBareComponent->SetMaterial(0, DynamicMaterial);

	DynamicMaterial = UMaterialInstanceDynamic::Create(Material, this);
	DynamicMaterial->SetVectorParameterValue("Color", FLinearColor(255, 0, 255));
	MHPoseableMeshSizedBareComponent->SetMaterial(0, DynamicMaterial);
}

void AMetaHumanBodyDriverActor::Update(const FFrameAnimationData& InAnimationData)
{
#if WITH_EDITOR
	if (CachedSMPLXShape == FMetaHumanSMPLX::InvalidShape() || (!InAnimationData.RawBodyAnimationSMPLXShape.IsEmpty() && InAnimationData.RawBodyAnimationSMPLXShape != CachedSMPLXShape))
	{
		CachedSMPLXShape = InAnimationData.RawBodyAnimationSMPLXShape;

		// SMPLX Skeleton
		SMPLX.SetAccountForHeight(true);
		SMPLX.SetShape(InAnimationData.RawBodyAnimationSMPLXShape);
		SMPLX.SetPose(TArray<float>());

		// SMPLX Skinned
		SMPLXPoseableMeshSkinnedComponent->SetSkinnedAssetAndUpdate(nullptr);
		SMPLXSkeletalMeshSkinned = SMPLX.CreateSMPLXSkeletalMesh(this, SMPLXSkeleton, true);
		SMPLXPoseableMeshSkinnedComponent->SetSkinnedAssetAndUpdate(SMPLXSkeletalMeshSkinned);

		// SMPLX Bare
		SMPLXPoseableMeshBareComponent->SetSkinnedAssetAndUpdate(nullptr);
		SMPLXSkeletalMeshBare = SMPLX.CreateSMPLXSkeletalMesh(this, SMPLXSkeleton, false);
		SMPLXPoseableMeshBareComponent->SetSkinnedAssetAndUpdate(SMPLXSkeletalMeshBare);

		// MH Archetype Bare
		MHPoseableMeshArchetypeBareComponent->SetSkinnedAssetAndUpdate(nullptr);
		MHSkeletalMeshArchetypeBare = SMPLX.CreateMHSkeletalMesh(this, MHSkeleton, nullptr);
		MHPoseableMeshArchetypeBareComponent->SetSkinnedAssetAndUpdate(MHSkeletalMeshArchetypeBare);

		// MH Sized Bare
		MHPoseableMeshSizedBareComponent->SetSkinnedAssetAndUpdate(nullptr);
		MHSkeletalMeshSizedBare = SMPLX.CreateMHSkeletalMesh(this, MHSkeleton, SMPLXSkeleton);
		MHPoseableMeshSizedBareComponent->SetSkinnedAssetAndUpdate(MHSkeletalMeshSizedBare);

		// Body driver
		BodyDriverSkeletalMesh = SMPLX.CreateMHSkeletalMesh(this, MHSkeleton, SMPLXSkeleton);

		IMetaHumanBodyTrackerInterface& BodyTracker = IModularFeatures::Get().GetModularFeature<IMetaHumanBodyTrackerInterface>(IMetaHumanBodyTrackerInterface::GetModularFeatureName());
		FSoftObjectPath SoftPath(BodyTracker.GetBodyControlRigAssetPath());
		BodyDriverSkeletalMesh->SetDefaultAnimatingRig(TSoftObjectPtr<UObject>(SoftPath));

		ShowDebugAssetsCVarChanged(nullptr);
	}

	SMPLX.SetPose(InAnimationData.RawBodyAnimationSMPLXPose);

	SetDynamicMeshVerts(InAnimationData.RawBodyAnimationSMPLXTranslation);

	SetBoneTransforms(SMPLXPoseableMeshSkinnedComponent, InAnimationData.RawBodyAnimationSMPLXData);
	SetBoneTransforms(SMPLXPoseableMeshBareComponent, InAnimationData.RawBodyAnimationSMPLXData);

	SetBoneTransforms(MHPoseableMeshArchetypeBareComponent, InAnimationData.BodyAnimationData);
	SetBoneTransforms(MHPoseableMeshSizedBareComponent, InAnimationData.BodyAnimationData);
#endif
}

void AMetaHumanBodyDriverActor::SetBodyDriverSkeletalMeshComponent(TWeakObjectPtr<USkeletalMeshComponent> InSkeletalMeshComponent)
{
	BodyDriverSkeletalMeshComponent = InSkeletalMeshComponent;

	ShowDebugAssetsCVarChanged(nullptr);
}

void AMetaHumanBodyDriverActor::SetVisualizationSkeletalMeshParams(bool bInVisible, const FVector& InOffset, const FColor& InColour)
{
	constexpr bool bPropagateToChildren = true;

	FVector Offset; // Account for parent actor rotation
	Offset.X = InOffset.Y;
	Offset.Y = -InOffset.X;
	Offset.Z = InOffset.Z;

	MHPoseableMeshSizedBareComponent->SetVisibility(bInVisible, bPropagateToChildren);
	MHPoseableMeshSizedBareComponent->SetRelativeLocation(Offset);

	UMaterialInterface* MaterialInterface = MHPoseableMeshSizedBareComponent->GetMaterial(0);
	if (MaterialInterface)
	{
		UMaterialInstanceDynamic* DynamicMaterial = Cast<UMaterialInstanceDynamic>(MaterialInterface);
		if (DynamicMaterial)
		{
			DynamicMaterial->SetVectorParameterValue("Color", InColour);
		}
	}
}

TMap<FString, FTransform> AMetaHumanBodyDriverActor::SetBoneTransforms(UPoseableMeshComponent* InPoseableMeshComponent, const TMap<FString, FTransform>& InBoneTransforms)
{
	TMap<FString, FTransform> ComponentBoneTransforms;

	TArray<FName> BoneNames;
	InPoseableMeshComponent->GetBoneNames(BoneNames);
	for (const FName& Bone : BoneNames)
	{
		InPoseableMeshComponent->ResetBoneTransformByName(Bone);
	}

	for (const TPair<FString, FTransform>& BoneTransformPair : InBoneTransforms)
	{
		const FName ParentBone = InPoseableMeshComponent->GetParentBone(FName(BoneTransformPair.Key));

		FTransform ParentComponentTransform;
		if (ParentBone.IsNone())
		{
			ParentComponentTransform = FTransform::Identity;
		}
		else
		{
			ParentComponentTransform = InPoseableMeshComponent->GetBoneTransformByName(ParentBone, EBoneSpaces::ComponentSpace);
		}

		const FTransform BoneComponentTransform = BoneTransformPair.Value * ParentComponentTransform;
		InPoseableMeshComponent->SetBoneTransformByName(FName(BoneTransformPair.Key), BoneComponentTransform, EBoneSpaces::ComponentSpace);

		ComponentBoneTransforms.Add(BoneTransformPair.Key, BoneComponentTransform);
	}

	InPoseableMeshComponent->RefreshBoneTransforms();
	InPoseableMeshComponent->MarkRenderDynamicDataDirty();
	InPoseableMeshComponent->MarkRenderStateDirty();

	return ComponentBoneTransforms;
}

void AMetaHumanBodyDriverActor::SetDynamicMeshVerts(const FVector& InTranslation)
{
	FDynamicMesh3& Mesh = SMPLXDynamicMeshComponent->GetDynamicMesh()->GetMeshRef();

	Mesh.Clear();

	if (!Mesh.HasAttributes())
	{
		Mesh.EnableAttributes();
	}

	SMPLX.SetAccountForHeight(false);

	const TArray<FVector>& SkinnedVerts = SMPLX.GetSkinnedVerticesUE();
	const TArray<TTuple<uint32, uint32, uint32>>& Faces = SMPLX.GetFaces();
	TMap<int32, int32> VertexID;

	for (int32 Vert = 0; Vert < SkinnedVerts.Num(); ++Vert)
	{
		VertexID.Add(Vert, Mesh.AppendVertex(FVector3d(SkinnedVerts[Vert] + InTranslation)));
	}

	for (int32 Face = 0; Face < Faces.Num(); ++Face)
	{
		const TTuple<uint32, uint32, uint32>& FaceVertices = Faces[Face];
		Mesh.AppendTriangle(VertexID[FaceVertices.Get<0>()], VertexID[FaceVertices.Get<1>()], VertexID[FaceVertices.Get<2>()]);
	}

	UE::Geometry::FDynamicMeshNormalOverlay* NormalOverlay = Mesh.Attributes()->PrimaryNormals();
	UE::Geometry::FMeshNormals::InitializeOverlayToPerVertexNormals(NormalOverlay);

	SMPLXDynamicMeshComponent->NotifyMeshUpdated();
}

void AMetaHumanBodyDriverActor::SaveDebugAssetsCVarChanged(IConsoleVariable* InVariable)
{
	SaveDebugAssets(CVarSaveDebugAssets.GetValueOnAnyThread());
}

bool AMetaHumanBodyDriverActor::SaveDebugAssets(const FString& InPath)
{
#if WITH_EDITOR
//	UPackage* SMPLXSkeletonPackage = CreatePackage(*(InPath + "/" + SMPLXSkeleton->GetName())); // Skeleton save may need introducing if this is dynamically created rather than a loaded asset
	UPackage* SMPLXSkeletalMeshSkinnedPackage = CreatePackage(*(InPath + "/" + SMPLXSkeletalMeshSkinned->GetName()));
	UPackage* SMPLXSkeletalMeshBarePackage = CreatePackage(*(InPath + "/" + SMPLXSkeletalMeshBare->GetName()));
	UPackage* MHSkeletalMeshSizedBarePackage = CreatePackage(*(InPath + "/" + MHSkeletalMeshSizedBare->GetName()));

	TArray<UPackage*> PackagesToSave;
//	PackagesToSave.Add(SMPLXSkeletonPackage);
	PackagesToSave.Add(SMPLXSkeletalMeshSkinnedPackage);
	PackagesToSave.Add(SMPLXSkeletalMeshBarePackage);
	PackagesToSave.Add(MHSkeletalMeshSizedBarePackage);

//	USkeleton* SMPLXSkeletonCopy = DuplicateObject<USkeleton>(SMPLXSkeleton, SMPLXSkeletonPackage);
	USkeletalMesh* SMPLXSkeletalMeshSkinnedCopy = DuplicateObject<USkeletalMesh>(SMPLXSkeletalMeshSkinned, SMPLXSkeletalMeshSkinnedPackage);
	USkeletalMesh* SMPLXSkeletalMeshBareCopy = DuplicateObject<USkeletalMesh>(SMPLXSkeletalMeshBare, SMPLXSkeletalMeshBarePackage);
	USkeletalMesh* MHSkeletalMeshSizedBareCopy = DuplicateObject<USkeletalMesh>(MHSkeletalMeshSizedBare, MHSkeletalMeshSizedBarePackage);

//	SMPLXSkeletonCopy->SetFlags(RF_Public | RF_Standalone);
	SMPLXSkeletalMeshSkinnedCopy->SetFlags(RF_Public | RF_Standalone);
	SMPLXSkeletalMeshBareCopy->SetFlags(RF_Public | RF_Standalone);
	MHSkeletalMeshSizedBareCopy->SetFlags(RF_Public | RF_Standalone);

//	SMPLXSkeletalMeshSkinnedCopy->SetSkeleton(SMPLXSkeletonCopy);
//	SMPLXSkeletalMeshSkinnedCopy->SetRefSkeleton(SMPLXSkeletonCopy->GetReferenceSkeleton());

//	SMPLXSkeletalMeshBareCopy->SetSkeleton(SMPLXSkeletonCopy);
//	SMPLXSkeletalMeshBareCopy->SetRefSkeleton(SMPLXSkeletonCopy->GetReferenceSkeleton());

//	SMPLXSkeletonCopy->Modify();
//	SMPLXSkeletonCopy->MarkPackageDirty();
//	SMPLXSkeletonPackage->MarkPackageDirty();

	SMPLXSkeletalMeshSkinnedCopy->Modify();
	SMPLXSkeletalMeshSkinnedCopy->MarkPackageDirty();
	SMPLXSkeletalMeshSkinnedPackage->MarkPackageDirty();

	SMPLXSkeletalMeshBareCopy->Modify();
	SMPLXSkeletalMeshBareCopy->MarkPackageDirty();
	SMPLXSkeletalMeshBarePackage->MarkPackageDirty();

	MHSkeletalMeshSizedBareCopy->Modify();
	MHSkeletalMeshSizedBareCopy->MarkPackageDirty();
	MHSkeletalMeshSizedBarePackage->MarkPackageDirty();

	return UEditorLoadingAndSavingUtils::SavePackages(PackagesToSave, false);
#else
	return false;
#endif
}

void AMetaHumanBodyDriverActor::ShowDebugAssetsCVarChanged(IConsoleVariable* InVariable)
{
	const int32 bVisibility = CVarShowDebugAssets.GetValueOnAnyThread();
	constexpr bool bPropagateToChildren = true;

	int32 Mask = 1;
	const float Spacing = CVarDebugAssetsSpacing.GetValueOnAnyThread();
	float X = Spacing;

	TArray<USceneComponent*> Components;
	Components.Add(SMPLXDynamicMeshComponent);
	Components.Add(SMPLXPoseableMeshSkinnedComponent);
	Components.Add(SMPLXPoseableMeshBareComponent);
	Components.Add(MHPoseableMeshArchetypeBareComponent);
	Components.Add(BodyDriverSkeletalMeshComponent.Get());

	for (USceneComponent* Component : Components)
	{
		Mask <<= 1;

		if (!Component)
		{
			continue;
		}

		if (bVisibility & (Mask >> 1))
		{
			Component->SetVisibility(true, bPropagateToChildren);
			Component->SetRelativeLocation(FVector(X, 0, 0));

			X += Spacing;
		}
		else
		{
			Component->SetVisibility(false, bPropagateToChildren);
		}
	}

	Components.Reset();
	Components.Add(OriginComponent);
	Components.Add(GroundplaneComponent);

	for (USceneComponent* Component : Components)
	{
		Mask <<= 1;

		if (!Component)
		{
			continue;
		}

		Component->SetVisibility(bVisibility & (Mask >> 1), bPropagateToChildren);
	}
}
