// Copyright Epic Games, Inc. All Rights Reserved.

#include "MetaHumanResize.h"

#include "Math/Vector.h"

#include "SupressWarnings.h"

MH_DISABLE_EIGEN_WARNINGS
#include "Eigen/Dense"
MH_ENABLE_WARNINGS



namespace UE::MetaHuman::BodyTracker
{

extern TMap<FString, Eigen::Vector3d> reproportion(const Eigen::MatrixXd& joints, double beta0);
extern const TMap<FString, FString> PARENT_MAP;

TMap<FString, TPair<FString, FVector>> FMetaHumanResize::Resize(float InBeta0, const TArray<FVector>& InGlobalSMPLJointPosition)
{
	check(InGlobalSMPLJointPosition.Num() == 55);

	Eigen::MatrixXd Joints(55, 3);

	for (int32 JointIndex = 0; JointIndex < 55; ++JointIndex)
	{
		Joints(JointIndex, 0) = InGlobalSMPLJointPosition[JointIndex][0];
		Joints(JointIndex, 1) = InGlobalSMPLJointPosition[JointIndex][1];
		Joints(JointIndex, 2) = InGlobalSMPLJointPosition[JointIndex][2];
	}

	TMap<FString, TPair<FString, FVector>> GlobalMHJointPosition;

	for (const TPair<FString, Eigen::Vector3d>& Item : reproportion(Joints, InBeta0))
	{
		GlobalMHJointPosition.Add(Item.Key, TPair<FString, FVector>(PARENT_MAP[Item.Key], FVector(Item.Value(0), Item.Value(1), Item.Value(2))));
	}

	return GlobalMHJointPosition;
}

// Everything below here is a machine translation from the source Python file

/**
 * smplx_to_mh_skeleton_simple.cpp — Standalone SMPL-X → MetaHuman skeleton reproportioner.
 *
 * Single-file port of the Python version. Requires Eigen3 and nlohmann/json libraries.
 *
 * Algorithm overview (PROP-19):
 * Given the SMPL-X neutral model's 55 joint world positions for a particular β₀
 * value, this converter produces world-space head positions for ~82 MH bones (Y-up, metres).
 *
 * Usage:
 *     smplx_to_mh_skeleton_simple <in.json> <out.json>
 *
 * Input JSON:
 *     {"beta0": 0.0, "joints": [[x,y,z], ...]}
 *
 * Output JSON:
 *     {
 *       "root":     {"pos": [0,0,0], "parent_name": null},
 *       "pelvis":   {"pos": [x,y,z], "parent_name": "root"},
 *       ...
 *     }
 */

using Vec3 = Eigen::Vector3d;
using MatrixXd = Eigen::MatrixXd;

// ---------------------------------------------------------------------------
// SMPL-X joint constants
// ---------------------------------------------------------------------------

static const TMap<int32, FString> SMPLX_JOINT_NAMES = {
    {0, TEXT("pelvis")}, {1, TEXT("left_hip")}, {2, TEXT("right_hip")}, {3, TEXT("spine1")},
    {4, TEXT("left_knee")}, {5, TEXT("right_knee")}, {6, TEXT("spine2")},
    {7, TEXT("left_ankle")}, {8, TEXT("right_ankle")}, {9, TEXT("spine3")},
    {10, TEXT("left_foot")}, {11, TEXT("right_foot")}, {12, TEXT("neck")},
    {13, TEXT("left_collar")}, {14, TEXT("right_collar")}, {15, TEXT("head")},
    {16, TEXT("left_shoulder")}, {17, TEXT("right_shoulder")},
    {18, TEXT("left_elbow")}, {19, TEXT("right_elbow")},
    {20, TEXT("left_wrist")}, {21, TEXT("right_wrist")}
};

static const TMap<FString, FString> MH_OF_SMPLX = {
    {TEXT("pelvis"), TEXT("pelvis")}, {TEXT("left_hip"), TEXT("thigh_l")}, {TEXT("right_hip"), TEXT("thigh_r")},
    {TEXT("spine1"), TEXT("spine_01")}, {TEXT("left_knee"), TEXT("calf_l")}, {TEXT("right_knee"), TEXT("calf_r")},
    {TEXT("spine2"), TEXT("spine_02")}, {TEXT("left_ankle"), TEXT("foot_l")}, {TEXT("right_ankle"), TEXT("foot_r")},
    {TEXT("spine3"), TEXT("spine_03")}, {TEXT("left_foot"), TEXT("ball_l")}, {TEXT("right_foot"), TEXT("ball_r")},
    {TEXT("neck"), TEXT("neck_01")}, {TEXT("left_collar"), TEXT("clavicle_l")}, {TEXT("right_collar"), TEXT("clavicle_r")},
    {TEXT("head"), TEXT("head")}, {TEXT("left_shoulder"), TEXT("upperarm_l")}, {TEXT("right_shoulder"), TEXT("upperarm_r")},
    {TEXT("left_elbow"), TEXT("lowerarm_l")}, {TEXT("right_elbow"), TEXT("lowerarm_r")},
    {TEXT("left_wrist"), TEXT("hand_l")}, {TEXT("right_wrist"), TEXT("hand_r")}
};

// Segments: (label, parent_idx, child_idx)
static const TArray<TTuple<FString, int32, int32>> SEGMENTS = {
    {TEXT("pelvis_to_spine1"), 0, 3}, {TEXT("spine1_to_spine2"), 3, 6},
    {TEXT("spine2_to_spine3"), 6, 9}, {TEXT("spine3_to_neck"), 9, 12},
    {TEXT("neck_to_head"), 12, 15}, {TEXT("l_collar"), 9, 13},
    {TEXT("l_shoulder"), 13, 16}, {TEXT("l_upperarm"), 16, 18},
    {TEXT("l_forearm"), 18, 20}, {TEXT("r_collar"), 9, 14},
    {TEXT("r_shoulder"), 14, 17}, {TEXT("r_upperarm"), 17, 19},
    {TEXT("r_forearm"), 19, 21}, {TEXT("l_pelvis_to_hip"), 0, 1},
    {TEXT("l_thigh"), 1, 4}, {TEXT("l_shin"), 4, 7},
    {TEXT("l_foot"), 7, 10}, {TEXT("r_pelvis_to_hip"), 0, 2},
    {TEXT("r_thigh"), 2, 5}, {TEXT("r_shin"), 5, 8},
    {TEXT("r_foot"), 8, 11}
};

static const TArray<TPair<FString, FString>> MIRROR_PAIRS = {
    {TEXT("l_collar"), TEXT("r_collar")}, {TEXT("l_shoulder"), TEXT("r_shoulder")},
    {TEXT("l_upperarm"), TEXT("r_upperarm")}, {TEXT("l_forearm"), TEXT("r_forearm")},
    {TEXT("l_pelvis_to_hip"), TEXT("r_pelvis_to_hip")}, {TEXT("l_thigh"), TEXT("r_thigh")},
    {TEXT("l_shin"), TEXT("r_shin")}, {TEXT("l_foot"), TEXT("r_foot")}
};

// ---------------------------------------------------------------------------
// H3 ratio fits (slope, intercept) per segment
// ---------------------------------------------------------------------------

static const TMap<FString, TPair<double, double>> H3_RATIOS = {
    {TEXT("pelvis_to_spine1"), {0.05499685455503195, 0.32684473851137424}},
    {TEXT("spine1_to_spine2"), {0.060412894609392476, 0.49687561415308473}},
    {TEXT("spine2_to_spine3"), {0.05440947650638659, 1.2001799466044654}},
    {TEXT("spine3_to_neck"), {-0.0022613394873957303, 2.2477168129889704}},
    {TEXT("neck_to_head"), {0.02314343491584826, 0.6497851000757491}},
    {TEXT("l_collar"), {-0.007843618317028778, 3.2989868287839164}},
    {TEXT("l_shoulder"), {-0.005675422684781784, 1.3303543204884873}},
    {TEXT("l_upperarm"), {-0.033328742645558886, 0.8642802786225061}},
    {TEXT("l_forearm"), {0.009743943496472895, 1.0290117661836098}},
    {TEXT("r_collar"), {-0.007843618317028778, 3.2989868287839164}},
    {TEXT("r_shoulder"), {-0.005675422684781784, 1.3303543204884873}},
    {TEXT("r_upperarm"), {-0.033328742645558886, 0.8642802786225061}},
    {TEXT("r_forearm"), {0.009743943496472895, 1.0290117661836098}},
    {TEXT("l_pelvis_to_hip"), {-0.01789588280962314, 0.8249256962862566}},
    {TEXT("l_thigh"), {-0.008967864048919316, 1.1137123837232124}},
    {TEXT("l_shin"), {-0.010281430080966969, 0.9573136595114576}},
    {TEXT("l_foot"), {-0.0012648774268518182, 1.1091021780444692}},
    {TEXT("r_pelvis_to_hip"), {-0.01789588280962314, 0.8249256962862566}},
    {TEXT("r_thigh"), {-0.008967864048919316, 1.1137123837232124}},
    {TEXT("r_shin"), {-0.010281430080966969, 0.9573136595114576}},
    {TEXT("r_foot"), {-0.0012648774268518182, 1.1091021780444692}}
};

static const TPair<double, double> ROOT_TO_PELVIS_CHAIN_RATIO = {
    -0.013812083503682881, 0.8620840023883989
};

// ---------------------------------------------------------------------------
// Finger chain constants
// ---------------------------------------------------------------------------

static const TArray<TPair<FString, FString>> FINGER_MIRROR_PAIRS = {
    {TEXT("index_l"), TEXT("index_r")}, {TEXT("middle_l"), TEXT("middle_r")},
    {TEXT("ring_l"), TEXT("ring_r")}, {TEXT("pinky_l"), TEXT("pinky_r")},
    {TEXT("thumb_l"), TEXT("thumb_r")}
};

// SMPL-X finger chains: (label, wrist_idx, phalanx_indices)
static const TArray<TTuple<FString, int32, TArray<int32>>> SMPLX_FINGER_CHAINS = {
    {TEXT("index_l"), 20, TArray<int32>({25, 26, 27})}, {TEXT("middle_l"), 20, TArray<int32>({28, 29, 30})},
    {TEXT("ring_l"), 20, TArray<int32>({31, 32, 33})}, {TEXT("pinky_l"), 20, TArray<int32>({34, 35, 36})},
    {TEXT("thumb_l"), 20, TArray<int32>({37, 38, 39})}, {TEXT("index_r"), 21, TArray<int32>({40, 41, 42})},
    {TEXT("middle_r"), 21, TArray<int32>({43, 44, 45})}, {TEXT("ring_r"), 21, TArray<int32>({46, 47, 48})},
    {TEXT("pinky_r"), 21, TArray<int32>({49, 50, 51})}, {TEXT("thumb_r"), 21, TArray<int32>({52, 53, 54})}
};

static const TMap<FString, TPair<FString, TArray<FString>>> MH_FINGER_CHAINS = {
    {TEXT("index_l"), {TEXT("hand_l"), TArray<FString>({TEXT("index_metacarpal_l"), TEXT("index_01_l"), TEXT("index_02_l"), TEXT("index_03_l")})}},
    {TEXT("middle_l"), {TEXT("hand_l"), TArray<FString>({TEXT("middle_metacarpal_l"), TEXT("middle_01_l"), TEXT("middle_02_l"), TEXT("middle_03_l")})}},
    {TEXT("ring_l"), {TEXT("hand_l"), TArray<FString>({TEXT("ring_metacarpal_l"), TEXT("ring_01_l"), TEXT("ring_02_l"), TEXT("ring_03_l")})}},
    {TEXT("pinky_l"), {TEXT("hand_l"), TArray<FString>({TEXT("pinky_metacarpal_l"), TEXT("pinky_01_l"), TEXT("pinky_02_l"), TEXT("pinky_03_l")})}},
    {TEXT("thumb_l"), {TEXT("hand_l"), TArray<FString>({TEXT("thumb_01_l"), TEXT("thumb_02_l"), TEXT("thumb_03_l")})}},
    {TEXT("index_r"), {TEXT("hand_r"), TArray<FString>({TEXT("index_metacarpal_r"), TEXT("index_01_r"), TEXT("index_02_r"), TEXT("index_03_r")})}},
    {TEXT("middle_r"), {TEXT("hand_r"), TArray<FString>({TEXT("middle_metacarpal_r"), TEXT("middle_01_r"), TEXT("middle_02_r"), TEXT("middle_03_r")})}},
    {TEXT("ring_r"), {TEXT("hand_r"), TArray<FString>({TEXT("ring_metacarpal_r"), TEXT("ring_01_r"), TEXT("ring_02_r"), TEXT("ring_03_r")})}},
    {TEXT("pinky_r"), {TEXT("hand_r"), TArray<FString>({TEXT("pinky_metacarpal_r"), TEXT("pinky_01_r"), TEXT("pinky_02_r"), TEXT("pinky_03_r")})}},
    {TEXT("thumb_r"), {TEXT("hand_r"), TArray<FString>({TEXT("thumb_01_r"), TEXT("thumb_02_r"), TEXT("thumb_03_r")})}}
};

static const TMap<FString, TPair<double, double>> FINGER_H3 = {
    {TEXT("index_l"), {-2.9275638002050988, 107.46801369220275}},
    {TEXT("middle_l"), {-2.993044973056973, 104.43194885401736}},
    {TEXT("ring_l"), {-4.212471939964715, 119.15097170754343}},
    {TEXT("pinky_l"), {-3.2590556096964387, 90.11244863037086}},
    {TEXT("thumb_l"), {-3.815342102761391, 112.03744229537605}},
    {TEXT("index_r"), {-2.9275638002050988, 107.46801369220275}},
    {TEXT("middle_r"), {-2.993044973056973, 104.43194885401736}},
    {TEXT("ring_r"), {-4.212471939964715, 119.15097170754343}},
    {TEXT("pinky_r"), {-3.2590556096964387, 90.11244863037086}},
    {TEXT("thumb_r"), {-3.815342102761391, 112.03744229537605}}
};

// ---------------------------------------------------------------------------
// Foot chain constants
// ---------------------------------------------------------------------------

static const TArray<TPair<FString, FString>> FOOT_MIRROR_PAIRS = {
    {TEXT("foot_l"), TEXT("foot_r")}
};

static const TArray<TTuple<FString, int32, int32>> SMPLX_FOOT_REFS = {
    {TEXT("foot_l"), 7, 10}, {TEXT("foot_r"), 8, 11}
};

static const TMap<FString, TPair<FString, FString>> MH_FOOT_CHAINS = {
    {TEXT("foot_l"), {TEXT("foot_l"), TEXT("ball_l")}}, {TEXT("foot_r"), {TEXT("foot_r"), TEXT("ball_r")}}
};

static const TMap<FString, TPair<double, double>> FOOT_H3 = {
    {TEXT("foot_l"), {-0.9587908026049055, 134.5512577759512}},
    {TEXT("foot_r"), {-0.9587908026049055, 134.5512577759512}}
};

static const TMap<FString, TArray<FString>> FOOT_CHAIN_BONES = {
    {TEXT("foot_l"), {TEXT("ball_l"), TEXT("bigtoe_01_l"), TEXT("bigtoe_02_l"), TEXT("indextoe_01_l"), TEXT("indextoe_02_l"),
                TEXT("middletoe_01_l"), TEXT("middletoe_02_l"), TEXT("ringtoe_01_l"), TEXT("ringtoe_02_l"),
                TEXT("littletoe_01_l"), TEXT("littletoe_02_l")}},
    {TEXT("foot_r"), {TEXT("ball_r"), TEXT("bigtoe_01_r"), TEXT("bigtoe_02_r"), TEXT("indextoe_01_r"), TEXT("indextoe_02_r"),
                TEXT("middletoe_01_r"), TEXT("middletoe_02_r"), TEXT("ringtoe_01_r"), TEXT("ringtoe_02_r"),
                TEXT("littletoe_01_r"), TEXT("littletoe_02_r")}}
};

// ---------------------------------------------------------------------------
// Baseline MH skeleton positions (Y-up, metres)
// Format: {bone_name: {head_x, head_y, head_z, tail_x, tail_y, tail_z}}
// ---------------------------------------------------------------------------

struct BoneBaseline {
    Vec3 head;
    Vec3 tail;
};

// Baseline data - abbreviated for readability, include full data in actual implementation
static const TMap<FString, TTuple<double, double, double, double, double, double>> BASELINE_MH_YUP_M_RAW = {
    {TEXT("ball_l"), {0.15558399, 0.01053916, 0.13698184, 0.12871177, 0.02211656, 0.15134460}},
    {TEXT("ball_r"), {-0.15558402, 0.01053919, 0.13698178, -0.12871181, 0.02211660, 0.15134450}},
    {TEXT("bigtoe_01_l"), {0.12871177, 0.02211656, 0.15134460, 0.13634117, 0.01779939, 0.18155754}},
    {TEXT("bigtoe_01_r"), {-0.12871181, 0.02211660, 0.15134450, -0.13634124, 0.01779943, 0.18155743}},
    {TEXT("bigtoe_02_l"), {0.13634117, 0.01779939, 0.18155754, 0.13634117, 0.01779939, 0.18135754}},
    {TEXT("bigtoe_02_r"), {-0.13634124, 0.01779943, 0.18155743, -0.13634124, 0.01779943, 0.18135743}},
    {TEXT("calf_l"), {0.11507245, 0.47199810, 0.01686990, 0.11507245, 0.47199810, 0.01686990}},
    {TEXT("calf_r"), {-0.11507246, 0.47199818, 0.01686982, -0.11507246, 0.47199818, 0.01686982}},
    {TEXT("clavicle_l"), {0.01153184, 1.38623248, -0.01647334, 0.13049355, 1.46067570, -0.01722400}},
    {TEXT("clavicle_r"), {-0.01153181, 1.38623300, -0.01647335, -0.13024680, 1.45963624, -0.01741250}},
    {TEXT("foot_l"), {0.13234060, 0.08078157, 0.00179463, 0.12210734, 0.09065165, -0.03918629}},
    {TEXT("foot_r"), {-0.13234061, 0.08078165, 0.00179449, -0.12210733, 0.09065177, -0.03918642}},
    {TEXT("hand_l"), {0.65825274, 1.38366490, -0.05090872, 0.68964298, 1.37452736, -0.03216816}},
    {TEXT("hand_r"), {-0.65949732, 1.34703434, -0.05750233, -0.69112351, 1.33423784, -0.04149942}},
    {TEXT("head"), {0.00000001, 1.55001874, 0.00948672, 0.00000001, 1.55001874, 0.00928672}},
    {TEXT("index_01_l"), {0.74122093, 1.37106740, -0.02758056, 0.76964969, 1.36215891, -0.02874625}},
    {TEXT("index_01_r"), {-0.74223260, 1.32523441, -0.04118455, -0.77000175, 1.31454254, -0.04305136}},
    {TEXT("index_02_l"), {0.78340847, 1.37409317, -0.02848583, 0.79475976, 1.36392623, -0.03055071}},
    {TEXT("index_02_r"), {-0.78449387, 1.32558128, -0.04309622, -0.79514130, 1.31475138, -0.04550841}},
    {TEXT("index_03_l"), {0.80824181, 1.37134569, -0.03124045, 0.81632419, 1.36243270, -0.03329347}},
    {TEXT("index_03_r"), {-0.80921859, 1.32130798, -0.04661017, -0.81668089, 1.31192789, -0.04891272}},
    {TEXT("index_metacarpal_l"), {0.68964298, 1.37452736, -0.03216816, 0.74122093, 1.37106740, -0.02758056}},
    {TEXT("index_metacarpal_r"), {-0.69112351, 1.33423784, -0.04149942, -0.74223260, 1.32523441, -0.04118455}},
    {TEXT("indextoe_01_l"), {0.15434497, 0.02542550, 0.14516263, 0.16344254, 0.01840239, 0.17351186}},
    {TEXT("indextoe_01_r"), {-0.15434501, 0.02542553, 0.14516257, -0.16344258, 0.01840242, 0.17351183}},
    {TEXT("indextoe_02_l"), {0.16344254, 0.01840239, 0.17351186, 0.16344254, 0.01840239, 0.17331186}},
    {TEXT("indextoe_02_r"), {-0.16344258, 0.01840242, 0.17351183, -0.16344258, 0.01840242, 0.17331183}},
    {TEXT("littletoe_01_l"), {0.20091006, 0.01676717, 0.11919429, 0.20406585, 0.01438258, 0.13426147}},
    {TEXT("littletoe_01_r"), {-0.20091011, 0.01676720, 0.11919422, -0.20406588, 0.01438262, 0.13426136}},
    {TEXT("littletoe_02_l"), {0.20406585, 0.01438258, 0.13426147, 0.20406585, 0.01438258, 0.13406147}},
    {TEXT("littletoe_02_r"), {-0.20406588, 0.01438262, 0.13426136, -0.20406588, 0.01438262, 0.13406136}},
    {TEXT("lowerarm_l"), {0.39978073, 1.36416108, -0.05702398, 0.65825274, 1.38366490, -0.05090872}},
    {TEXT("lowerarm_r"), {-0.40076921, 1.35518131, -0.04269782, -0.65949732, 1.34703434, -0.05750233}},
    {TEXT("middle_01_l"), {0.74193208, 1.38049421, -0.04982162, 0.77625037, 1.36974520, -0.05894858}},
    {TEXT("middle_01_r"), {-0.74216978, 1.33498562, -0.06329653, -0.77555719, 1.32234343, -0.07339879}},
    {TEXT("middle_02_l"), {0.78721567, 1.38122093, -0.06093952, 0.79941805, 1.37191708, -0.06500173}},
    {TEXT("middle_02_r"), {-0.78712468, 1.33317805, -0.07556527, -0.79865820, 1.32322931, -0.08003373}},
    {TEXT("middle_03_l"), {0.81537066, 1.37828027, -0.06908294, 0.82064686, 1.37007049, -0.07108855}},
    {TEXT("middle_03_r"), {-0.81482544, 1.32870953, -0.08452977, -0.81956701, 1.32022972, -0.08673739}},
    {TEXT("middle_metacarpal_l"), {0.68709288, 1.37936303, -0.04812030, 0.74193208, 1.38049421, -0.04982162}},
    {TEXT("middle_metacarpal_r"), {-0.68783915, 1.33964202, -0.05713161, -0.74216978, 1.33498562, -0.06329653}},
    {TEXT("middletoe_01_l"), {0.17276936, 0.02374988, 0.14069034, 0.17977869, 0.01627181, 0.16307351}},
    {TEXT("middletoe_01_r"), {-0.17276939, 0.02374991, 0.14069027, -0.17977872, 0.01627183, 0.16307347}},
    {TEXT("middletoe_02_l"), {0.17977869, 0.01627181, 0.16307351, 0.17977869, 0.01627181, 0.16287351}},
    {TEXT("middletoe_02_r"), {-0.17977872, 0.01627183, 0.16307347, -0.17977872, 0.01627183, 0.16287347}},
    {TEXT("neck_01"), {0.00000001, 1.44561701, -0.01242076, 0.00000001, 1.49682788, -0.00122667}},
    {TEXT("pelvis"), {-0.00000000, 0.89800720, 0.02364323, 0.00000000, 0.93508440, 0.02350188}},
    {TEXT("pinky_01_l"), {0.72697944, 1.37410759, -0.08987680, 0.74188130, 1.36396802, -0.10506789}},
    {TEXT("pinky_01_r"), {-0.72346121, 1.33104125, -0.10206828, -0.73732260, 1.32046952, -0.11793406}},
    {TEXT("pinky_02_l"), {0.74813797, 1.37151804, -0.10989535, 0.75577521, 1.36312936, -0.11798976}},
    {TEXT("pinky_02_r"), {-0.74391995, 1.32776220, -0.12270367, -0.75081662, 1.31916780, -0.13123491}},
    {TEXT("pinky_03_l"), {0.76207445, 1.36928307, -0.12222818, 0.76759072, 1.36212120, -0.12451450}},
    {TEXT("pinky_03_r"), {-0.75752579, 1.32501806, -0.13559632, -0.76252590, 1.31760220, -0.13823245}},
    {TEXT("pinky_metacarpal_l"), {0.68540946, 1.38176838, -0.07067326, 0.72697944, 1.37410759, -0.08987680}},
    {TEXT("pinky_metacarpal_r"), {-0.68462459, 1.34267298, -0.07944190, -0.72346121, 1.33104125, -0.10206828}},
    {TEXT("ring_01_l"), {0.73461564, 1.38038450, -0.07100265, 0.76571804, 1.37090053, -0.07902648}},
    {TEXT("ring_01_r"), {-0.73321252, 1.33608598, -0.08380833, -0.76347048, 1.32490969, -0.09282349}},
    {TEXT("ring_02_l"), {0.77358675, 1.38201300, -0.08297714, 0.78862036, 1.37311109, -0.08931537}},
    {TEXT("ring_02_r"), {-0.77190787, 1.33560886, -0.09674029, -0.78621264, 1.32597328, -0.10366552}},
    {TEXT("ring_03_l"), {0.80226917, 1.37947299, -0.09537569, 0.80503741, 1.37237660, -0.09635786}},
    {TEXT("ring_03_r"), {-0.80005954, 1.33164720, -0.10995476, -0.80237085, 1.32442226, -0.11116033}},
    {TEXT("ring_metacarpal_l"), {0.68665259, 1.38078411, -0.05891939, 0.73461564, 1.38038450, -0.07100265}},
    {TEXT("ring_metacarpal_r"), {-0.68669126, 1.34132235, -0.06784107, -0.73321252, 1.33608598, -0.08380833}},
    {TEXT("ringtoe_01_l"), {0.18726473, 0.02165326, 0.13137279, 0.19425658, 0.01634882, 0.15082540}},
    {TEXT("ringtoe_01_r"), {-0.18726479, 0.02165330, 0.13137273, -0.19425666, 0.01634885, 0.15082535}},
    {TEXT("ringtoe_02_l"), {0.19425658, 0.01634882, 0.15082540, 0.19425658, 0.01634882, 0.15062540}},
    {TEXT("ringtoe_02_r"), {-0.19425666, 0.01634885, 0.15082535, -0.19425666, 0.01634885, 0.15062535}},
    {TEXT("spine_01"), {0.00000000, 0.93508440, 0.02350188, -0.00000000, 0.99922709, 0.03831386}},
    {TEXT("spine_02"), {-0.00000000, 0.99922709, 0.03831386, -0.00000000, 1.07143121, 0.04549811}},
    {TEXT("spine_03"), {-0.00000000, 1.07143121, 0.04549811, 0.00000000, 1.16213169, 0.03943561}},
    {TEXT("spine_04"), {9.210994e-11, 1.16213169, 0.03943561, 0.00000000, 1.32262081, 0.00427647}},
    {TEXT("spine_05"), {3.456390e-07, 1.32262081, 0.00427647, 0.00000000, 1.44561701, -0.01242076}},
    {TEXT("thigh_l"), {0.09556605, 0.88891661, 0.02548708, 0.11507245, 0.47199810, 0.01686990}},
    {TEXT("thigh_r"), {-0.09556607, 0.88891661, 0.02548706, -0.11507246, 0.47199818, 0.01686982}},
    {TEXT("thumb_01_l"), {0.67391825, 1.36391330, -0.03215702, 0.71399175, 1.37688449, -0.00644220}},
    {TEXT("thumb_01_r"), {-0.67440154, 1.32535309, -0.04032812, -0.71628787, 1.33410773, -0.01558586}},
    {TEXT("thumb_02_l"), {0.70360523, 1.36683666, 0.00283627, 0.72337238, 1.35998206, 0.00228624}},
    {TEXT("thumb_02_r"), {-0.70504837, 1.32521328, -0.00604774, -0.72407218, 1.31656481, -0.00720555}},
    {TEXT("thumb_03_l"), {0.72720382, 1.36185630, 0.01726639, 0.73878326, 1.35803864, 0.00949893}},
    {TEXT("thumb_03_r"), {-0.72861660, 1.31826146, 0.00759560, -0.73949988, 1.31330164, -0.00052535}},
    {TEXT("upperarm_l"), {0.17739084, 1.38555628, -0.02019965, 0.39978073, 1.36416108, -0.05702398}},
    {TEXT("upperarm_r"), {-0.17738359, 1.38458486, -0.02021711, -0.40076921, 1.35518131, -0.04269782}}
};

static Vec3 get_head(const FString& name) {
    const auto* it = BASELINE_MH_YUP_M_RAW.Find(name);
    if (!it) {
        return Vec3::Zero();
    }
    const auto& t = *it;
    return Vec3(t.Get<0>(), t.Get<1>(), t.Get<2>());
}

static Vec3 get_tail(const FString& name) {
    const auto* it = BASELINE_MH_YUP_M_RAW.Find(name);
    if (!it) {
        return Vec3::Zero();
    }
    const auto& t = *it;
    return Vec3(t.Get<3>(), t.Get<4>(), t.Get<5>());
}

// ---------------------------------------------------------------------------
// Parent hierarchy map
// ---------------------------------------------------------------------------

const TMap<FString, FString> PARENT_MAP = {
    {TEXT("ball_l"), TEXT("foot_l")}, {TEXT("ball_r"), TEXT("foot_r")},
    {TEXT("bigtoe_01_l"), TEXT("ball_l")}, {TEXT("bigtoe_01_r"), TEXT("ball_r")},
    {TEXT("bigtoe_02_l"), TEXT("bigtoe_01_l")}, {TEXT("bigtoe_02_r"), TEXT("bigtoe_01_r")},
    {TEXT("calf_l"), TEXT("thigh_l")}, {TEXT("calf_r"), TEXT("thigh_r")},
    {TEXT("clavicle_l"), TEXT("spine_05")}, {TEXT("clavicle_r"), TEXT("spine_05")},
    {TEXT("foot_l"), TEXT("calf_l")}, {TEXT("foot_r"), TEXT("calf_r")},
    {TEXT("hand_l"), TEXT("lowerarm_l")}, {TEXT("hand_r"), TEXT("lowerarm_r")},
    {TEXT("head"), TEXT("neck_02")},
    {TEXT("index_01_l"), TEXT("index_metacarpal_l")}, {TEXT("index_01_r"), TEXT("index_metacarpal_r")},
    {TEXT("index_02_l"), TEXT("index_01_l")}, {TEXT("index_02_r"), TEXT("index_01_r")},
    {TEXT("index_03_l"), TEXT("index_02_l")}, {TEXT("index_03_r"), TEXT("index_02_r")},
    {TEXT("index_metacarpal_l"), TEXT("hand_l")}, {TEXT("index_metacarpal_r"), TEXT("hand_r")},
    {TEXT("indextoe_01_l"), TEXT("ball_l")}, {TEXT("indextoe_01_r"), TEXT("ball_r")},
    {TEXT("indextoe_02_l"), TEXT("indextoe_01_l")}, {TEXT("indextoe_02_r"), TEXT("indextoe_01_r")},
    {TEXT("littletoe_01_l"), TEXT("ball_l")}, {TEXT("littletoe_01_r"), TEXT("ball_r")},
    {TEXT("littletoe_02_l"), TEXT("littletoe_01_l")}, {TEXT("littletoe_02_r"), TEXT("littletoe_01_r")},
    {TEXT("lowerarm_l"), TEXT("upperarm_l")}, {TEXT("lowerarm_r"), TEXT("upperarm_r")},
    {TEXT("middle_01_l"), TEXT("middle_metacarpal_l")}, {TEXT("middle_01_r"), TEXT("middle_metacarpal_r")},
    {TEXT("middle_02_l"), TEXT("middle_01_l")}, {TEXT("middle_02_r"), TEXT("middle_01_r")},
    {TEXT("middle_03_l"), TEXT("middle_02_l")}, {TEXT("middle_03_r"), TEXT("middle_02_r")},
    {TEXT("middle_metacarpal_l"), TEXT("hand_l")}, {TEXT("middle_metacarpal_r"), TEXT("hand_r")},
    {TEXT("middletoe_01_l"), TEXT("ball_l")}, {TEXT("middletoe_01_r"), TEXT("ball_r")},
    {TEXT("middletoe_02_l"), TEXT("middletoe_01_l")}, {TEXT("middletoe_02_r"), TEXT("middletoe_01_r")},
    {TEXT("neck_01"), TEXT("spine_05")}, {TEXT("pelvis"), TEXT("root")},
    {TEXT("pinky_01_l"), TEXT("pinky_metacarpal_l")}, {TEXT("pinky_01_r"), TEXT("pinky_metacarpal_r")},
    {TEXT("pinky_02_l"), TEXT("pinky_01_l")}, {TEXT("pinky_02_r"), TEXT("pinky_01_r")},
    {TEXT("pinky_03_l"), TEXT("pinky_02_l")}, {TEXT("pinky_03_r"), TEXT("pinky_02_r")},
    {TEXT("pinky_metacarpal_l"), TEXT("hand_l")}, {TEXT("pinky_metacarpal_r"), TEXT("hand_r")},
    {TEXT("ring_01_l"), TEXT("ring_metacarpal_l")}, {TEXT("ring_01_r"), TEXT("ring_metacarpal_r")},
    {TEXT("ring_02_l"), TEXT("ring_01_l")}, {TEXT("ring_02_r"), TEXT("ring_01_r")},
    {TEXT("ring_03_l"), TEXT("ring_02_l")}, {TEXT("ring_03_r"), TEXT("ring_02_r")},
    {TEXT("ring_metacarpal_l"), TEXT("hand_l")}, {TEXT("ring_metacarpal_r"), TEXT("hand_r")},
    {TEXT("ringtoe_01_l"), TEXT("ball_l")}, {TEXT("ringtoe_01_r"), TEXT("ball_r")},
    {TEXT("ringtoe_02_l"), TEXT("ringtoe_01_l")}, {TEXT("ringtoe_02_r"), TEXT("ringtoe_01_r")},
    {TEXT("root"), TEXT("")},
    {TEXT("spine_01"), TEXT("pelvis")}, {TEXT("spine_02"), TEXT("spine_01")},
    {TEXT("spine_03"), TEXT("spine_02")}, {TEXT("spine_04"), TEXT("spine_03")},
    {TEXT("spine_05"), TEXT("spine_04")}, {TEXT("thigh_l"), TEXT("pelvis")},
    {TEXT("thigh_r"), TEXT("pelvis")}, {TEXT("thumb_01_l"), TEXT("hand_l")},
    {TEXT("thumb_01_r"), TEXT("hand_r")}, {TEXT("thumb_02_l"), TEXT("thumb_01_l")},
    {TEXT("thumb_02_r"), TEXT("thumb_01_r")}, {TEXT("thumb_03_l"), TEXT("thumb_02_l")},
    {TEXT("thumb_03_r"), TEXT("thumb_02_r")}, {TEXT("upperarm_l"), TEXT("clavicle_l")},
    {TEXT("upperarm_r"), TEXT("clavicle_r")}
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

static double smplx_leg_chain_length(const MatrixXd& joints) {
    // Left: pelvis(0)->l_hip(1)->l_knee(4)->l_ankle(7)->l_foot(10)
    double l_chain =
        (joints.row(1) - joints.row(0)).norm() +
        (joints.row(4) - joints.row(1)).norm() +
        (joints.row(7) - joints.row(4)).norm() +
        (joints.row(10) - joints.row(7)).norm();

    // Right: pelvis(0)->r_hip(2)->r_knee(5)->r_ankle(8)->r_foot(11)
    double r_chain =
        (joints.row(2) - joints.row(0)).norm() +
        (joints.row(5) - joints.row(2)).norm() +
        (joints.row(8) - joints.row(5)).norm() +
        (joints.row(11) - joints.row(8)).norm();

    return 0.5 * (l_chain + r_chain);
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

TMap<FString, Vec3> reproportion(const MatrixXd& joints, double beta0) {
    TMap<FString, Vec3> mh_new;

    // Step 0: Root at world origin
    mh_new.Add(TEXT("root"), Vec3::Zero());

    // Step 1: Compute raw SMPL-X segment lengths
    TMap<FString, double> raw_lengths;
    for (const auto& seg : SEGMENTS) {
        const auto& label = seg.Get<0>();
        int32 p_idx = seg.Get<1>();
        int32 c_idx = seg.Get<2>();
        raw_lengths.Add(label, (joints.row(c_idx) - joints.row(p_idx)).norm());
    }

    // Step 2: L/R symmetrise
    for (const auto& pair : MIRROR_PAIRS) {
        const auto& l_label = pair.Key;
        const auto& r_label = pair.Value;
        double avg = 0.5 * (raw_lengths[l_label] + raw_lengths[r_label]);
        raw_lengths[l_label] = avg;
        raw_lengths[r_label] = avg;
    }

    // Step 3: Pelvis via chain-ratio then segment walk
    double slope_rtp = ROOT_TO_PELVIS_CHAIN_RATIO.Key;
    double intercept_rtp = ROOT_TO_PELVIS_CHAIN_RATIO.Value;
    double smplx_leg_chain = smplx_leg_chain_length(joints);
    double L_mh_new = (intercept_rtp + slope_rtp * beta0) * smplx_leg_chain;

    Vec3 p_ref = Vec3::Zero();
    Vec3 c_ref = get_head(TEXT("pelvis"));
    Vec3 diff = c_ref - p_ref;
    double diff_norm = diff.norm();
    Vec3 direction = (diff_norm < 1e-9) ? Vec3(0.0, 1.0, 0.0) : (diff / diff_norm);
    mh_new.Add(TEXT("pelvis"), mh_new[TEXT("root")] + L_mh_new * direction);

    // Segment walk for all segments
    for (const auto& seg : SEGMENTS) {
        const auto& label = seg.Get<0>();
        int32 p_idx = seg.Get<1>();
        int32 c_idx = seg.Get<2>();

        const FString& p_smplx_name = SMPLX_JOINT_NAMES[p_idx];
        const FString& c_smplx_name = SMPLX_JOINT_NAMES[c_idx];
        const FString& p_mh_name = MH_OF_SMPLX[p_smplx_name];
        const FString& c_mh_name = MH_OF_SMPLX[c_smplx_name];

        double L_smplx = raw_lengths[label];
        if (L_smplx < 1e-9) {
            mh_new.Add(c_mh_name, get_head(c_mh_name));
            continue;
        }

        const TPair<double, double>& ratio_pair = H3_RATIOS[label];
        double slope = ratio_pair.Key;
        double intercept = ratio_pair.Value;
        double ratio = intercept + slope * beta0;
        double L_mh_new_seg = ratio * L_smplx;

        Vec3 p_ref_seg = get_head(p_mh_name);
        Vec3 c_ref_seg = get_head(c_mh_name);
        Vec3 diff_seg = c_ref_seg - p_ref_seg;
        double diff_norm_seg = diff_seg.norm();

        if (diff_norm_seg < 1e-9) {
            mh_new.Add(c_mh_name, get_head(c_mh_name));
            continue;
        }

        Vec3 direction_seg = diff_seg / diff_norm_seg;
        Vec3 parent_pos = mh_new.Contains(p_mh_name) ? mh_new[p_mh_name] : get_head(p_mh_name);
        mh_new.Add(c_mh_name, parent_pos + L_mh_new_seg * direction_seg);
    }

    // Step 4: Finger chain scaling
    TMap<FString, double> smplx_finger_chains;
    for (const auto& fc : SMPLX_FINGER_CHAINS) {
        const auto& label = fc.Get<0>();
        int32 wrist_idx = fc.Get<1>();
        const auto& phalanx_idxs = fc.Get<2>();

        Vec3 prev = joints.row(wrist_idx);
        double total = 0.0;
        for (int32 idx : phalanx_idxs) {
            total += (joints.row(idx) - prev.transpose()).norm();
            prev = joints.row(idx);
        }
        smplx_finger_chains.Add(label, total / 100.0);
    }

    // Symmetrise finger chains
    for (const auto& pair : FINGER_MIRROR_PAIRS) {
        double avg = 0.5 * (smplx_finger_chains[pair.Key] + smplx_finger_chains[pair.Value]);
        smplx_finger_chains[pair.Key] = avg;
        smplx_finger_chains[pair.Value] = avg;
    }

    // Apply finger chain scaling
    for (const auto& fc : SMPLX_FINGER_CHAINS) {
        const auto& label = fc.Get<0>();
        const TPair<double, double>& finger_ratio = FINGER_H3[label];
        double slope_f = finger_ratio.Key;
        double intcpt_f = finger_ratio.Value;
        double smplx_chain = smplx_finger_chains[label];

        if (smplx_chain < 1e-9) continue;

        double target_chain = (intcpt_f + slope_f * beta0) * smplx_chain;
        const TPair<FString, TArray<FString>>& finger_chain = MH_FINGER_CHAINS[label];
        const auto& wrist_bone = finger_chain.Key;
        const auto& chain_bones = finger_chain.Value;

        Vec3 baseline_anchor = get_head(wrist_bone);
        Vec3 prev_bl = baseline_anchor;
        double baseline_chain = 0.0;

        for (const auto& bn : chain_bones) {
            Vec3 pos_bl = get_head(bn);
            baseline_chain += (pos_bl - prev_bl).norm();
            prev_bl = pos_bl;
        }

        const FString& last_bn = chain_bones.Last();
        baseline_chain += (get_tail(last_bn) - get_head(last_bn)).norm();

        if (baseline_chain < 1e-9) continue;

        double scale_factor = target_chain / baseline_chain;
        Vec3 anchor = mh_new.Contains(wrist_bone) ? mh_new[wrist_bone] : get_head(wrist_bone);

        for (const auto& bn : chain_bones) {
            Vec3 offset = get_head(bn) - baseline_anchor;
            mh_new.Add(bn, anchor + offset * scale_factor);
        }
    }

    // Step 5: Foot chain scaling
    TMap<FString, double> smplx_foot_chains;
    for (const auto& fr : SMPLX_FOOT_REFS) {
        const auto& label = fr.Get<0>();
        int32 ankle_idx = fr.Get<1>();
        int32 foot_idx = fr.Get<2>();
        double dist = (joints.row(foot_idx) - joints.row(ankle_idx)).norm();
        smplx_foot_chains.Add(label, dist / 100.0);
    }

    // Symmetrise foot chains
    for (const auto& pair : FOOT_MIRROR_PAIRS) {
        double avg = 0.5 * (smplx_foot_chains[pair.Key] + smplx_foot_chains[pair.Value]);
        smplx_foot_chains[pair.Key] = avg;
        smplx_foot_chains[pair.Value] = avg;
    }

    // Apply foot chain scaling
    for (const auto& fc_entry : MH_FOOT_CHAINS) {
        const auto& label = fc_entry.Key;
        const auto& mh_foot_bone = fc_entry.Value.Key;
        const auto& mh_ball_bone = fc_entry.Value.Value;

        const TPair<double, double>& foot_ratio = FOOT_H3[label];
        double slope_ft = foot_ratio.Key;
        double intcpt_ft = foot_ratio.Value;
        double smplx_chain = smplx_foot_chains[label];

        if (smplx_chain < 1e-9) continue;

        double target_chain = (intcpt_ft + slope_ft * beta0) * smplx_chain;

        Vec3 foot_head_bl = get_head(mh_foot_bone);
        Vec3 ball_head_bl = get_head(mh_ball_bone);
        Vec3 ball_tail_bl = get_tail(mh_ball_bone);
        double baseline_chain = (ball_head_bl - foot_head_bl).norm() +
                               (ball_tail_bl - ball_head_bl).norm();

        if (baseline_chain < 1e-9) continue;

        double scale_factor = target_chain / baseline_chain;
        Vec3 anchor = mh_new.Contains(mh_foot_bone) ? mh_new[mh_foot_bone] : get_head(mh_foot_bone);

        const TArray<FString>* foot_bones = FOOT_CHAIN_BONES.Find(label);
        if (foot_bones) {
            for (const auto& bn : *foot_bones) {
                if (BASELINE_MH_YUP_M_RAW.Contains(bn)) {
                    Vec3 offset = get_head(bn) - foot_head_bl;
                    mh_new.Add(bn, anchor + offset * scale_factor);
                }
            }
        }
    }

    // Step 6: Interpolate spine_04 and spine_05
    if (mh_new.Contains(TEXT("spine_03")) && mh_new.Contains(TEXT("neck_01"))) {
        Vec3 bl_spine03 = get_head(TEXT("spine_03"));
        Vec3 bl_neck01 = get_head(TEXT("neck_01"));
        Vec3 bl_seg = bl_neck01 - bl_spine03;
        double bl_seg_len = bl_seg.norm();

        if (bl_seg_len > 1e-9) {
            Vec3 sc_spine03 = mh_new[TEXT("spine_03")];
            Vec3 sc_neck01 = mh_new[TEXT("neck_01")];

            for (const FString& bone : {TEXT("spine_04"), TEXT("spine_05")}) {
                Vec3 bl_bone = get_head(bone);
                double t = (bl_bone - bl_spine03).norm() / bl_seg_len;
                mh_new.Add(bone, sc_spine03 + t * (sc_neck01 - sc_spine03));
            }
        }
    }

    return mh_new;
}

}

