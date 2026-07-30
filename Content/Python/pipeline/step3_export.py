# Copyright Supergene. Phase 3 - 익스포트.
# 처리된 MetaHumanPerformance -> AnimSequence(.uasset).
#
# ⚠️ 바디 익스포트 필수 설정 (MetaHumanPerformanceExportAnimationSettings, 5.8 헤더 확인):
#   - export_body = True           (기본 False! 안 켜면 바디 본 트랙 없이 정적 출력)
#   - source_skeletal_mesh         (Transient, "스크립트 호출자는 수동 설정 필수". 콜드 로드로는 못 얻음)
#   - export_skeleton: PERFORMER_SKELETON(솔브 스켈레톤 복사, 타깃 불필요) / EXISTING_SKELETON(타깃 리타게팅+BodyRetargeter)
#   - export_face: 바디 온리면 False
# 콜드 로드(저장된 Performance만 로드)로는 source_skeletal_mesh/바디 데이터가 세션에 없어 정적이 됨.
# => 바디 익스포트는 솔브와 같은 세션(batch_mha)에서 실행.
#
# 근거: .../MetaHumanAnimator/Content/Python/export_performance.py + MetaHumanPerformanceExportUtils.h

import unreal


def export_anim_sequence(perf: unreal.MetaHumanPerformance,
                         out_dir: str, out_name: str, *,
                         export_body: bool = True,
                         export_face: bool = False,
                         export_skeleton: str = "performer",   # "performer" | "existing"
                         target_skeleton_path: str | None = None,     # existing 모드용
                         body_retargeter_path: str | None = None,     # existing 모드용 IKRetargeter
                         source_skeletal_mesh: unreal.SkeletalMesh | None = None,  # 바디 익스포트 필수
                         source_skeletal_mesh_path: str | None = None,
                         unsolved_behavior: str = "last_valid",  # "last_valid" | "neutral"
                         enable_head_movement: bool = False,
                         whole_sequence: bool = False,
                         save: bool = True) -> unreal.AnimSequence:
    """처리된 Performance에서 AnimSequence를 생성한다. 기본은 바디 온리."""
    s = unreal.MetaHumanPerformanceExportAnimationSettings()
    s.set_editor_property("show_export_dialog", False)     # 헤드리스 필수
    s.set_editor_property("auto_save_anim_sequence", save)
    s.set_editor_property("package_path", out_dir)
    s.set_editor_property("asset_name", out_name)
    s.set_editor_property("enable_head_movement", enable_head_movement)
    s.set_editor_property("export_range",
                          unreal.PerformanceExportRange.WHOLE_SEQUENCE if whole_sequence
                          else unreal.PerformanceExportRange.PROCESSING_RANGE)
    s.set_editor_property("export_face", export_face)
    s.set_editor_property("export_body", export_body)

    if export_body:
        # source skeletal mesh (스크립트 호출자 수동 설정 필수)
        src = source_skeletal_mesh
        if src is None and source_skeletal_mesh_path:
            src = unreal.load_asset(source_skeletal_mesh_path)
        if src is not None:
            s.set_editor_property("source_skeletal_mesh", src)

        if export_skeleton == "existing":
            s.set_editor_property("export_skeleton", unreal.PerformanceExportSkeleton.EXISTING_SKELETON)
            if target_skeleton_path:
                s.set_editor_property("target_skeleton_or_skeletal_mesh",
                                      unreal.load_asset(target_skeleton_path))
            if body_retargeter_path:
                s.set_editor_property("body_retargeter", unreal.load_asset(body_retargeter_path))
        else:
            s.set_editor_property("export_skeleton", unreal.PerformanceExportSkeleton.PERFORMER_SKELETON)

        s.set_editor_property("body_unsolved_behavior",
                              unreal.BodyUnsolvedFrameBehavior.NEUTRAL_POSE if unsolved_behavior == "neutral"
                              else unreal.BodyUnsolvedFrameBehavior.LAST_VALID_FRAME)
    elif export_face and target_skeleton_path:
        # face 온리: 타깃 스켈레톤 지정
        s.set_editor_property("target_skeleton_or_skeletal_mesh",
                              unreal.load_asset(target_skeleton_path))

    seq: unreal.AnimSequence = unreal.MetaHumanPerformanceExportUtils.export_animation_sequence(perf, s)
    if not seq:
        raise RuntimeError("AnimSequence export failed (returned None)")
    unreal.log(f"[MHA] exported AnimSequence: {seq.get_name()}")

    if save:
        unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_asset(seq.get_outer().get_name(), False)
    return seq
