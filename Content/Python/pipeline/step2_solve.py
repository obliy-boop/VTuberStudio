# Copyright Supergene. Phase 2 - 솔브.
# FootageCaptureData(CD) -> MetaHumanPerformance 생성 + 처리(블로킹).
# 현재 바디 모캡 중심(body_tracking=True). face는 선택.
#
# 근거: 5.8 동봉 스크립트
#   .../MetaHumanAnimator/Content/Python/process_monocular_performance.py (생성)
#   .../MetaHumanAnimator/Content/Python/process_performance.py (process_shot: 블로킹 처리)
# 프로퍼티명은 MetaHumanPerformance.h 에서 검증(§ 계획서 프로퍼티 표).

import unreal


def create_performance(capture_data_path: str, storage_path: str, *,
                       asset_name: str | None = None) -> unreal.MetaHumanPerformance:
    """CD 애셋으로부터 모노 footage용 MetaHumanPerformance 애셋을 생성한다."""
    cd = unreal.load_asset(capture_data_path)
    if cd is None:
        raise RuntimeError(f"capture data not found: {capture_data_path}")

    name = asset_name or f"{cd.get_name()}_Performance"
    perf: unreal.MetaHumanPerformance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        asset_name=name, package_path=storage_path,
        asset_class=unreal.MetaHumanPerformance,
        factory=unreal.MetaHumanPerformanceFactoryNew())

    # set_editor_property 사용 -> PostEditChangeProperty 트리거되어 셋업이 완료됨
    perf.set_editor_property("input_type", unreal.DataInputType.MONO_FOOTAGE)
    perf.set_editor_property("footage_capture_data", cd)
    return perf


def configure_processing(perf: unreal.MetaHumanPerformance, *,
                         enable_body: bool = True,
                         enable_face: bool = False,
                         auto_body_height: bool = True,
                         body_height_cm: float | None = None,
                         body_detection_confidence: float = 0.7,
                         body_tracking_confidence: float = 0.5,
                         enable_foot_locking: bool = True,
                         solve_tongue: bool = False) -> None:
    """처리 파라미터 설정. 기본은 바디 온리(face off)."""
    # 얼굴 솔브 (모노 footage 유효)
    perf.set_editor_property("face_tracking", enable_face)
    if enable_face:
        perf.set_editor_property("skip_tongue_solve", not solve_tongue)

    # 바디 트래킹 (5.8 신규). set_body_tracking() 메서드는 Python 미노출 →
    # set_editor_property 로 설정하면 BlueprintSetter(SetBodyTracking)가 자동 호출됨.
    perf.set_editor_property("body_tracking", enable_body)
    if enable_body:
        perf.set_editor_property("auto_body_height", auto_body_height)
        if not auto_body_height and body_height_cm is not None:
            perf.set_editor_property("body_height", float(body_height_cm))  # 권장 [145,190]
        perf.set_editor_property("body_detection_confidence", body_detection_confidence)
        perf.set_editor_property("body_tracking_confidence", body_tracking_confidence)
        perf.set_editor_property("enable_foot_locking", enable_foot_locking)


def process(perf: unreal.MetaHumanPerformance, *,
            start_frame: int | None = None,
            end_frame: int | None = None) -> None:
    """블로킹으로 파이프라인 실행(완료까지 반환 안 함). 헤드리스 배치에 안정적."""
    if start_frame is not None:
        perf.set_editor_property("start_frame_to_process", start_frame)
    if end_frame is not None:
        perf.set_editor_property("end_frame_to_process", end_frame)  # 상한 프레임은 미처리(N-1)

    perf.set_blocking_processing(True)
    unreal.log(f"[MHA] start_pipeline: {perf.get_name()}")
    err = perf.start_pipeline()
    if err is not unreal.StartPipelineErrorType.NONE:
        raise RuntimeError(f"start_pipeline failed: {err}")
    unreal.log(f"[MHA] pipeline finished: {perf.get_name()}")


def create_and_process(capture_data_path: str, storage_path: str, *,
                       enable_body: bool = True, enable_face: bool = False,
                       start_frame: int | None = None,
                       end_frame: int | None = None) -> unreal.MetaHumanPerformance:
    """생성 -> 설정 -> 처리 원샷 헬퍼."""
    perf = create_performance(capture_data_path, storage_path)
    configure_processing(perf, enable_body=enable_body, enable_face=enable_face)
    process(perf, start_frame=start_frame, end_frame=end_frame)
    return perf
