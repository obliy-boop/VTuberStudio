# Copyright Supergene. 에디터 메뉴에서 호출하는 파이프라인 명령 함수.
# 콘텐츠 브라우저에서 선택한 애셋을 입력으로 동작한다(입력창 불필요).
# 인터랙티브 에디터에서는 RHI가 실제라 솔브가 그대로 동작(-AllowCommandletRendering 불필요).

import unreal

import config
import step2_solve
import step3_export


def _selected():
    return list(unreal.EditorUtilityLibrary.get_selected_assets())


def _notify(msg: str, error: bool = False):
    (unreal.log_error if error else unreal.log)(f"[MHA-TOOL] {msg}")


def solve_export_selected():
    """선택한 FootageCaptureData(들) → 솔브(바디) → AnimSequence 익스포트."""
    cds = [a for a in _selected() if isinstance(a, unreal.FootageCaptureData)]
    if not cds:
        _notify("FootageCaptureData(CD) 애셋을 콘텐츠 브라우저에서 선택하세요.", error=True)
        return
    with unreal.ScopedSlowTask(len(cds), "MHA: Solve + Export") as task:
        task.make_dialog(True)
        for cd in cds:
            if task.should_cancel():
                break
            task.enter_progress_frame(1, f"Solving {cd.get_name()}")
            perf = step2_solve.create_and_process(
                cd.get_path_name(), config.PERF_DIR, enable_body=True, enable_face=False)
            step3_export.export_anim_sequence(
                perf, config.EXPORT_DIR, f"AS_{cd.get_name()}",
                export_body=True, export_face=False, export_skeleton="performer")
    _notify(f"완료: {len(cds)}개 처리 → {config.EXPORT_DIR}")


def solve_selected():
    """선택한 CD → 솔브만 (익스포트 없음)."""
    cds = [a for a in _selected() if isinstance(a, unreal.FootageCaptureData)]
    if not cds:
        _notify("CD 애셋을 선택하세요.", error=True)
        return
    for cd in cds:
        step2_solve.create_and_process(cd.get_path_name(), config.PERF_DIR,
                                       enable_body=True, enable_face=False)
    _notify(f"솔브 완료: {len(cds)}개 → {config.PERF_DIR}")


def export_selected():
    """선택한 (이미 같은 세션에서 처리된) MetaHumanPerformance → AnimSequence.
    ⚠️ 콜드 로드(저장만 된 Performance)는 바디가 정적이 됨 — 같은 세션에서 솔브 직후 사용 권장."""
    perfs = [a for a in _selected() if isinstance(a, unreal.MetaHumanPerformance)]
    if not perfs:
        _notify("MetaHumanPerformance 애셋을 선택하세요.", error=True)
        return
    for perf in perfs:
        step3_export.export_anim_sequence(
            perf, config.EXPORT_DIR, f"AS_{perf.get_name()}",
            export_body=True, export_face=False, export_skeleton="performer")
    _notify(f"익스포트 완료: {len(perfs)}개 → {config.EXPORT_DIR}")
