# Copyright Supergene. 에디터 메뉴에서 호출하는 파이프라인 명령 함수.
# 콘텐츠 브라우저에서 선택한 애셋을 입력으로 동작한다(입력창 불필요).
# 인터랙티브 에디터에서는 RHI가 실제라 솔브가 그대로 동작(-AllowCommandletRendering 불필요).

import unreal

import config
import step1_ingest
import step2_solve
import step3_export


def _selected():
    return list(unreal.EditorUtilityLibrary.get_selected_assets())


def _notify(msg: str, error: bool = False):
    (unreal.log_error if error else unreal.log)(f"[MHA-TOOL] {msg}")


def _popup(title: str, msg: str):
    unreal.EditorDialog.show_message(unreal.Text(title), unreal.Text(msg), unreal.AppMsgType.OK)


def _ask_ingest_folder():
    """폴더 입력을 받는다. 취소하거나 미지정이면 None.

    ⚠️ `unreal.EditorDialog.open_directory_dialog` 는 **존재하지 않는다.**
    OpenDirectoryDialog 는 C++ IDesktopPlatform 의 가상 함수일 뿐 UFUNCTION 이 아니라
    Python/BP 에 노출되지 않는다. EditorDialog 가 제공하는 것은 show_message /
    show_suppressable_warning_dialog / show_object_details_view(s) 4개뿐.

    대신 MonoVideoIngestDeviceSettings(UObject) 의 디테일 뷰를 띄운다. TakeDirectory 가
    FDirectoryPath 라 폴더 찾아보기 버튼이 그대로 렌더링된다.
    (같은 창의 DisplayName / VideoDiscoveryExpression 필드는 이 경로에서 쓰이지 않는다 —
     CaptureManagerIngestBlueprintLibrary 는 FindTakeDirectories 로 직접 판별하므로
     디스커버리 표현식이 필요 없다. 폴더 피커를 얻기 위해 이 타입을 재활용할 뿐.)
    """
    settings = unreal.MonoVideoIngestDeviceSettings()

    if config.INGEST_DEFAULT_DIR:
        settings.set_editor_property(
            "take_directory", unreal.DirectoryPath(config.INGEST_DEFAULT_DIR))

    options = unreal.EditorDialogLibraryObjectDetailsViewOptions()
    options.set_editor_property("allow_resizing", True)
    options.set_editor_property("min_width", 640)

    accepted = unreal.EditorDialog.show_object_details_view(
        unreal.Text("MHA Phase 3 — 인제스트할 폴더 (Take Directory)"), settings, options)
    if not accepted:
        return None

    folder = settings.take_directory.path
    if not folder:
        _notify("폴더를 지정하지 않았습니다.", error=True)
        return None
    return folder


def _report_ingest(folder: str, created: list, failures: list):
    lines = [f"폴더: {folder}", ""]
    lines.append(f"CD {len(created)}개 생성:" if created else "생성된 CD 없음")
    lines.extend(f"  {cd.get_name()}" for cd in created)
    if failures:
        lines.append(f"\n⚠️ 실패 {len(failures)}건:")
        lines.extend(f"  {path}: {err}" for path, err in failures[:5])
    msg = "\n".join(lines)
    _notify(msg.replace("\n", " | "), error=bool(failures))
    _popup("MHA Phase 3", msg)


def _run_ingest(folder: str, per_file: bool):
    with unreal.ScopedSlowTask(1, f"MHA: Ingest {folder}") as task:
        task.make_dialog(True)
        if per_file:
            return step1_ingest.ingest_videos_in_folder(folder)
        return step1_ingest.ingest_folder(folder)


def ingest_pick_folder():
    """폴더를 골라 스캔 인제스트 → 생성된 CD 를 보고. (테이크 폴더 구조 기준)"""
    folder = _ask_ingest_folder()
    if not folder:
        return
    try:
        created, failures = _run_ingest(folder, per_file=False)
    except Exception as exc:
        _notify(f"인제스트 실패: {exc}", error=True)
        _popup("MHA Phase 3", f"FAILED\n\n{exc}")
        return
    _report_ingest(folder, created, failures)


def ingest_pick_folder_per_file():
    """폴더 바로 아래의 비디오 파일 각각을 개별 테이크로 인제스트."""
    folder = _ask_ingest_folder()
    if not folder:
        return
    try:
        created, failures = _run_ingest(folder, per_file=True)
    except Exception as exc:
        _notify(f"인제스트 실패: {exc}", error=True)
        _popup("MHA Phase 3", f"FAILED\n\n{exc}")
        return
    _report_ingest(folder, created, failures)


def ingest_solve_export_pick_folder(per_file: bool = True):
    """폴더 선택 → 인제스트 → 생성된 CD 전부 솔브 + 익스포트 (Phase 3→2 연결 확인용).

    인제스트가 CD 객체를 직접 돌려주므로 같은 세션에서 그대로 솔브로 넘긴다
    (바디 익스포트는 솔브와 같은 세션이어야 정적이 되지 않음 — §Phase 1 🔴 발견).
    """
    folder = _ask_ingest_folder()
    if not folder:
        return

    try:
        created, failures = _run_ingest(folder, per_file=per_file)
        if not created:
            _report_ingest(folder, created, failures)
            return

        results = []
        with unreal.ScopedSlowTask(len(created), "MHA: Solve + Export") as task:
            task.make_dialog(True)
            for capture_data in created:
                if task.should_cancel():
                    break
                name = capture_data.get_name()
                task.enter_progress_frame(1, f"Solving {name}")
                perf = step2_solve.create_and_process(
                    capture_data.get_path_name(), config.PERF_DIR,
                    enable_body=True, enable_face=False)
                seq = step3_export.export_anim_sequence(
                    perf, config.EXPORT_DIR, f"AS_{name}",
                    export_body=True, export_face=False, export_skeleton="performer")
                results.append(seq.get_name())
    except Exception as exc:
        _notify(f"실패: {exc}", error=True)
        _popup("MHA Phase 3", f"FAILED\n\n{exc}")
        return

    msg = f"완료 {len(results)}개 → {config.EXPORT_DIR}\n" + "\n".join(f"  {r}" for r in results)
    if failures:
        msg += f"\n\n⚠️ 인제스트 실패 {len(failures)}건 — Output Log 확인"
    _notify(msg.replace("\n", " | "))
    _popup("MHA Phase 3", msg)


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
