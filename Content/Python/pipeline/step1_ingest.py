# Copyright Supergene. Phase 3 — mp4 폴더 → CaptureManager 인제스트 → FootageCaptureData(CD) 애셋.
#
# ⚠️ 2026-07-31 전면 재작성 (Q1 확정)
#
# 최초 구현은 엔진의 `capture.*` 파이썬 패키지 + UnrealEndpointManager 를 썼다.
# 실행 결과 `TimeoutError: Timed out waiting for endpoint: <host>` 로 실패.
# 원인: 그 예제들(CaptureManagerApp/Content/Python/examples/)은 **LiveLink Hub 앱**에서
# 실행되도록 만들어진 것이고, UnrealEndpointManager 는 Hub 가 *별도의* 에디터
# 인스턴스를 네트워크로 발견하는 장치다. 에디터 안에서 호출하면 자기 자신을 찾지
# 못해 반드시 타임아웃한다.
#
# 에디터 내에서 쓸 API 는 따로 있다:
#   Engine/Plugins/VirtualProduction/CaptureManager/CaptureManagerEditor/
#     Source/CaptureManagerIngestBlueprint/Private/CaptureManagerIngestBlueprintLibrary.h
#     Content/Python/example_ingest_sync.py    ← 이 파일의 근거
#     Content/Python/example_ingest_scan.py    ← 폴더 스캔/라우팅 근거
#
# 이 API 는 엔드포인트/Hub 가 필요 없고, **FootageCaptureData 를 직접 반환**하므로
# 스냅샷 차집합(Q4)도 불필요해졌다.

import os
import sys

import unreal

import config

_CAPTURE_DATA_CLASS = "FootageCaptureData"


# ---------------------------------------------------------------------------
# 변환 설정
# ---------------------------------------------------------------------------

def make_conversion_params() -> "unreal.CaptureManagerConversionParams":
    """변환 파라미터. Phase 1 기준선이 jpeg 였으므로 JPG 로 맞춘다(구조체 기본값은 PNG)."""
    params = unreal.CaptureManagerConversionParams()
    params.set_editor_property("image_format", unreal.CaptureManagerImageFormat.JPG)
    params.set_editor_property("image_file_prefix", "frame")
    params.set_editor_property("audio_file_prefix", "audio")
    return params


# ---------------------------------------------------------------------------
# 단일 테이크 인제스트
# ---------------------------------------------------------------------------

def _slate_from_path(path: str) -> str:
    """파일/폴더 경로에서 슬레이트 이름을 만든다."""
    base = os.path.basename(os.path.normpath(path))
    stem = os.path.splitext(base)[0]
    # 애셋 이름에 쓰이므로 공백/특수문자를 정리
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in stem) or "Take"


def ingest_video_file(video_path: str,
                      *,
                      audio_path: str = "",
                      slate: str = None,
                      take_number: int = 1,
                      params=None):
    """단일 mp4/mov 파일을 모노 비디오로 인제스트한다. (동기, 블로킹)

    :return: unreal.FootageCaptureData
    :raises RuntimeError: 인제스트 실패 시 (에디터가 준 오류 메시지 포함)
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"비디오 파일이 없습니다: {video_path}")

    library = unreal.CaptureManagerIngestBlueprintLibrary
    slate_name = slate or _slate_from_path(video_path)

    unreal.log(f"[MHA-INGEST] mono <- {video_path} (slate={slate_name} take={take_number})")

    capture_data, error = library.ingest_mono_video_sync(
        video_path, audio_path, slate_name, take_number,
        params if params is not None else make_conversion_params())

    if not capture_data:
        raise RuntimeError(f"인제스트 실패 [{video_path}]: {error}")

    unreal.log(f"[MHA-INGEST]   OK -> {capture_data.get_path_name()}")
    return capture_data


# ---------------------------------------------------------------------------
# 폴더 스캔 + 라우팅 — Phase 3/4 의 주 진입점
# ---------------------------------------------------------------------------

def ingest_folder(search_root: str, *, recursive: bool = True, params=None) -> tuple:
    """폴더를 스캔해 발견한 테이크를 전부 인제스트한다.

    `FindTakeDirectories` 로 인벤토리를 만들고 내용에 따라 라우팅한다
    (take archive / LiveLink Face / stereo / mono / calibration).
    바디 모캡 용도에서는 대개 mono 경로만 타게 된다.

    :return: (capture_data_list, failures)
             failures = [(경로, 오류문자열), ...]
    """
    if not os.path.isdir(search_root):
        raise NotADirectoryError(f"폴더가 없습니다: {search_root}")

    library = unreal.CaptureManagerIngestBlueprintLibrary
    conv = params if params is not None else make_conversion_params()

    take_dirs = library.find_take_directories(search_root, recursive)
    unreal.log(f"[MHA-INGEST] {search_root} 에서 테이크 폴더 {len(take_dirs)}개 발견")

    if not take_dirs:
        return [], []

    created, failures = [], []

    for index, info in enumerate(take_dirs, start=1):
        path = info.path
        slate = _slate_from_path(path)
        unreal.log(f"[MHA-INGEST] --- [{index}/{len(take_dirs)}] {path}")

        capture_data, error = None, ""
        try:
            if info.is_take_archive:
                unreal.log("[MHA-INGEST]   type=TakeArchive")
                capture_data, error = library.ingest_take_archive_sync(path, conv)

            elif info.is_live_link_face:
                unreal.log("[MHA-INGEST]   type=LiveLinkFace")
                capture_data, error = library.ingest_live_link_face_sync(path, conv)

            elif len(info.video_files) == 2:
                audio = info.audio_files[0] if info.audio_files else ""
                calib = info.calibration_files[0] if info.calibration_files else ""
                unreal.log("[MHA-INGEST]   type=StereoVideo")
                capture_data, error = library.ingest_stereo_video_sync(
                    info.video_files[0], info.video_files[1], audio, calib, slate, index, conv)

            elif len(info.image_seq_dirs) == 2:
                audio = info.audio_files[0] if info.audio_files else ""
                calib = info.calibration_files[0] if info.calibration_files else ""
                unreal.log("[MHA-INGEST]   type=StereoImageSequence")
                capture_data, error = library.ingest_stereo_video_sync(
                    info.image_seq_dirs[0], info.image_seq_dirs[1], audio, calib, slate, index, conv)

            elif len(info.video_files) == 1:
                audio = info.audio_files[0] if info.audio_files else ""
                unreal.log(f"[MHA-INGEST]   type=MonoVideo {info.video_files[0]}")
                capture_data, error = library.ingest_mono_video_sync(
                    info.video_files[0], audio, slate, index, conv)

            elif info.calibration_files:
                unreal.log("[MHA-INGEST]   type=Calibration")
                capture_data, error = library.ingest_calibration_sync(
                    info.calibration_files[0], slate)

            else:
                unreal.log_warning(f"[MHA-INGEST]   판별 불가 — 건너뜀: {path}")
                failures.append((path, "인제스트 타입을 판별할 수 없음"))
                continue

        except Exception as exc:
            unreal.log_error(f"[MHA-INGEST]   예외: {exc}")
            failures.append((path, str(exc)))
            continue

        if capture_data:
            unreal.log(f"[MHA-INGEST]   OK -> {capture_data.get_path_name()}")
            created.append(capture_data)
        else:
            unreal.log_error(f"[MHA-INGEST]   FAILED -> {error}")
            failures.append((path, str(error)))

    unreal.log(f"[MHA-INGEST] 완료: 성공 {len(created)}, 실패 {len(failures)}")
    return created, failures


# 다중 mp4 폴더를 하나의 테이크 폴더로 보는 경우가 아니라
# "폴더 안의 mp4 각각이 개별 테이크"인 경우를 위한 변형.
def ingest_videos_in_folder(folder: str, *, params=None, extensions=(".mp4", ".mov")) -> tuple:
    """폴더 바로 아래의 비디오 파일 각각을 개별 mono 테이크로 인제스트.

    `find_take_directories` 는 한 폴더에 mp4 가 3개 있으면 그 폴더를 테이크 1개로
    묶어버려(video_files 길이가 2도 1도 아니면 판별 불가) 원하는 결과가 안 나온다.
    파일 단위로 처리하고 싶을 때 이 함수를 쓴다.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"폴더가 없습니다: {folder}")

    videos = sorted(
        os.path.join(folder, name) for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in extensions
        and os.path.isfile(os.path.join(folder, name))
    )

    unreal.log(f"[MHA-INGEST] {folder} 에서 비디오 {len(videos)}개 발견")

    conv = params if params is not None else make_conversion_params()
    created, failures = [], []

    for take_number, video in enumerate(videos, start=1):
        try:
            created.append(ingest_video_file(
                video, take_number=take_number, params=conv))
        except Exception as exc:
            unreal.log_error(f"[MHA-INGEST] {video}: {exc}")
            failures.append((video, str(exc)))

    unreal.log(f"[MHA-INGEST] 완료: 성공 {len(created)}, 실패 {len(failures)}")
    return created, failures


# ---------------------------------------------------------------------------
# 보조 유틸 — 리포팅/검증용 (인제스트 결과 확보에는 더 이상 필요 없음)
# ---------------------------------------------------------------------------

def list_capture_data(search_root: str = None, *, rescan: bool = False) -> set:
    """search_root 아래 FootageCaptureData 애셋의 패키지 경로 집합.

    이름(CD_ 접두사)이 아니라 **클래스**로 판별한다.
    """
    root = search_root or config.IMPORT_ROOT
    registry = unreal.AssetRegistryHelpers.get_asset_registry()

    if not unreal.EditorAssetLibrary.does_directory_exist(root):
        return set()

    if rescan:
        registry.scan_paths_synchronous([root], force_rescan=True)

    return {
        str(data.package_name)
        for data in registry.get_assets_by_path(root, recursive=True)
        if str(data.asset_class_path.asset_name) == _CAPTURE_DATA_CLASS
    }


# ---------------------------------------------------------------------------
# 헤드리스 진입점
#   UnrealEditor-Cmd.exe <uproject> -run=pythonscript
#     -script="<...>/step1_ingest.py --folder=D:/DanceVideo/converted"
#     -AllowCommandletRendering -unattended -nosplash -nopause -stdout
# ---------------------------------------------------------------------------

def run():
    import argparse

    parser = argparse.ArgumentParser(description="MHA Phase 3: capture ingest")
    parser.add_argument("--folder", required=True, help="스캔할 폴더")
    parser.add_argument("--per-file", action="store_true",
                        help="폴더 내 비디오 파일 각각을 개별 테이크로 처리")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    if args.per_file:
        created, failures = ingest_videos_in_folder(args.folder)
    else:
        created, failures = ingest_folder(args.folder, recursive=not args.no_recursive)

    for capture_data in created:
        unreal.log(f"[MHA-INGEST] CD: {capture_data.get_path_name()}")

    if failures or not created:
        unreal.log_error(f"[MHA-INGEST] FAILED (성공 {len(created)}, 실패 {len(failures)})")
        sys.exit(1)

    unreal.log(f"[MHA-INGEST] DONE: {len(created)} capture data asset(s)")


if __name__ == "__main__":
    run()
