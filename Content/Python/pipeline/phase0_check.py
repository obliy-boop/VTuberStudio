# Copyright Supergene. Phase 0 자가진단 — MetaHuman/CaptureManager API 로드 점검.
# 헤드리스 검증(§Phase 0)과 동일한 클래스 존재 확인을 에디터 버튼에서 실행하고
# 결과를 Output Log + 팝업으로 보고한다. 헤드리스 결과와 동일해야 한다.

import unreal

import config

# 파이프라인 필수 UCLASS (플러그인 활성화로 unreal 네임스페이스에 노출되어야 함)
_REQUIRED_CLASSES = (
    "MetaHumanPerformance",            # 솔브/처리 (MetaHuman Animator)
    "MonoVideoIngestDevice",           # 모노 비디오 인제스트 (CaptureManagerDevices)
    "MetaHumanPerformanceExportUtils",  # AnimSequence 익스포트
    "IngestCapability_Options",         # 인제스트 옵션
)


def run():
    """필수 클래스/스켈레톤 로드를 점검하고 팝업+로그로 보고. checks dict 반환."""
    checks = {name: hasattr(unreal, name) for name in _REQUIRED_CLASSES}

    lines = [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks.items()]
    passed = sum(checks.values())

    # 스켈레톤은 프로젝트 애셋 존재 여부라 카운트에서 분리(INFO). 바디는 있어야 익스포트 가능.
    for label, path in (("BODY", config.BODY_SKELETON), ("FACE", config.FACE_SKELETON)):
        loaded = unreal.load_asset(path) if unreal.EditorAssetLibrary.does_asset_exist(path) else None
        lines.append(f"INFO  {label} skeleton: {'OK' if loaded else 'None'}")

    msg = "\n".join(lines) + f"\n\nRESULT: {passed}/{len(checks)}"
    unreal.log(f"[PHASE0]\n{msg}")
    unreal.EditorDialog.show_message("MHA Phase 0", msg, unreal.AppMsgType.OK)  # 에디터 팝업
    return checks