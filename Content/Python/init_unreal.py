# Copyright Supergene.
# UE는 Python 경로의 init_unreal.py 를 에디터 시작 시 자동 실행한다.
# 여기서 파이프라인 경로를 sys.path에 넣고 MHA 메뉴를 등록한다.
# (헤드리스 -run=pythonscript 에서는 메인 메뉴가 없어 register()가 조용히 skip)

import os
import sys

import unreal

_pipeline_dir = os.path.join(
    unreal.SystemLibrary.get_project_content_directory(), "Python", "pipeline")
if _pipeline_dir not in sys.path:
    sys.path.append(_pipeline_dir)

try:
    import mha_menu
    mha_menu.register()
except Exception as exc:  # 시작 실패가 에디터를 막지 않도록 방어
    unreal.log_error(f"[MHA] init_unreal menu registration failed: {exc}")
