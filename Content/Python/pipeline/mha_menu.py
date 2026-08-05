# Copyright Supergene. 메인 메뉴에 "MHA" 드롭다운을 등록한다.
# 각 항목은 mha_tool 의 함수를 파이썬 문자열 커맨드로 호출.

import unreal

_MENU_OWNER = "MHAPipelineMenu"


def _add_entry(menu, name, label, tooltip, py_command):
    entry = unreal.ToolMenuEntry(name=name, type=unreal.MultiBlockType.MENU_ENTRY)
    entry.set_label(label)
    entry.set_tool_tip(tooltip)
    entry.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "", string=py_command)
    menu.add_menu_entry("MHASection", entry)


def register():
    menus = unreal.ToolMenus.get()
    main = menus.find_menu("LevelEditor.MainMenu")
    if not main:
        unreal.log_warning("[MHA] LevelEditor.MainMenu not found (headless?) - menu skipped")
        return

    # 상단에 'MHA' 드롭다운 추가
    mha = main.add_sub_menu(main.menu_name, "", "MHA", "MHA")
    mha.add_section("MHASection", unreal.Text("MotionCapture Pipeline"))

    _add_entry(mha, "Phase0Check", "Phase 0: API 자가진단",
               "필수 클래스/스켈레톤 로드 점검 (헤드리스 결과와 동일해야 함)",
               "import phase0_check; phase0_check.run()")
    _add_entry(mha, "IngestPerFile", "Phase 3: 인제스트 (폴더 내 파일별)",
               "폴더 바로 아래의 mp4/mov 각각을 개별 테이크로 인제스트",
               "import mha_tool; mha_tool.ingest_pick_folder_per_file()")
    _add_entry(mha, "IngestFolder", "Phase 3: 인제스트 (테이크 폴더 스캔)",
               "FindTakeDirectories 로 폴더 구조를 스캔해 타입별로 인제스트",
               "import mha_tool; mha_tool.ingest_pick_folder()")
    _add_entry(mha, "IngestSolveExport", "Phase 3: 인제스트 + 솔브 + 익스포트",
               "폴더 선택 → 파일별 인제스트 → 생성된 CD 전부 솔브 후 AnimSequence 익스포트",
               "import mha_tool; mha_tool.ingest_solve_export_pick_folder()")
    _add_entry(mha, "SolveExport", "Solve + Export (선택 CD)",
               "선택한 FootageCaptureData를 바디 솔브 후 AnimSequence로 익스포트",
               "import mha_tool; mha_tool.solve_export_selected()")
    _add_entry(mha, "SolveOnly", "Solve only (선택 CD)",
               "선택한 CD를 바디 솔브만 (Performance 생성)",
               "import mha_tool; mha_tool.solve_selected()")
    _add_entry(mha, "ExportPerf", "Export (선택 Performance)",
               "선택한 (같은 세션 처리된) Performance를 AnimSequence로 익스포트",
               "import mha_tool; mha_tool.export_selected()")

    menus.refresh_all_widgets()
    unreal.log("[MHA] menu registered")
