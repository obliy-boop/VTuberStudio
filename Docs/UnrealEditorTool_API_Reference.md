# 언리얼 에디터 툴(Editor Tool) 제작 API 레퍼런스 (UE 5.8)

> 대상: `C:\SupergeneGithub\VTuberStudio` (`MetaHuman.uproject`, Engine **5.8**)
> 목적: MetaHuman(MHA) 파이프라인 자동화를 **에디터 안 버튼/메뉴/패널**로 노출하기 위한 API 조사.
> 관련 문서: [`MotionCapture_Automation_Plan.md`](./MotionCapture_Automation_Plan.md)
> 현재 구현: `Content/Python/init_unreal.py` → `mha_menu.register()` (아래 **§3 ToolMenus** 방식 사용 중).

---

## 0. 요약 — 에디터 툴을 만드는 5가지 경로

| # | 경로 | 언어 | UI 형태 | 자동 로드 | 적합한 용도 | MHA 적용 |
|---|---|---|---|---|---|---|
| A | **ToolMenus** (`unreal.ToolMenus`) | Python/C++ | 상단 메뉴·툴바 버튼 | `init_unreal.py` | 한 번 클릭으로 스크립트 실행 | ✅ **현재 사용** (`mha_menu.py`) |
| B | **Editor Utility Widget (EUW)** | UMG(BP)+Python | 독립 패널/도킹 탭 | 수동 or 코드 | 큐/로그 대시보드, 입력 폼 | 🔜 Phase 6 대시보드 |
| C | **Scripted Actions** (`AssetActionUtility`/`ActorActionUtility`) | BP+Python | 우클릭 컨텍스트 메뉴 | 자동(클래스 스캔) | 선택 애셋/액터 대상 동작 | 대안 (선택 CD 처리) |
| D | **Interactive Tools Framework / `UEdMode`** | C++ | 뷰포트 모달 툴 + 툴 팔레트 | 플러그인 모듈 | 뷰포트 상호작용 툴(모델링류) | ❌ 과함 |
| E | **Scriptable Tools System** | BP | 뷰포트 모달 툴(팔레트) | 플러그인 | D를 BP로 (C++ 없이) | ❌ 과함 |

**MHA 결론:** 지금처럼 **A(ToolMenus) 버튼**으로 파이프라인 함수를 호출하는 것이 가장 견고하다. 운영 대시보드가 필요해지면 **B(EUW)** 를 추가한다. C~E는 뷰포트 상호작용이 필요 없는 배치 파이프라인에는 과하다.

---

## 1. 사전 준비 (Python 스크립팅 활성화)

에디터 툴을 Python으로 만들려면 아래가 필요하다(본 프로젝트는 이미 설정 완료).

**플러그인** (`MetaHuman.uproject`에 활성화됨):
- `PythonScriptPlugin` — `unreal` 모듈, `-run=pythonscript` 커맨드릿.
- `EditorScriptingUtilities` — `EditorAssetLibrary`, `EditorUtilityLibrary` 등 헬퍼.

**스크립트 경로 등록** (`Config/DefaultEngine.ini`):
```ini
[/Script/PythonScriptPlugin.PythonScriptPluginSettings]
+AdditionalPaths=(Path="/Game/Python/pipeline")
```

**자동 실행 진입점** — 엔진은 Python 경로에서 `init_unreal.py`를 **에디터 기동 시 자동 실행**한다. 여기서 메뉴를 등록한다.
```python
# Content/Python/init_unreal.py (현재 구현)
import os, sys, unreal
_pipeline_dir = os.path.join(
    unreal.SystemLibrary.get_project_content_directory(), "Python", "pipeline")
if _pipeline_dir not in sys.path:
    sys.path.append(_pipeline_dir)
import mha_menu
mha_menu.register()
```
> ⚠️ `startup_unreal.py`는 매 Python 인터프리터 기동 시, `init_unreal.py`는 에디터/커맨드릿 컨텍스트에서 실행. 메뉴 등록은 `init_unreal.py`가 맞다.
> ⚠️ 헤드리스(`-run=pythonscript`)에서는 메인 메뉴가 없으므로 `find_menu("LevelEditor.MainMenu")`가 `None` → 조용히 skip 처리해야 한다(현재 코드가 그렇게 방어).

---

## 2. 핵심 개념 — 메뉴/툴바는 전부 `UToolMenus`로 통합됨

UE4.24부터 에디터의 **모든 메뉴·툴바**는 `UToolMenus` 서브시스템이 데이터로 관리한다. 각 메뉴는 **문자열 이름(경로)** 으로 식별된다.

**주요 메뉴 이름(경로):**

| 이름 | 위치 |
|---|---|
| `LevelEditor.MainMenu` | 레벨 에디터 상단 메뉴바 |
| `LevelEditor.MainMenu.Tools` | 상단 `Tools` 드롭다운 |
| `LevelEditor.MainMenu.Build` | 상단 `Build` 드롭다운 |
| `LevelEditor.LevelEditorToolBar.PlayToolBar` | 상단 툴바(플레이 근처) |
| `LevelEditor.LevelEditorToolBar.User` | 상단 툴바 사용자 섹션 |
| `ContentBrowser.AssetContextMenu` | 콘텐츠 브라우저 애셋 우클릭 |
| `ContentBrowser.FolderContextMenu` | 콘텐츠 브라우저 폴더 우클릭 |
| `LevelEditor.ActorContextMenu` | 뷰포트 액터 우클릭 |
| `LevelEditor.StatusBar.ToolBar` | 하단 상태 바 |

> 메뉴 이름을 모르겠으면: **Editor Preferences → General → Miscellaneous → "Display UI Extension Points"** 를 켜면 에디터 UI 위에 각 확장 지점의 이름이 초록색으로 표시된다.

**용어:**
- **Menu** (`ToolMenu`): 메뉴/툴바 하나.
- **Section** (`ToolMenuSection`): 메뉴 내 구분 그룹(라벨 있는 소제목).
- **Entry** (`ToolMenuEntry`): 실제 버튼/체크박스/서브메뉴 항목.
- **Owner**: 등록 주체. 소유자 이름으로 일괄 해제(`unregister_owner_by_name`) 가능 → **핫리로드/재등록 시 중복 방지의 핵심**.

---

## 3. 경로 A — ToolMenus (Python) ★ 현재 사용 중

### 3.1 클래스 시그니처

**`unreal.ToolMenus`** (싱글톤 서브시스템):
```python
ToolMenus.get() -> ToolMenus                       # 싱글톤 취득

# 인스턴스 메서드
find_menu(name: Name) -> ToolMenu                  # 등록/확장된 메뉴 찾기 (없으면 None)
extend_menu(name: Name) -> ToolMenu                # 소유권 없이 메뉴 확장 (없어도 호출 OK)
register_menu(name, parent='None',
              type=MultiBoxType.MENU, warn_if_already_registered=True) -> ToolMenu
is_menu_registered(name: Name) -> bool
remove_menu(menu_name: Name) -> None
remove_entry(menu_name, section, name) -> None
remove_section(menu_name, section) -> None
set_section_label(menu_name, section_name, label) -> None
set_section_position(menu_name, section_name, other_section_name, position_type) -> None
refresh_all_widgets() -> None                      # 다음 tick에 위젯 재빌드 (등록 후 필수)
refresh_menu_widget(name: Name) -> bool
unregister_owner_by_name(owner_name: Name) -> None # ★ owner 기준 일괄 해제

# 클래스 메서드
ToolMenus.add_menu_entry_object(menu_entry_object: ToolMenuEntryScript) -> bool
ToolMenus.remove_menu_entry_object(menu_entry_object) -> bool
ToolMenus.find_context(context, class_) -> Object
```

**`unreal.ToolMenu`** (메뉴 하나):
```python
add_section(section_name: Name, label='', insert_name='None',
            insert_type=ToolMenuInsertType.DEFAULT,
            alignment=ToolMenuSectionAlign.DEFAULT) -> None
add_menu_entry(section_name: Name, args: ToolMenuEntry) -> None
add_menu_entry_object(object: ToolMenuEntryScript) -> None
add_sub_menu(owner: Name, section_name: Name, name: Name,
             label: Text, tool_tip='') -> ToolMenu     # 하위 드롭다운 생성
add_dynamic_section(section_name: Name, object: ToolMenuSectionDynamic) -> None
```

**`unreal.ToolMenuEntry`** (항목 하나):
```python
# 생성자 (키워드 인자)
ToolMenuEntry(name='None', owner=..., type=MultiBlockType.MENU_ENTRY,
              user_interface_action_type=..., insert_position=..., ...)

set_label(label: Text) -> None
set_tool_tip(tool_tip: Text) -> None
set_string_command(type: ToolMenuStringCommandType, custom_type: Name, string: str) -> None
set_icon(style_set_name: Name, style_name='None', small_style_name='None') -> None
```

**주요 enum:**
```python
unreal.MultiBlockType:        MENU_ENTRY, MENU_SEPARATOR, HEADING, TOOL_BAR_BUTTON, ...
unreal.ToolMenuStringCommandType:  COMMAND(콘솔 명령), PYTHON(파이썬 코드), CUSTOM
unreal.MultiBoxType:          MENU, TOOL_BAR, VERTICAL_TOOL_BAR, ...
```
> `set_string_command(PYTHON, "", string="...")` — 클릭 시 지정 Python **문자열을 그대로 실행**. 콜백 UObject의 수명 관리가 필요 없어 **가장 견고**하다(MHA가 채택한 방식).

### 3.2 최소 패턴 — 상단 'MHA' 드롭다운 (현재 구현)

```python
# Content/Python/pipeline/mha_menu.py (현재 구현 요약)
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
    if not main:                                   # 헤드리스 방어
        unreal.log_warning("[MHA] MainMenu 없음(headless) - skip")
        return
    mha = main.add_sub_menu(main.menu_name, "", "MHA", "MHA")   # 상단바에 'MHA' 추가
    mha.add_section("MHASection", unreal.Text("MotionCapture Pipeline"))
    _add_entry(mha, "SolveExport", "Solve + Export (선택 CD)", "...",
               "import mha_tool; mha_tool.solve_export_selected()")
    menus.refresh_all_widgets()                    # ★ 재빌드
```

### 3.3 중복 등록 방지 (핫리로드/재실행 안전)

에디터에서 스크립트를 다시 돌리거나 핫리로드하면 메뉴가 중복된다. **등록 전에 owner로 해제**하면 멱등(idempotent)해진다:
```python
def register():
    menus = unreal.ToolMenus.get()
    menus.unregister_owner_by_name(_MENU_OWNER)    # ★ 먼저 기존 것 제거
    # ... (아래에서 owner를 지정해 등록)
```
> ToolMenuEntry 생성 시 `owner`를 `_MENU_OWNER`로 지정하거나 `FToolMenuOwnerScoped`(C++) 사용. Python에서는 `unregister_owner_by_name` + 재등록 조합이 실무적으로 가장 간단.

### 3.4 툴바 버튼 / 아이콘 / 서브메뉴

- **툴바 버튼:** `extend_menu("LevelEditor.LevelEditorToolBar.PlayToolBar")` 로 얻은 메뉴에 `type=MultiBlockType.TOOL_BAR_BUTTON` 엔트리 추가.
- **아이콘:** `entry.set_icon("EditorStyle", "GraphEditor.Macro_16x")` 처럼 스타일셋/스타일명 지정. (스타일명은 `EditorStyleSlateStyle` 참고, `Starship`/`EditorStyle` 셋에 다수 존재.)
- **동적 항목:** `add_dynamic_section` + `ToolMenuSectionDynamic` 서브클래스로 열 때마다 항목을 코드로 생성(예: 큐 상태에 따라 항목 변동).

### 3.5 클릭 콜백을 "문자열"이 아니라 "함수"로 받고 싶을 때

`ToolMenuEntryScript`를 상속해 `execute()`/`can_execute()`/`get_label()` 등을 오버라이드하면 실제 UObject 콜백으로 동작한다. 단, **인스턴스 수명을 전역에 유지**해야 GC되지 않는다. MHA는 문자열 커맨드로 충분하므로 권장하지 않음.

---

## 4. 경로 A(C++) — 게임/에디터 모듈에서 ToolMenus 확장

Python 없이 **C++ 에디터 모듈**에서 메뉴를 넣는 정석 패턴. 플러그인 배포·컴파일 타임 안정성이 필요할 때.

```cpp
// FMyEditorModule::StartupModule()
UToolMenus::RegisterStartupCallback(
    FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FMyEditorModule::RegisterMenus));

// FMyEditorModule::ShutdownModule()
UToolMenus::UnRegisterStartupCallback(this);
UToolMenus::UnregisterOwner(this);          // owner로 일괄 해제

void FMyEditorModule::RegisterMenus()
{
    FToolMenuOwnerScoped OwnerScoped(this);                       // ★ 이 스코프의 등록물에 owner 자동 부여
    UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Tools");
    FToolMenuSection& Section = Menu->FindOrAddSection("MHA");
    Section.AddMenuEntry(
        "MHA_Run",
        LOCTEXT("MHA_Run", "MHA: Run Pipeline"),
        LOCTEXT("MHA_Run_Tip", "선택 CD 솔브+익스포트"),
        FSlateIcon(),
        FUIAction(FExecuteAction::CreateRaw(this, &FMyEditorModule::OnRun)));
}
```
- **`FUICommandList` + `FUICommandInfo`**: 키보드 단축키까지 붙이려면 `TCommands<>` 로 커맨드를 선언하고 `AddMenuEntryWithCommandList()` 사용.
- **모듈 타입**: `.uplugin`/`.Build.cs`에서 `"Editor"` 타입 모듈 + `ToolMenus`, `UnrealEd`, `Slate`, `SlateCore`, `EditorStyle` 의존성.

> MHA는 Python으로 충분하지만, 파이프라인을 **플러그인으로 패키징**해 다른 프로젝트에 재사용하려면 이 경로가 낫다.

---

## 5. 경로 B — Editor Utility Widget (EUW, 대시보드/폼)

버튼 하나가 아니라 **패널**(목록·진행바·여러 버튼·입력 필드)이 필요할 때. UMG로 UI를 그리고, 로직은 Python/BP로 둔다. → **Phase 6 운영 대시보드**에 적합.

**생성:** 콘텐츠 브라우저 우클릭 → **Editor Utilities → Editor Utility Widget** (`WidgetBlueprint` 기반, 부모 `EditorUtilityWidget`). 더블클릭 후 **Run** 하거나 우클릭 **Run Editor Utility Widget**.

**`unreal.EditorUtilitySubsystem`** — 코드로 탭 열기/조회/닫기:
```python
subsystem = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
bp = unreal.load_asset("/Game/Python/EUW_MHA_Dashboard")   # EditorUtilityWidgetBlueprint

# 탭으로 띄우기
widget = subsystem.spawn_and_register_tab(bp)
widget, tab_id = subsystem.spawn_and_register_tab_and_get_id(bp)   # id도 필요할 때

# 조회 / 닫기
subsystem.does_tab_exist(tab_id) -> bool
subsystem.close_tab_by_id(tab_id) -> bool
subsystem.find_utility_widget_from_blueprint(bp) -> EditorUtilityWidget  # 떠있지 않으면 None

# 태스크 실행 (EditorUtilityTask 기반 장기 작업)
subsystem.register_and_execute_task(new_task, optional_parent_task=None)
subsystem.try_run(asset) -> bool     # Blutility 실행
```

**Python 로직 연결 2가지:**
1. 위젯 BP의 버튼 `OnClicked` → **Execute Python Command** 노드로 `import mha_tool; mha_tool.xxx()` 실행 (가장 간단).
2. 위젯을 `spawn_and_register_tab`로 얻은 뒤 Python에서 위젯 프로퍼티/함수를 직접 호출.

> EUW는 **비모달**(다른 작업과 공존)이라 MHA 큐/로그 대시보드에 적합. 단, `.uasset`(위젯 BP)이라 순수 텍스트 diff가 안 됨 — 계획서대로 **로직은 `.py`에, EUW는 표시/버튼만** 두는 원칙 유지.

---

## 6. 경로 C — Scripted Actions (선택 애셋/액터 우클릭 메뉴)

콘텐츠 브라우저에서 **CD를 우클릭 → 솔브+익스포트** 같은 흐름을 만들 때. 상단 메뉴(경로 A)의 대안/보완.

- **부모 클래스**
  - `AssetActionUtility` — 콘텐츠 브라우저 **애셋** 우클릭.
  - `ActorActionUtility` — 레벨/아웃라이너 **액터** 우클릭.
- **노출 규칙:** 해당 클래스의 함수/이벤트에 **`Call in Editor` 체크**가 켜져 있으면 우클릭 메뉴 항목으로 자동 등장. 입력 파라미터가 있으면 실행 시 값 입력 프롬프트가 뜬다.
- **대상 제한:** Class Defaults의 **Supported Classes**(+) 에 대상 클래스(예: `FootageCaptureData`)를 추가하면 그 타입에서만 메뉴가 뜬다.
- **자동 등록:** 엔진이 이 부모 클래스의 파생을 **스캔**해 등록 → 별도 `init_unreal` 코드 불필요.
- **상태:** UE 5.8 기준 **Beta**.

**Python으로 만드는 법** — Python 클래스가 `AssetActionUtility`를 상속하면 된다:
```python
import unreal

@unreal.uclass()
class MHASolveAction(unreal.AssetActionUtility):
    @unreal.ufunction(override=True)
    def get_supported_class(self):
        return unreal.FootageCaptureData        # 이 타입에서만 노출

    @unreal.ufunction(ret=None, params=[], meta=dict(CallInEditor=True))
    def solve_and_export(self):
        import mha_tool
        mha_tool.solve_export_selected()        # 선택 애셋 대상 동작
```
> `EditorUtilityLibrary.get_selected_assets()`로 우클릭 대상 애셋을 가져온다(현재 `mha_tool.py`가 이미 사용). Python `@unreal.uclass` 방식은 재기동 시 재등록되므로 `init_unreal.py`에서 import만 되면 유지된다.

---

## 7. 경로 D/E — Interactive Tools Framework & Scriptable Tools (뷰포트 모달 툴)

> **MHA에는 불필요**하지만, MetaHuman/Modeling Mode가 쓰는 "진짜 에디터 툴" 프레임워크라 참고용으로 정리. 뷰포트에서 **마우스로 상호작용**하는 툴(클릭/드래그/기즈모)을 만들 때만 필요.

**D) Interactive Tools Framework (ITF, C++)** — `InteractiveToolsFramework` 런타임 모듈 + `EditorInteractiveToolsFramework` 에디터 모듈.
- `UInteractiveToolsContext` — 툴/기즈모/InputRouter가 사는 최상위 "세계".
- `UInteractiveTool` — **모달** 툴 하나(활성 중 다른 툴 불가). `Setup`/`Render`/`Tick`/`Shutdown` 등 소수 API.
- `UInteractiveToolBuilder` — 툴 생성 팩토리.
- `UEdMode` (신형 에디터 모드) — 툴 팔레트를 소유하고 ITF와 `FEdMode`를 잇는다. `UEdModeInteractiveToolsContext`가 중개. `Enter`/`Exit`/`GetModeCommands`/툴 등록.
- 등록 흐름: `UEdMode::RegisterTool(Command, ToolIdentifier, Builder)` → 좌측 툴 팔레트에 버튼.

**E) Scriptable Tools System (BP, UE 5.8 Beta)** — **C++ 없이** ITF 툴을 Blueprint로.
- 플러그인: **Scriptable Tools Editor Mode** (Edit → Plugins → "scriptable tools").
- 부모 클래스:
  - `UScriptableInteractiveTool` — 모든 스크립터블 툴의 토대.
  - `UScriptableModularBehaviorTool` — **권장**(마우스/키보드 입력 처리 포함).
  - 에디터 전용: `UEditorScriptableInteractiveTool`, `UEditorScriptableModularBehaviorTool`.
- 구현 이벤트: `On Script Setup`, `On Script Tick`, `On Script Shutdown`. 설정으로 Tool Name/Category/Icon/Shutdown Type. `Add Property Set of Type`로 사용자 설정 노출.
- 기능: 3D 기즈모(TRS), 라인/포인트/삼각형 드로잉, 클릭·드래그·호버·키보드 입력, 사용자 메시지.

**모달 툴 vs EUW:** 모달 툴은 활성 중 다른 툴 불가(상태 안전, 저장/PIE 전에 자동 shutdown). EUW는 비모달 패널. 뷰포트 상호작용 = 툴, 대시보드/폼 = EUW.

---

## 8. 에디터 툴에서 자주 쓰는 지원 라이브러리

파이프라인 함수가 실제로 호출하는 헬퍼들 (MHA에서 이미 다수 사용 중):

**선택/애셋:**
```python
unreal.EditorUtilityLibrary.get_selected_assets() -> [Object]          # 콘텐츠 브라우저 선택
unreal.EditorUtilityLibrary.get_selected_asset_data() -> [AssetData]
unreal.EditorAssetLibrary.list_assets(dir, recursive=True, include_folder=False) -> [str]
unreal.EditorAssetLibrary.load_asset(path) / save_asset(path) / does_asset_exist(path)
unreal.load_asset(path)                                                # 짧은 헬퍼
unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, path, cls, factory)
unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_all(only_if_dirty=True)
```

**진행 표시 (긴 작업):**
```python
with unreal.ScopedSlowTask(total_frames, "MHA: Solve + Export") as task:
    task.make_dialog(True)                       # 취소 버튼 있는 다이얼로그
    task.enter_progress_frame(1, "Solving ...")
    if task.should_cancel(): break
```

**대화상자 / 알림:**
```python
unreal.EditorDialog.show_message("MHA", "완료", unreal.AppMsgType.OK)  # 모달 팝업
unreal.EditorDialog.open_directory_dialog(title, default_path)         # 폴더 선택(반환형 5.8 실확인)
unreal.EditorDialog.open_file_dialog(...)
# 비모달 토스트
n = unreal.EditorNotificationController()  # 또는 SystemLibrary 로그
unreal.log("...") / unreal.log_warning("...") / unreal.log_error("...")
```

**경로/시스템:**
```python
unreal.SystemLibrary.get_project_content_directory()
unreal.SystemLibrary.launch_url(f"file:///{path}")     # 탐색기로 폴더 열기
unreal.Paths.get_base_filename(path)
```

---

## 9. 헤드리스(커맨드릿)와 에디터 툴의 경계

- 에디터 툴(메뉴/EUW)은 **UI가 있는 에디터 세션**에서만 표시된다. 헤드리스 `-run=pythonscript`에서는 `LevelEditor.MainMenu`가 없어 `find_menu`가 `None` → **메뉴 등록은 반드시 방어적으로 skip**(현재 구현됨).
- 따라서 파이프라인 로직은 **UI와 분리된 순수 함수**(`step2_solve.create_and_process`, `step3_export.export_anim_sequence`, `batch_mha.process_one`)로 두고, **에디터 버튼과 헤드리스 배치가 같은 함수를 호출**하게 한다(계획서의 "동치 보장" 원칙과 일치).
- 헤드리스 배치는 GPU RHI가 필요하므로 `-AllowCommandletRendering` 필수(계획서 §헤드리스 필수 플래그). 에디터 버튼 실행은 RHI가 이미 살아 있어 이 이슈와 무관.

---

## 10. MHA 프로젝트 권장 적용

1. **유지:** 현재 `ToolMenus` 상단 'MHA' 드롭다운(경로 A) — Phase 0~5 버튼에 최적. `mha_menu.register()`에 `unregister_owner_by_name("MHAPipelineMenu")`를 **맨 앞에 추가**해 재실행 시 중복 방지.
2. **보완(선택):** CD/Performance를 **우클릭**해 바로 처리하고 싶으면 경로 C(`AssetActionUtility`, `get_supported_class = FootageCaptureData`)를 추가. 상단 메뉴와 병행 가능.
3. **Phase 6:** 큐/재시도/로그 **대시보드**는 경로 B(EUW) — 로직은 `tool_actions.py`/`batch_mha.py`에 두고 EUW는 목록·버튼만. `EditorUtilitySubsystem.spawn_and_register_tab`으로 코드에서 열 수 있게.
4. **비권장:** 경로 D/E(뷰포트 모달 툴) — 파일 기반 배치 파이프라인에는 상호작용 요소가 없어 불필요.

---

## 출처 (Epic 공식 문서 우선, UE 5.8 기준)

- [unreal.ToolMenus — Python API](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/ToolMenus)
- [unreal.ToolMenu — Python API](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/ToolMenu)
- [unreal.ToolMenuEntry — Python API](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/ToolMenuEntry)
- [unreal.EditorUtilitySubsystem — Python API](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/EditorUtilitySubsystem)
- [Scripted Actions in Unreal Engine 5.8](https://dev.epicgames.com/documentation/unreal-engine/scripted-actions-in-unreal-engine)
- [unreal.AssetActionUtility / ActorActionUtility — Python API](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/ActorActionUtility)
- [Scriptable Tools System in Unreal Engine (5.8 Beta)](https://dev.epicgames.com/documentation/en-us/unreal-engine/scriptable-tools-system-in-unreal-engine)
- [InteractiveToolsFramework — UE 5.8 API](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/InteractiveToolsFramework)
- [EditorInteractiveToolsFramework — UE 5.8 API](https://dev.epicgames.com/documentation/unreal-engine/API/Editor/EditorInteractiveToolsFramework)
- [Adding Custom Buttons to the Unreal Editor Toolbars and Menus (minifloppy.it, C++ ToolMenus)](https://minifloppy.it/posts/2024/adding-custom-buttons-unreal-editor-toolbars-menus/)
- [Using tool menus for editor extension — Unreal Community Wiki](https://unrealcommunity.wiki/using-tool-menus-for-editor-extension-ipmtgt9o)
- [Editor Utility Widgets 레시피 — bralkor/unreal_python_recipe_book](https://github.com/bralkor/unreal_python_recipe_book/blob/5.2/documentation/07_editor_utility_widgets.md)
</content>
</invoke>
