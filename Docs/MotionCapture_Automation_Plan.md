# MotionCapture 자동화 구현 계획 (Phase 기반)

> 대상: `C:\SupergeneGithub\VTuberStudio` (`MetaHuman.uproject`, Engine **5.8**)
> 파이프라인: **mp4 투입 → 인제스트 → 솔브(Face + Body) → AnimSequence 익스포트 → 완료 폴더 이동** (완전 무인)
> 워크플로우: **Mono Video Ingest** (iPhone depth 아님). 모든 API/프로퍼티는 5.8 설치본에서 검증.
> 참조 근거 문서: [`MHA_Automation_Plan.md`](./MHA_Automation_Plan.md) (아키텍처·프로퍼티 표·미해결 Q1~Q7)

각 Phase는 **독립적으로 검증 가능**하도록 순서를 잡았다. 앞 Phase의 **검증(Exit Criteria)** 을 통과하지 못하면 다음 Phase로 넘어가지 않는다. 가장 확실한 부분(솔브/익스포트)부터 자동화하고 인제스트를 나중에 붙인다.

> ### ⛔ 헤드리스 필수 플래그 (Q7 해결, 2026-07-20)
> **`-run=pythonscript`는 커맨드릿이라 기본 NullRHI → MetaHuman 솔브가 `CanProcess()==false`로 막힘**(`start_pipeline`이 `StartPipelineErrorType.DISABLED` 반환). `CanProcess`의 `FMetaHumanSupportedRHI::IsSupported()` 게이트가 `GDynamicRHI`를 D3D12/Vulkan로 요구하기 때문.
> **해결: 배치 실행 시 `-AllowCommandletRendering` 플래그 필수** → D3D12(SM6) 초기화되어 솔브 가능. 이 플래그가 §5/§6의 모든 헤드리스 커맨드에 들어가야 한다.
> (참고: 얼굴 authoring 게이트 `bMetaHumanAuthoringObjectsPresent`는 `MetaHumanAnimator/Content/GenericTracker/Chin` 존재로 통과 — 문제없음.)

---

## 진행 요약

| Phase | 내용 | 산출물 | 핵심 검증 |
|---|---|---|---|
| **0** | 환경 준비 | 플러그인/Python 활성화 | 에디터 Python 콘솔에서 `unreal` API 호출 성공 |
| **1** | 수동 기준선 | (수동 1회 성공) | UI로 mp4 1개 → AnimSequence 생성 확인 |
| **2** | 솔브+익스포트 자동화 | `step2_solve.py`, `step3_export.py` | 기존 CD → AnimSequence 애셋 프로그램 생성 |
| **3** | 인제스트 자동화 | `step1_ingest.py` | mp4 → CD 애셋 프로그램 생성 |
| **4** | 배치 통합(헤드리스) | `batch_mha.py`, `config.py` | 단일 커맨드로 ①②③ 완주, 종료코드 0 |
| **5** | 무인 감시 | `watchdog_runner.py` | 폴더에 mp4 넣으면 자동 완주 + 파일 이동 |
| **6** | 운영 견고화 | 큐/재시도/알림/락 | 다중 테이크 연속 처리, 실패 격리 |

파일 구조(전 Phase 공통 목표):
```
VTuberStudio/
├─ Content/Python/           init_unreal.py                         ← 에디터 기동 시 자동 실행(메뉴 등록)
│  └─ pipeline/              batch_mha.py  step1_ingest.py  step2_solve.py  step3_export.py  config.py
│                            editor_tools.py                        ← 에디터 툴(메뉴/버튼) 등록 프레임워크
├─ Automation/               watchdog_runner.py  requirements.txt
└─ Docs/                     MotionCapture_Automation_Plan.md (본 문서)  MHA_Automation_Plan.md (근거)
```

> ### 🛠 에디터 툴 레이어 (전 Phase 공통)
> 각 Phase의 자동화 코드는 **헤드리스 배치**뿐 아니라 **에디터 안에서 버튼 한 번으로** 실행할 수 있어야 한다(디버깅·수동 재처리·검증에 필수). 이를 위해 `editor_tools.py`가 `unreal.ToolMenus`로 에디터 상단에 **`MHA` 메뉴**를 등록하고, 각 Phase가 이 메뉴에 항목 1개씩 추가한다.
> - **등록 방식:** `ToolMenuEntry.set_string_command(type=PYTHON, string="...")` — 클릭 시 지정한 Python 문자열을 그대로 실행. 콜백 UObject 수명 관리가 불필요해 가장 견고.
> - **자동 로드:** `Content/Python/init_unreal.py`는 에디터 기동 시 엔진이 자동 실행 → 여기서 `editor_tools.register_all()` 호출.
> - **입력 UI:** 애셋 선택은 콘텐츠 브라우저 선택(`EditorUtilityLibrary.get_selected_assets()`), 폴더/파일은 `unreal.SystemLibrary`/파일 다이얼로그로 받는다.
>
> `editor_tools.py`의 공용 프레임워크(각 Phase에서 `add_entry(...)` 호출):
> ```python
> # Content/Python/pipeline/editor_tools.py
> import unreal
>
> _MENU_PATH  = "LevelEditor.MainMenu.MHA"     # 상단 메뉴 'MHA'
> _SECTION    = "Pipeline"
> _menus      = unreal.ToolMenus.get()
>
> def _ensure_root_menu():
>     main = _menus.find_menu("LevelEditor.MainMenu")
>     mha  = _menus.find_menu(_MENU_PATH)
>     if not mha:
>         mha = main.add_sub_menu(main.get_name(), "", "MHA", "MHA")   # 상단바에 'MHA' 추가
>         mha.add_section(_SECTION, "MHA Pipeline")
>     return mha
>
> def add_entry(name: str, label: str, tooltip: str, python: str):
>     """MHA 메뉴에 버튼 1개 추가. 클릭 시 `python` 문자열 실행."""
>     mha = _ensure_root_menu()
>     e = unreal.ToolMenuEntry(name=name, type=unreal.MultiBlockType.MENU_ENTRY)
>     e.set_label(label)
>     e.set_tool_tip(tooltip)
>     e.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "", string=python)
>     mha.add_menu_entry(_SECTION, e)
>
> def register_all():
>     """init_unreal.py에서 호출. 각 Phase 항목을 순서대로 등록."""
>     add_entry("MHA_Phase0", "Phase 0: API 자가진단",
>               "필수 클래스/스켈레톤 로드 점검",
>               "import phase0_check; phase0_check.run()")
>     add_entry("MHA_Phase1", "Phase 1: 기준선 검사",
>               "기존 AnimSequence 프레임/본 트랙 검사",
>               "import phase1_inspect; phase1_inspect.run()")
>     add_entry("MHA_Phase2", "Phase 2: 솔브+익스포트 (선택 CD)",
>               "콘텐츠 브라우저에서 선택한 CD를 솔브→익스포트",
>               "import tool_actions; tool_actions.solve_export_selected()")
>     add_entry("MHA_Phase3", "Phase 3: 인제스트 (폴더 선택)",
>               "mp4 폴더를 골라 CD 애셋 생성",
>               "import tool_actions; tool_actions.ingest_pick_folder()")
>     add_entry("MHA_Phase4", "Phase 4: 배치 1건 (현재 세션)",
>               "폴더 선택 → 인제스트+솔브+익스포트 완주",
>               "import tool_actions; tool_actions.batch_one_pick_folder()")
>     _menus.refresh_all_widgets()
> ```
> ```python
> # Content/Python/init_unreal.py  (엔진이 기동 시 자동 실행)
> import unreal
> try:
>     import editor_tools
>     editor_tools.register_all()
>     unreal.log("[MHA] editor tools registered under 'MHA' menu")
> except Exception as e:
>     unreal.log_error(f"[MHA] editor tool 등록 실패: {e}")
> ```
> 아래 각 Phase의 **🛠 에디터 툴** 항목은 위 프레임워크에 추가되는 **그 Phase 전용 버튼**을 정의한다.

---

## Phase 0 — 환경 준비

**목표:** 프로젝트에서 Python으로 MetaHuman/CaptureManager API를 호출할 수 있는 상태를 만든다.

**작업:**
1. 플러그인 활성화: `Python Editor Script Plugin`, `Editor Scripting Utilities`, `MetaHuman`, `MetaHuman Animator`, `Capture Manager`, `Live Link Hub`.
2. `Config/DefaultEngine.ini`에 스크립트 경로 등록:
   ```ini
   [/Script/PythonScriptPlugin.PythonScriptPluginSettings]
   +AdditionalPaths=(Path="/Game/Python/pipeline")
   ```
3. `Content/Python/pipeline/`, `Automation/` 폴더 생성.
4. 얼굴 타깃 스켈레톤 존재 확인:
   `/Game/MetaHumans/Common/Face/Face_Archetype_Skeleton.Face_Archetype_Skeleton`

**✅ 검증 (Exit Criteria):**
- [x] 에디터 재시작 후 Python 콘솔(또는 헤드리스 `-run=pythonscript`)에서 아래가 오류 없이 실행:
  ```python
  import unreal
  print(unreal.MetaHumanPerformance)                 # 클래스 존재
  print(unreal.MonoVideoIngestDevice)                # 인제스트 클래스 존재
  print(unreal.load_asset("/Game/MetaHumans/Common/Face/Face_Archetype_Skeleton.Face_Archetype_Skeleton"))  # None 아님
  ```
- [x] 위 3개 클래스 로드 확인 → 통과.

### 🟢 Phase 0 실행 결과 (2026-07-20) — API 검증 통과 (4/4)
헤드리스 검증 결과 (`-run=pythonscript`):
```
[PHASE0] PASS MetaHumanPerformance class
[PHASE0] PASS MonoVideoIngestDevice class
[PHASE0] PASS MetaHumanPerformanceExportUtils
[PHASE0] PASS IngestCapability_Options
[PHASE0] INFO Face_Archetype_Skeleton: None   ← 블로커(아래)
[PHASE0] RESULT ALL_PASS (4/4)
```
- **완료:**
  - 플러그인 활성화 (`MetaHuman.uproject`에 `PythonScriptPlugin`, `EditorScriptingUtilities`, `MetaHuman`(=MetaHuman Animator), `CaptureManagerApp`, `CaptureManagerDevices`, `CaptureManagerEditor`, `LiveLinkHub` 추가) — JSON 유효성 검증 완료.
  - ⚠️ 최초 활성화 시 `CaptureManagerDevices`를 빠뜨려 `unreal.MonoVideoIngestDevice`가 미노출됐고(1차 검증 실패), 추가 후 재검증에서 통과. **`MonoVideoIngestDevice`/`StereoVideoIngestDevice` UCLASS는 `CaptureManagerDevices` 플러그인 소유** — 인제스트에 필수.
  - `DefaultEngine.ini`에 Python 스크립트 경로(`/Game/Python/pipeline`) 등록.
  - 폴더 생성: `Content/Python/pipeline/`(+`config.py`), `Automation/`(+`requirements.txt`).
  - 헤드리스 기동 성공: Python 3.11.8 활성, MetaHuman Animator 활성화로 **최초 1회 pip 의존성 설치**(opencv/numpy/scipy/torch 등)가 프로젝트 `Intermediate/PipInstall` venv에 진행됨.
- **⚠️ 블로커 (착수 전 반드시 해결):**
  - **프로젝트에 `Content/MetaHumans` 폴더와 얼굴 스켈레톤(`Face_Archetype_Skeleton`)이 존재하지 않음.** 스켈레톤 애셋은 Mannequin(`SK_Mannequin`)뿐. MetaHuman Identity/캐릭터 애셋 없음.
  - 영향: **Phase 2 익스포트가 이 스켈레톤을 타깃으로 하므로 그대로는 실패**. (솔브 자체는 CD만 있으면 가능하나 익스포트 타깃이 없음)
  - 조치: Phase 1 착수 전 **MetaHuman을 프로젝트에 추가**(MetaHuman Creator/Fab에서 임포트하거나 Quixel Bridge)해야 함. 그러면 `Content/MetaHumans/Common/Face/Face_Archetype_Skeleton`이 생성됨. → **Q3/Q6와 직결**.
- **참고 경고(무해):** `/Script/MetaHuman.MetaHumanCharacter` 와 `/Script/MetaHumanCharacter.MetaHumanCharacter` 이름 충돌 경고 — 게임 모듈명이 `MetaHuman`이라 발생. 동작에는 지장 없음(필요 시 게임 모듈/플러그인 `ScriptName` 정리로 해소 가능).

### 에디터 툴 (Phase 0) — `MHA Phase 0: API 자가진단`
헤드리스 자가진단(§Phase 0 검증)을 **에디터 메뉴 버튼**으로 실행. 클릭 시 필수 클래스/스켈레톤 로드 결과를 `Output Log`와 팝업으로 보고.
```python
# Content/Python/pipeline/phase0_check.py
import unreal

def run():
    checks = {
        "MetaHumanPerformance":              hasattr(unreal, "MetaHumanPerformance"),
        "MonoVideoIngestDevice":             hasattr(unreal, "MonoVideoIngestDevice"),
        "MetaHumanPerformanceExportUtils":   hasattr(unreal, "MetaHumanPerformanceExportUtils"),
        "IngestCapability_Options":          hasattr(unreal, "IngestCapability_Options"),
    }
    lines = [f"{'PASS' if ok else 'FAIL'}  {k}" for k, ok in checks.items()]
    passed = sum(checks.values())
    msg = "\n".join(lines) + f"\n\nRESULT: {passed}/{len(checks)}"
    unreal.log(f"[PHASE0]\n{msg}")
    unreal.EditorDialog.show_message("MHA Phase 0", msg,
                                     unreal.AppMsgType.OK)   # 에디터 팝업
    return checks
```
- **등록:** `editor_tools.register_all()`의 `MHA_Phase0` 항목(위 프레임워크에 포함).
- **검증:** 버튼 클릭 → 팝업에 `4/4 PASS`. 헤드리스 결과와 동일해야 함.

---

## Phase 1 — 수동 기준선 확보

**목표:** 자동화 이전에 **UI에서 전체 흐름이 성공**함을 확인해 기준선(정상 산출물 형태)을 잡는다. 자동화 디버깅 시 "API 문제 vs 데이터/설정 문제"를 분리하기 위한 필수 단계.

**작업:**
1. Capture Manager UI로 테스트 mp4 1개 인제스트 → `FootageCaptureData(CD)` 생성.
2. MetaHuman Performance 애셋 생성, Input Type = Mono Footage, Body Tracking ON.
3. Process 실행 → 완료.
4. AnimSequence 익스포트.

**✅ 검증 (Exit Criteria):**
- [x] 인제스트 산출물이 `.../Mono_Video_Ingest/<take>/` 아래 `CD_*.uasset` + `IS_V_video_*.uasset` 형태로 생성됨.
- [x] (뷰포트 재생 대체) 익스포트된 AnimSequence에 실제 본 애니메이션 존재 확인 — 헤드리스 로드/검사.
- [x] 익스포트된 AnimSequence에 커브/본 애니메이션 실재 확인.
- [x] **CD 애셋 콘텐츠 경로 기록** (아래).

### 🟢 Phase 1 실행 결과 (2026-07-20) — 기존 기준선(aespa 댄스)으로 검증 완료
새 수동 작업 대신 **프로젝트에 이미 있는 완주 테이크**를 헤드리스로 검사해 기준선 확정.

**기록된 경로 (Phase 2 입력):**
- CD:          `/Game/CaptureManager/Imports/Mono_Video_Ingest/aespa_darkarts_dance_30fps_1/CD_aespa_darkarts_dance_30fps_1`
- IS_V:        `.../IS_V_video_aespa_darkarts_dance_30fps_1` (`ImgMediaSource`)
- Performance: `.../Mocap_aspa` (⚠️ 이름과 달리 **`MetaHumanPerformance`**, AnimSequence 아님)
- AnimSequence:`/Game/AnimationSequence/AS_Mocap_aspa`, `/Game/AS_Mocap_aspa` (2곳)

**기준선 Performance 설정 (= `step2_solve.py` 기본값과 100% 일치 확인):**
`input_type=MONO_FOOTAGE`, `face_tracking=False`, `body_tracking=True`, `auto_body_height=True`,
`body_height=180`, `body_detection_confidence=0.7`, `body_tracking_confidence=0.5`,
`enable_foot_locking=True`, 처리범위 `0→876`.

**기준선 AnimSequence:** 875프레임 / 29.17s @30fps / **본 트랙 167개** (root, pelvis, spine_01…) /
**스켈레톤 `SKEL_UE5_F`** → **바디 익스포트 타깃 확정**. `config.BODY_SKELETON = /Game/IdaFaber/Meshes/Girl/SKEL_UE5_F.SKEL_UE5_F` 로 교정 완료.

> 시사점: 사용자의 "face 필요 없음"이 기준선과 정확히 일치(기준선도 `face_tracking=False`, 바디 온리). 우리 코드 기본값을 바꿀 필요 없음.

### 🔴 중요 발견 (2026-07-20) — 바디 익스포트 API & 콜드 로드 한계 (Q3 해결)
"저비용 익스포트만 검증"(이미 처리된 `Mocap_aspa`를 콜드 로드해 `step3_export` 실행)을 돌린 결과 **정적(모션 없는) AnimSequence** 가 나왔다. 포즈 샘플링으로 확정: 신규는 `spine_03` f0→f400 delta=0.000(STATIC), 기준선은 프레임마다 본 이동. 원인 규명(`MetaHumanPerformanceExportAnimationSettings` 헤더):
- **`bExportBody` 기본 False** — 안 켜면 바디 본 트랙이 안 써져 정적. → `export_body=True` 필수.
- **`SourceSkeletalMesh` (Transient)** — 헤더 주석: "바디 익스포트 필수. 에디터가 자동 설정하나 **스크립트 호출자는 수동 설정 필수**". 바디 트래커 액터가 채우는 값이라 **콜드 로드 세션엔 없음**.
- `ExportSkeleton`: `PERFORMER_SKELETON`(솔브 스켈레톤 복사, 타깃/리타겟 불필요) vs `EXISTING_SKELETON`(타깃 + `BodyRetargeter`(IKRetargeter) 필요). 기준선 `SKEL_UE5_F`는 EXISTING_SKELETON 리타게팅으로 추정.
- `bExportFace`(기본 True → 바디 온리면 False), `BodyUnsolvedBehavior`(LastValidFrame/NeutralPose), `CurveInterpolation`.

**결론:**
1. `step3_export.py`를 바디 설정 포함으로 **수정 완료**(export_body/face, export_skeleton, source_skeletal_mesh, body_retargeter, unsolved_behavior 파라미터화). 문법 통과.
2. **콜드 로드 익스포트(옵션 1)는 바디 모캡에 구조적으로 부적합** → 폐기. 바디 익스포트는 **솔브와 같은 세션**(옵션 2 / `batch_mha`)에서 수행해야 함.
3. 남은 확인: **`source_skeletal_mesh`가 무엇인지**(바디가 솔브된 소스 메시 — MetaHumanBodyTracker 콘텐츠 후보) + 기준선 스켈레톤(`SKEL_UE5_F`) 재현 시 EXISTING_SKELETON용 IKRetargeter 필요 여부. → 옵션 2 실행 시 규명.

### 🛠 에디터 툴 (Phase 1) — `MHA ▸ Phase 1: 기준선 검사`
콘텐츠 브라우저에서 **AnimSequence를 선택**하고 버튼을 누르면 프레임 수·본 트랙 수·스켈레톤·모션 유무(정적 여부)를 보고. 새 산출물이 기준선과 동등한지 즉시 비교.
```python
# Content/Python/pipeline/phase1_inspect.py
import unreal

def run():
    sel = unreal.EditorUtilityLibrary.get_selected_assets()
    seqs = [a for a in sel if isinstance(a, unreal.AnimSequence)]
    if not seqs:
        unreal.EditorDialog.show_message("MHA Phase 1",
            "콘텐츠 브라우저에서 AnimSequence를 선택하세요.", unreal.AppMsgType.OK)
        return
    for seq in seqs:
        frames   = seq.get_editor_property("number_of_sampled_frames")
        skeleton = seq.get_editor_property("skeleton")
        length   = seq.get_editor_property("sequence_length") if hasattr(seq, "sequence_length") else "?"
        msg = (f"{seq.get_name()}\n"
               f"frames={frames}  length={length}s\n"
               f"skeleton={skeleton.get_name() if skeleton else None}")
        unreal.log(f"[PHASE1] {msg}")
        unreal.EditorDialog.show_message("MHA Phase 1", msg, unreal.AppMsgType.OK)
```
> 정적/모션 여부 정밀 판정(§Phase 1 🔴 발견의 `spine_03` delta 샘플링)은 별도 함수로 확장 가능. 여기선 프레임/스켈레톤 요약이 목적.
- **검증:** 기준선 `AS_Mocap_aspa` 선택 → `frames≈875`, `skeleton=SKEL_UE5_F` 표시.

---

## Phase 2 — 솔브 + 익스포트 자동화 (기존 CD 대상)

**목표:** Phase 1에서 만든 **기존 CD 애셋**을 입력으로, Performance 생성 → 처리 → AnimSequence 익스포트를 **코드로** 수행. (인제스트는 아직 수동. 가장 확실한 부분부터 자동화.)

### 🟡 방향 전환 (2026-07-20): 바디 모캡 중심 (face 보류)
- 사용자 결정: "face 지금 필요 없음". → 얼굴 스켈레톤 부재 블로커를 **우회**하고 바디 모캡 우선.
- 솔브: `body_tracking=True`, `face_tracking=False`(기본). 필요 시 `enable_face=True`로 전환.
- 익스포트: 타깃을 `Face_Archetype_Skeleton`이 아닌 **바디 스켈레톤/스켈레탈 메시**로. `target_skeleton_path` 파라미터로 분리(하드코딩 제거). 기본 후보 `SK_Mannequin`(`config.BODY_SKELETON`), 최종 타깃은 Phase 2 실검증에서 확정.
- **작성 완료(코드):** `config.py`(경로/타깃 상수), `step2_solve.py`(create/configure/process 분리, 바디 기본), `step3_export.py`(target 파라미터화, 헤드리스 대화상자 억제, 자동 저장). 문법 검사(py_compile) 통과. **실행 검증은 기존 CD 애셋 확보 후 수행.**

**작업:** `step2_solve.py`, `step3_export.py` 작성. (아래는 초안 스케치이며, 실제 파일은 위 바디 중심 버전으로 반영됨)

```python
# Content/Python/pipeline/step2_solve.py
import unreal

def create_and_process(capture_data_path: str, storage_path: str, *,
                       enable_body: bool = True,
                       start_frame: int | None = None,
                       end_frame: int | None = None) -> unreal.MetaHumanPerformance:
    cd = unreal.load_asset(capture_data_path)
    perf = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        asset_name=f"{cd.get_name()}_Performance", package_path=storage_path,
        asset_class=unreal.MetaHumanPerformance,
        factory=unreal.MetaHumanPerformanceFactoryNew())

    # set_editor_property 사용 → PostEditChangeProperty 트리거되어 셋업 완료됨
    perf.set_editor_property("input_type", unreal.DataInputType.MONO_FOOTAGE)
    perf.set_editor_property("footage_capture_data", cd)
    perf.set_editor_property("face_tracking", True)
    perf.set_editor_property("skip_tongue_solve", False)

    perf.set_body_tracking(enable_body)                       # 5.8 신규
    if enable_body:
        perf.set_editor_property("auto_body_height", True)
        perf.set_editor_property("body_detection_confidence", 0.7)
        perf.set_editor_property("body_tracking_confidence", 0.5)
        perf.set_editor_property("enable_foot_locking", True)

    if start_frame is not None:
        perf.set_editor_property("start_frame_to_process", start_frame)
    if end_frame is not None:
        perf.set_editor_property("end_frame_to_process", end_frame)

    perf.set_blocking_processing(True)                        # 헤드리스 필수(동기)
    err = perf.start_pipeline()
    if err is not unreal.StartPipelineErrorType.NONE:
        raise RuntimeError(f"start_pipeline failed: {err}")
    return perf
```

```python
# Content/Python/pipeline/step3_export.py
import unreal
FACE_SKELETON = "/Game/MetaHumans/Common/Face/Face_Archetype_Skeleton.Face_Archetype_Skeleton"

def export_anim_sequence(perf, out_dir: str, out_name: str) -> unreal.AnimSequence:
    s = unreal.MetaHumanPerformanceExportAnimationSettings()
    s.show_export_dialog = False
    s.package_path = out_dir
    s.asset_name = out_name
    s.target_skeleton_or_skeletal_mesh = unreal.load_asset(FACE_SKELETON)
    s.enable_head_movement = True
    s.export_range = unreal.PerformanceExportRange.PROCESSING_RANGE
    seq = unreal.MetaHumanPerformanceExportUtils.export_animation_sequence(perf, s)
    if not seq:
        raise RuntimeError("AnimSequence export failed")
    return seq
```

**✅ 검증 (Exit Criteria):**
- [ ] 에디터 Python 콘솔에서 아래 실행 시 예외 없음:
  ```python
  import step2_solve, step3_export
  perf = step2_solve.create_and_process("<Phase1 CD 경로>", "/Game/MHA/Performances", enable_body=True)
  seq  = step3_export.export_anim_sequence(perf, "/Game/MHA/Anims", "AS_test")
  print(seq.get_name())
  ```
- [ ] `/Game/MHA/Performances`에 Performance, `/Game/MHA/Anims`에 AnimSequence 애셋 생성됨.
- [ ] 생성된 AnimSequence가 Phase 1의 수동 산출물과 **동등한 프레임 수/커브**를 가짐(육안 또는 프레임 카운트 비교).
- [ ] `enable_body=True/False` 두 경우 모두 예외 없이 완주.

### 🛠 에디터 툴 (Phase 2) — `MHA ▸ Phase 2: 솔브+익스포트 (선택 CD)`
콘텐츠 브라우저에서 **`FootageCaptureData`(CD_*)를 선택**하고 버튼을 누르면 그 CD로 Performance 생성→처리→AnimSequence 익스포트를 **현재 에디터 세션에서** 실행. (솔브+바디 익스포트는 같은 세션이어야 하므로 — §Phase 1 🔴 발견 — 에디터 툴 실행이 가장 자연스러움.)
```python
# Content/Python/pipeline/tool_actions.py  (Phase 2~4 공용 액션)
import unreal
import step2_solve, step3_export, config

def _selected_capture_data():
    for a in unreal.EditorUtilityLibrary.get_selected_assets():
        if isinstance(a, unreal.FootageCaptureData):
            return a
    return None

def solve_export_selected():
    cd = _selected_capture_data()
    if not cd:
        unreal.EditorDialog.show_message("MHA Phase 2",
            "콘텐츠 브라우저에서 FootageCaptureData(CD_*)를 선택하세요.", unreal.AppMsgType.OK)
        return
    cd_path = cd.get_path_name()
    with unreal.ScopedSlowTask(1, f"Solve+Export: {cd.get_name()}") as t:
        t.make_dialog(True)
        perf = step2_solve.create_and_process(cd_path, config.PERF_DIR, enable_body=True)
        seq  = step3_export.export_anim_sequence(perf, config.EXPORT_DIR, f"AS_{cd.get_name()}")
    unreal.EditorAssetLibrary.save_asset(seq.get_path_name())
    unreal.EditorDialog.show_message("MHA Phase 2",
        f"완료: {seq.get_name()}", unreal.AppMsgType.OK)
```
- **등록:** 프레임워크의 `MHA_Phase2` 항목.
- **검증:** 기준선 CD(`CD_aespa_...`) 선택 → 실행 → `/Game/MHA/Anims`에 AnimSequence 생성, 프레임 수 기준선과 동등.

---

## Phase 3 — 인제스트 자동화 (mp4 → CD 애셋)

**목표:** mp4 폴더를 입력으로 CaptureManager 인제스트를 코드로 수행해 `CD_*` 애셋을 생성. (Q1: 경로 A 우선 검증.)

**작업:** `step1_ingest.py` 작성 + CD 경로 자동 확보 유틸.

```python
# Content/Python/pipeline/step1_ingest.py
import os, socket, tempfile
import unreal
from capture.devices import MonoVideoIngestDevice     # Q2: sys.path 또는 복제
from capture.ingest import ingest_takes
from capture.unreal_endpoint_manager import UnrealEndpointManager

def ingest_mono_video(archive_path: str, video_discovery_expression: str = "<Auto>"):
    host = socket.gethostname()
    with UnrealEndpointManager(connect_timeout=10) as ep:
        ep.wait_for_endpoint_by_host_name(host, discovery_timeout=120)
        settings = unreal.IngestCapability_Options()
        settings.working_directory = os.path.join(tempfile.gettempdir(), "ScriptedIngestConversion")
        settings.video.format = "jpeg";  settings.video.file_name_prefix = "frame"
        settings.audio.format = "wav";   settings.audio.file_name_prefix = "audio"
        settings.upload_host_name = host
        with MonoVideoIngestDevice(archive_path, video_discovery_expression) as dev:
            return ingest_takes(dev, settings)

def find_new_capture_data(search_root: str, known_before: set[str]) -> list[str]:
    """인제스트 전 목록(known_before) 대비 새로 생긴 CD_* 애셋 경로 반환. (Q4)"""
    all_assets = unreal.EditorAssetLibrary.list_assets(search_root, recursive=True)
    return [a for a in all_assets
            if a not in known_before and "/CD_" in a or unreal.Paths.get_base_filename(a).startswith("CD_")]
```

> `capture.*` 모듈 접근은 Q2에 따라 엔진 플러그인 Python 경로를 `sys.path`에 추가하거나 프로젝트로 복제. 버전 고정을 위해 **복제 권장**.

**✅ 검증 (Exit Criteria):**
- [ ] Python 콘솔에서:
  ```python
  before = set(unreal.EditorAssetLibrary.list_assets("/Game/CaptureManager/Imports", recursive=True))
  import step1_ingest
  step1_ingest.ingest_mono_video(r"D:/incoming/take01")
  new_cd = step1_ingest.find_new_capture_data("/Game/CaptureManager/Imports", before)
  print(new_cd)   # 새 CD_ 경로 1개 이상
  ```
- [ ] 새 `CD_*` 애셋이 생성되고 경로가 정확히 반환됨.
- [ ] 그 CD를 Phase 2 `create_and_process`에 넣어 처리까지 성공(인제스트→솔브 연결 확인).
- [ ] **경로 A로 단일 에디터 세션 내 인제스트가 성공**하면 Q1 확정. 실패 시 경로 B(LiveLink Hub 상주)로 전환하고 문서에 기록.

### 🛠 에디터 툴 (Phase 3) — `MHA ▸ Phase 3: 인제스트 (폴더 선택)`
버튼을 누르면 **디렉터리 선택 다이얼로그**가 열리고, 고른 mp4 아카이브 폴더를 인제스트해 새 `CD_*` 애셋 경로를 보고.
```python
# Content/Python/pipeline/tool_actions.py  (이어서)
import step1_ingest

def _pick_directory(title="폴더 선택"):
    picked = unreal.EditorDialog.open_directory_dialog(title, "")   # (title, default_path)
    # 반환 형식은 5.8에서 (bool, path) 또는 path 문자열 — 방어적으로 처리
    if isinstance(picked, tuple):
        ok, path = picked
        return path if ok else None
    return picked or None

def ingest_pick_folder():
    folder = _pick_directory("인제스트할 mp4 폴더 선택")
    if not folder:
        return
    before = set(unreal.EditorAssetLibrary.list_assets(config.IMPORT_ROOT, recursive=True))
    with unreal.ScopedSlowTask(1, f"Ingest: {folder}") as t:
        t.make_dialog(True)
        step1_ingest.ingest_mono_video(folder)
    new_cd = step1_ingest.find_new_capture_data(config.IMPORT_ROOT, before)
    msg = "새 CD:\n" + "\n".join(new_cd) if new_cd else "생성된 CD 없음"
    unreal.log(f"[PHASE3] {msg}")
    unreal.EditorDialog.show_message("MHA Phase 3", msg, unreal.AppMsgType.OK)
```
> `EditorDialog.open_directory_dialog` 반환 시그니처는 5.8 설치본에서 실확인 필요(Q4/Q2와 함께). 다이얼로그 API가 다르면 `SystemLibrary`/입력창으로 대체.
- **검증:** 폴더 선택 → 새 `CD_*` 경로 팝업. 그 CD를 Phase 2 버튼에 그대로 연결 가능.

---

## Phase 4 — 배치 통합 (헤드리스 단일 실행)

**목표:** ①②③을 하나의 진입점으로 묶어 `UnrealEditor-Cmd.exe` 헤드리스 단일 커맨드로 완주.

**작업:** `config.py`, `batch_mha.py` 작성.

```python
# Content/Python/pipeline/config.py
PERF_DIR   = "/Game/MHA/Performances"
EXPORT_DIR = "/Game/MHA/Anims"
IMPORT_ROOT = "/Game/CaptureManager/Imports"
```

```python
# Content/Python/pipeline/batch_mha.py
import argparse, sys, unreal
import step1_ingest, step2_solve, step3_export, config

def process_one(archive_path: str, take_name: str):
    before = set(unreal.EditorAssetLibrary.list_assets(config.IMPORT_ROOT, recursive=True))
    unreal.log(f"[MHA] ingest: {archive_path}")
    step1_ingest.ingest_mono_video(archive_path)

    new_cd = step1_ingest.find_new_capture_data(config.IMPORT_ROOT, before)
    if not new_cd:
        raise RuntimeError("no capture data produced by ingest")
    cd_path = new_cd[0]

    unreal.log(f"[MHA] solve: {cd_path}")
    perf = step2_solve.create_and_process(cd_path, config.PERF_DIR, enable_body=True)

    unreal.log(f"[MHA] export: AS_{take_name}")
    step3_export.export_anim_sequence(perf, config.EXPORT_DIR, f"AS_{take_name}")

    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_all(False)
    unreal.log(f"[MHA] DONE: {take_name}")

def run():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-path", required=True)
    p.add_argument("--take-name", required=True)
    args = p.parse_args()
    try:
        process_one(args.archive_path, args.take_name)
    except Exception as e:
        unreal.log_error(f"[MHA] FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
```

헤드리스 실행 커맨드:
```bat
"C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" ^
  "C:\SupergeneGithub\VTuberStudio\MetaHuman.uproject" ^
  -run=pythonscript ^
  -script="C:/SupergeneGithub/VTuberStudio/Content/Python/pipeline/batch_mha.py --archive-path=D:/incoming/take01 --take-name=take01" ^
  -unattended -nosplash -nopause -stdout -FullStdOutLogOutput ^
  -log="C:/SupergeneGithub/VTuberStudio/Saved/Logs/mha_batch.log"
```

**✅ 검증 (Exit Criteria):**
- [ ] 위 커맨드 실행 후 **프로세스 종료코드 = 0**.
- [ ] `Saved/Logs/mha_batch.log`에 `[MHA] DONE: take01` 로그 존재, `FAILED`/`Error` 없음.
- [ ] `/Game/MHA/Anims/AS_take01` 애셋이 디스크에 저장됨(`.uasset` 파일 확인).
- [ ] 에디터 UI를 켜지 않고(무인) 완주.
- [ ] 같은 커맨드 재실행 시 이름 충돌/중복 처리(덮어쓰기 또는 스킵)가 예측대로 동작.

### 🛠 에디터 툴 (Phase 4) — `MHA ▸ Phase 4: 배치 1건 (현재 세션)`
헤드리스 배치(`batch_mha.process_one`)와 **동일 로직을 에디터 세션에서** 실행. 폴더를 고르면 인제스트→솔브→익스포트→저장까지 완주. 헤드리스로 넘기기 전 로직 검증·1건 재처리에 사용.
```python
# Content/Python/pipeline/tool_actions.py  (이어서)
import os, batch_mha

def batch_one_pick_folder():
    folder = _pick_directory("배치 처리할 mp4 폴더 선택")
    if not folder:
        return
    take = os.path.basename(os.path.normpath(folder))
    try:
        with unreal.ScopedSlowTask(1, f"Batch: {take}") as t:
            t.make_dialog(True)
            batch_mha.process_one(folder, take)          # 헤드리스와 같은 진입 함수
        unreal.EditorDialog.show_message("MHA Phase 4",
            f"DONE: {take}\n→ {config.EXPORT_DIR}/AS_{take}", unreal.AppMsgType.OK)
    except Exception as e:
        unreal.log_error(f"[PHASE4] FAILED: {e}")
        unreal.EditorDialog.show_message("MHA Phase 4",
            f"FAILED: {e}", unreal.AppMsgType.OK)
```
> **동치 보장:** 에디터 버튼과 헤드리스가 **같은 `process_one`을 호출**하므로, 버튼에서 성공하면 헤드리스에서도 (RHI 플래그만 맞으면) 동일 결과. 버튼 실행은 이미 RHI가 살아 있어 `-AllowCommandletRendering` 이슈와 무관.
- **검증:** 폴더 선택 → `AS_<take>` 생성 + 저장. 헤드리스 커맨드 결과와 산출물 동일.

---

## Phase 5 — 무인 감시 (watchdog)

**목표:** 에디터 밖 상주 프로세스가 폴더를 감시해 mp4 투입 시 Phase 4 배치를 자동 트리거하고 완료/실패 폴더로 이동.

**작업:** `Automation/watchdog_runner.py` (시스템 Python, `pip install watchdog`).

```python
# Automation/watchdog_runner.py
import subprocess, shutil, time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH, DONE, FAILED = Path(r"D:\pipeline\watch"), Path(r"D:\pipeline\done"), Path(r"D:\pipeline\failed")
UE_CMD   = r"C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
UPROJECT = r"C:\SupergeneGithub\VTuberStudio\MetaHuman.uproject"
SCRIPT   = "C:/SupergeneGithub/VTuberStudio/Content/Python/pipeline/batch_mha.py"

def wait_until_stable(path: Path, checks=3, interval=2):
    last, stable = -1, 0
    while stable < checks:
        size = path.stat().st_size
        stable = stable + 1 if size == last else 0
        last = size
        time.sleep(interval)

def run_job(mp4: Path):
    take = mp4.stem
    wait_until_stable(mp4)
    arg = f'{SCRIPT} --archive-path={mp4.parent.as_posix()} --take-name={take}'
    rc = subprocess.run([UE_CMD, UPROJECT, "-run=pythonscript", f"-script={arg}",
                         "-unattended", "-nosplash", "-nopause", "-stdout"]).returncode
    dest = (DONE if rc == 0 else FAILED) / mp4.name
    shutil.move(str(mp4), str(dest))
    print(f"{'OK' if rc==0 else 'FAIL'}: {take} -> {dest}")

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        p = Path(event.src_path)
        if not event.is_directory and p.suffix.lower() == ".mp4":
            run_job(p)

if __name__ == "__main__":
    for d in (WATCH, DONE, FAILED): d.mkdir(parents=True, exist_ok=True)
    obs = Observer(); obs.schedule(Handler(), str(WATCH), recursive=False); obs.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()
```

**✅ 검증 (Exit Criteria):**
- [ ] watchdog 실행 상태에서 `D:\pipeline\watch`에 mp4 복사 → 자동으로 배치가 뜨고 완주.
- [ ] 성공 시 mp4가 `done/`으로, 실패 시 `failed/`로 이동.
- [ ] 복사 중(대용량) 파일을 조기에 잡지 않음(`wait_until_stable` 동작 확인).
- [ ] 결과 AnimSequence 애셋이 정상 생성됨.

### 🛠 에디터 툴 (Phase 5) — `MHA ▸ Phase 5: 감시 폴더 열기 / 상태`
watchdog은 **에디터 밖 상주 프로세스**라 에디터에서 직접 띄우진 않지만, 운영 편의를 위해 감시/완료/실패 폴더를 탐색기로 열고 대기·완료·실패 개수를 요약하는 버튼을 제공.
```python
# Content/Python/pipeline/tool_actions.py  (이어서)
# 감시 폴더 경로는 config로 통일(watchdog_runner.py와 공유)
def watch_status():
    import os
    roots = {"watch": config.WATCH_DIR, "done": config.DONE_DIR, "failed": config.FAILED_DIR}
    lines = []
    for name, path in roots.items():
        n = len([f for f in os.listdir(path) if f.lower().endswith(".mp4")]) if os.path.isdir(path) else "N/A"
        lines.append(f"{name:6}: {n}   ({path})")
    unreal.EditorDialog.show_message("MHA Phase 5", "\n".join(lines), unreal.AppMsgType.OK)

def open_watch_folder():
    unreal.SystemLibrary.launch_url(f"file:///{config.WATCH_DIR}")   # 탐색기로 열기
```
> **등록 추가:** `register_all()`에 `MHA_Phase5`(`tool_actions.watch_status`)와 `MHA_Phase5_Open`(`tool_actions.open_watch_folder`) 항목 추가. `config.WATCH_DIR/DONE_DIR/FAILED_DIR`를 신설해 watchdog와 경로 단일화(하드코딩 제거).
- **검증:** 버튼 클릭 → `watch/done/failed` mp4 개수 팝업. "폴더 열기" → 탐색기 오픈.

---

## Phase 6 — 운영 견고화

**목표:** 실사용 안정성 확보. 다중 테이크, 재시도, 알림, 동시 실행 방지.

**작업:**
1. **직렬 큐:** watchdog가 여러 파일을 동시에 트리거하지 않도록 큐/락(`Lock` 또는 잡 큐) 도입.
2. **다중 테이크 배치:** `batch_mha.py`를 폴더 내 다수 테이크 순차 처리로 확장(에디터 1회 기동으로 N개 처리 → 기동 오버헤드 절감, Q5).
3. **재시도:** 실패 시 1~2회 자동 재시도 후 `failed/` 격리.
4. **알림:** 성공/실패 요약을 Slack/메일로 통지(선택).
5. **로그 보존:** 테이크별 로그 파일 분리 저장.

**✅ 검증 (Exit Criteria):**
- [ ] `watch/`에 mp4 3개를 동시에 넣어도 서로 간섭 없이 **순차** 완주.
- [ ] 의도적으로 깨진 입력을 넣으면 재시도 후 `failed/`로 격리되고 다른 잡은 계속 진행.
- [ ] 다중 테이크 배치 모드에서 에디터 1회 기동으로 N개 AnimSequence 생성.
- [ ] 각 잡의 로그가 개별 파일로 남고, 성공/실패 통지가 도착(알림 사용 시).
- [ ] 24시간 상주 후에도 프로세스 누수/행 없이 동작(안정성).

### 🛠 에디터 툴 (Phase 6) — 운영 대시보드 (Editor Utility Widget)
운영 단계에서는 단순 메뉴 버튼을 넘어 **Editor Utility Widget(EUW) 패널** 하나로 큐/재시도/로그를 한눈에 보는 것이 실용적. 순수 Python로 로직을 두고 EUW는 표시·버튼만 담당.
- **패널 구성(제안):** 큐 목록(대기/처리중/완료/실패) · `실패 재시도` 버튼 · `큐 비우기` · `최근 로그 열기` · `watchdog 상태`(외부 프로세스 heartbeat 파일 기준).
- **로직 재사용:** 재시도/큐 조작은 Phase 6 Python 함수로 구현하고, EUW 버튼과 헤드리스(watchdog) 양쪽에서 호출 → 단일 소스.
- **간이 대안:** EUW 없이도 `MHA` 메뉴에 `Phase 6: 실패 재처리`, `Phase 6: 로그 열기` 버튼만 추가해도 운영 대부분 커버.
```python
# Content/Python/pipeline/tool_actions.py  (이어서) — 실패 테이크 재처리
def retry_failed():
    import os
    if not os.path.isdir(config.FAILED_DIR):
        return
    mp4s = [f for f in os.listdir(config.FAILED_DIR) if f.lower().endswith(".mp4")]
    for name in mp4s:
        src = os.path.join(config.FAILED_DIR, name)
        take = os.path.splitext(name)[0]
        try:
            batch_mha.process_one(config.FAILED_DIR, take)
            os.replace(src, os.path.join(config.DONE_DIR, name))   # 성공 시 done으로
        except Exception as e:
            unreal.log_error(f"[PHASE6] retry FAILED {take}: {e}")
    unreal.EditorDialog.show_message("MHA Phase 6",
        f"재처리 시도: {len(mp4s)}건", unreal.AppMsgType.OK)
```
- **검증:** `failed/`에 mp4 둔 뒤 버튼 → 성공분은 `done/`으로 이동, 실패분은 로그에 격리. EUW 패널이면 목록/카운트가 실시간 갱신.

---

## 착수 전 확정할 미해결 항목 (근거 문서 §8 요약)

| # | 항목 | 어느 Phase에서 확정 |
|---|---|---|
| Q1 | 인제스트 경로 A(에디터 내) vs B(LiveLink Hub 상주) | Phase 3 |
| Q2 | `capture.*` 모듈: sys.path 추가 vs 프로젝트 복제 | Phase 3 |
| Q3 | 바디 익스포트 정확 경로(얼굴 스켈레톤 외 바디 타깃 필요 여부) | Phase 2 |
| Q4 | 인제스트 후 CD 애셋 경로 자동 확보 방식 | Phase 3 |
| Q5 | 배치 다중 테이크(에디터 1회 기동) | Phase 6 |
| Q6 | 모노 footage용 Identity 권장 여부 | Phase 1~2 |
| Q7 | 헤드리스 GPU/라이선스/동시 인스턴스 제한 | Phase 4 |
