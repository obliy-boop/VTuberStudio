# Git LFS 운영 노트

> 대상: `VTuberStudio` (`MetaHuman.uproject`, Engine 5.8) — `origin` = `github.com/obliy-boop/VTuberStudio`

## 현재 설정

LFS는 이미 구성되어 있고 추가 설정이 필요하지 않다.

| 항목 | 상태 |
|---|---|
| git-lfs | 3.7.1 |
| 로컬 필터 | `filter.lfs.process = git-lfs filter-process` |
| `.gitattributes` | `*.uasset`, `*.umap`, 이미지/오디오/비디오/메시 확장자 등록됨 |
| 추적 중 LFS 객체 | 약 1,278개 |

새 클론에서 해야 할 일은 `git lfs install`(사용자 계정에 1회) 뿐이며, 그 후 `git clone`이 포인터를 자동으로 실체 파일로 받는다.

## ⚠️ 제외된 애셋 — `ViTPose.uasset`

```
Plugins/Marketplace/MetaHumanBodyTracker/Content/Models/Offline/ViTPose.uasset
```

- **크기:** 2,549,205,520 바이트 (2.37 GiB)
- **상태:** `.gitignore`로 제외. 리포지토리에 **없다.**
- **이유:** GitHub LFS의 **파일당 2 GB 상한**을 초과한다. 이 제한은 Free/Pro/Team 플랜 공통이며(Enterprise Cloud만 5 GB), LFS 사용 여부와 무관하게 푸시가 거부된다.

### 압축은 해결책이 아니다 (2026-07-31 실측)

`ViTPose.zip`으로 압축해봤으나 **2,372,552,948 바이트 = 2.21 GiB**로 여전히 상한을 넘었다.
신경망 가중치는 이미 고엔트로피 데이터라 **압축률이 약 7%에 불과**하다 (2.37 GiB → 2.21 GiB).
`ViTPose.zip`도 `.gitignore`에 추가해 두었다.

남은 우회 수단은 **분할**(1.1 GiB × 2 파트 + 재조립 스크립트) 또는 **외부 스토리지**뿐이다.
현재는 둘 다 채택하지 않고, 플러그인 재설치로 획득하는 방식을 쓴다.

### 획득 방법

Fab에서 **MetaHuman Animator Markerless Motion Capture** 플러그인을 설치하면 함께 받아지는 재배포 애셋이다. 직접 관리할 필요가 없다.

1. Epic Games Launcher → Fab → 위 플러그인을 엔진 **5.8**용으로 설치, 또는 이미 받아둔 사본을 복사
2. `Plugins/Marketplace/MetaHumanBodyTracker/` 아래에 배치
3. 위 경로에 `ViTPose.uasset`이 있는지 확인 (2.37 GiB)

플러그인 정보: `MetaHumanBodyTracker.uplugin` — `VersionName 1.0.0`, `EngineVersion 5.8.0`

### 없으면 무슨 일이 생기는가

이 파일은 바디 트래킹 오프라인 솔브에 쓰이는 모델이다. 없으면 Phase 2 이후의 **바디 솔브가 실패**한다
(→ [`MotionCapture_Automation_Plan.md`](./MotionCapture_Automation_Plan.md)). 클론 직후 파이프라인을 돌리기 전에
위 경로의 존재를 먼저 확인할 것.

### 같은 폴더의 다른 모델 파일

`Models/Offline/`의 나머지 모델들은 **2 GB 미만이라 정상적으로 LFS로 추적·푸시되어 있다.** 별도 조치가 필요 없다.

| 파일 | 크기 |
|---|---|
| `chmr_backbone.uasset` | 625 MB |
| `hue_step_simplified.uasset` | 578 MB |
| `hue_finalStep_simplified.uasset` | 448 MB |
| `camera_calib.uasset` | 117 MB |
| `chmr_head.uasset` | 68 MB |
| `ViTPosePost.uasset` | 32 KB |

즉 **`ViTPose.uasset` 하나만 예외**다. 일관성 측면에서 `Models/Offline` 전체를 제외하는 방안도 검토했으나,
이미 푸시된 파일을 히스토리에서 걷어내려면 `git filter-repo` + 강제 푸시 + 협업자 재클론이 필요해
위험 대비 이득이 작다고 판단해 현 상태를 유지한다.

## 새 대용량 애셋을 추가할 때

`.gitattributes`가 확장자 단위로 잡아주므로 별도 `git lfs track`은 대개 불필요하다. 다만 커밋 전에 크기를 확인할 것:

```bash
# 2 GB 초과 파일 탐지 (커밋 전)
find . -type f -size +2000M -not -path "./.git/*" -exec ls -la {} \;

# LFS 포인터로 올라가는지 확인
git check-attr filter -- <경로>     # → "filter: lfs" 여야 함

# 실제로 LFS에 들어갔는지 확인 (add 후)
git lfs status
```

2 GB를 넘으면 위 `ViTPose.uasset`과 동일하게 `.gitignore` + 이 문서에 획득 방법을 기록하는 방식으로 처리한다.
