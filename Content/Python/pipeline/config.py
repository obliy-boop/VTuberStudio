# Copyright Supergene. MotionCapture 자동화 파이프라인 공통 설정.
# 콘텐츠 경로 상수 (Phase 2~4에서 사용). 현재 바디 모캡 중심.

# 인제스트된 FootageCaptureData(CD_*)가 생성되는 루트
IMPORT_ROOT = "/Game/CaptureManager/Imports"

# --- Phase 3 인제스트 설정 ---
# 인제스트 중간 산출물(jpeg 프레임/wav)이 쌓이는 임시 작업 폴더.
# 엔진 공식 예제는 tempdir 아래를 쓴다. 테이크당 수 GB가 될 수 있으니 여유 있는 드라이브로.
import os as _os
import tempfile as _tempfile

INGEST_WORK_DIR = _os.path.join(_tempfile.gettempdir(), "MHA_IngestConversion")

# 인제스트 폴더 선택 다이얼로그의 기본 경로 (빈 문자열이면 미지정)
INGEST_DEFAULT_DIR = r"D:\incoming"

# 솔브 산출물(MetaHumanPerformance) 저장 위치
PERF_DIR = "/Game/MHA/Performances"

# 익스포트 산출물(AnimSequence) 저장 위치
EXPORT_DIR = "/Game/MHA/Anims"

# --- 익스포트 타깃 스켈레톤 ---
# 바디 애니메이션을 구울 타깃 스켈레톤.
# Phase 1 기준선(aespa 댄스, AS_Mocap_aspa) 검증 결과 확정: SKEL_UE5_F (IdaFaber Girl 스켈레톤).
#   본 트랙 167개(root/pelvis/spine_01...), 875프레임/29.17s @30fps.
BODY_SKELETON = "/Game/IdaFaber/Meshes/Girl/SKEL_UE5_F.SKEL_UE5_F"

# 얼굴 타깃 스켈레톤 (face 사용 시. 현재 프로젝트에 없음 → 보류. MetaHuman 임포트 후 생성됨)
FACE_SKELETON = "/Game/MetaHumans/Common/Face/Face_Archetype_Skeleton.Face_Archetype_Skeleton"
