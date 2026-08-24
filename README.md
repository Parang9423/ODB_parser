# ODB++ CAM Image Renderer

ODB++ `.tgz` 데이터를 열어 PNL 좌표계 기준으로 필요한 Layer와 Step만 선택해 AOI 비교검사용 Composite CAM을 미리보기/PNG로 생성하는 데스크톱 앱입니다.

## 현재 UI 흐름

1. `파일 업로드` 또는 `파일 > ODB++ 파일 열기...`로 `.tgz` / `.tar.gz` 선택
2. 압축 해제 후 **Matrix/Step 메타데이터만 우선 로드**
3. `렌더링 레이어 선택` 팝업에서 필요한 Layer만 체크
   - Layer Name
   - Type (`SIGNAL`, `MIXED`, `DRILL`, `SOLDER_MASK` 등)
   - Context
   - Side
   - Polarity
4. 체크한 Layer의 `features`만 실제 렌더링 시 읽어서 메모리 캐시
5. 선택된 Layer는 자동 Composite되어 PNL 기준 미리보기로 표시
6. `PNL / STRIP / UNIT` 체크박스로 표시할 Step level 선택
7. 필요하면 `레이어 선택/변경` 버튼으로 다른 Layer를 추가/제거
8. Layer별 `Operation / GV`를 수정한 뒤 Composite 결과 저장

기존 Step 드롭다운과 단일 Layer 보기 모드는 UI에서 제거했습니다. 화면 좌표계는 가능한 경우 항상 `PNL`을 root로 사용합니다.

## Lazy Layer Loading

초기 ODB 로딩 시 모든 Layer의 feature를 읽지 않습니다. 먼저 `matrix/matrix`와 Step 목록만 확인하고, 레이어 선택 팝업에서 체크된 Layer만 렌더링 시점에 읽습니다.

```text
ODB++ TGZ
  ↓
압축 해제
  ↓
Matrix + Step metadata
  ↓
Layer 선택 팝업
  ↓
선택 Layer features만 read/cache
  ↓
Composite render
```

`FastODBRenderer`는 다음 데이터를 캐시합니다.

- Layer feature text + symbol table
- STEP-REPEAT 정보
- Step profile contour

따라서 같은 Job에서 Step 체크 상태나 GV를 바꾸고 다시 렌더링할 때 반복 파일 I/O와 문자열 parsing을 줄입니다.

## Step 표시

좌측 `Step 표시`에서 다음 항목을 독립적으로 켜고 끌 수 있습니다.

- `PNL`
- `STRIP`
- `UNIT`

PNL이 존재하면 PNL 좌표계를 root로 유지한 상태에서 선택된 Step의 feature/profile만 표시합니다. STEP-REPEAT의 translation / rotation / mirror도 적용됩니다.

Profile 테두리 색상:

```text
PNL    : Blue
STRIP  : Green
UNIT   : Orange
```

CAM 위에 마우스를 올리면 PNL 기준 물리 좌표가 mm 단위로 표시됩니다.

```text
PNL X: 65.788 mm   Y: 33.127 mm
```

이 값은 이후 AOI 검사영역 Align origin을 찾기 위한 기준 좌표로 사용할 수 있습니다.

## Layer Composite / GV

선택된 Layer는 Composite 목록에 표시되며 각 Layer의 `Operation`과 `GV(0~255)`를 조정할 수 있습니다.

기본값은 Layer type을 기준으로 시작값만 자동 지정하며 사용자가 수정할 수 있습니다.

```text
SIGNAL / 일반 Layer  → REPLACE / GV 255
DRILL 계열           → REPLACE / GV 96
MIXED                → ADD     / GV 220
SOLDER_MASK          → ADD     / GV 160
```

Operation:

- `ADD`: 기존 값과 지정 GV 중 큰 값 사용
- `REPLACE`: 해당 형상 영역을 지정 GV로 덮어쓰기
- `SUBTRACT`: 해당 영역을 GV 0으로 제거

레이어를 다시 선택할 때 기존에 설정한 Operation/GV는 유지됩니다.

## 미리보기 성능

대형 PNL을 높은 DPI 그대로 rasterize하지 않도록 미리보기는 약 `12 MP` 기준으로 DPI를 자동 하향합니다. 설정한 Preview DPI는 상한값으로 동작하며, 이 자동 조정은 **미리보기에만 적용**됩니다.

최종 PNG 저장 시에는 지정한 DPI 또는 AOI `µm/pixel` 해상도를 그대로 사용합니다.

## AOI 해상도 모드

최종 출력은 일반 DPI와 AOI 물리 해상도 두 방식을 지원합니다.

```text
DPI = 25400 / (µm/pixel)
```

AOI 모드에서는 X/Y 해상도를 독립적으로 지정할 수 있습니다.

## 실행

Python 3.10+ 권장.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Windows 실행파일

```powershell
pyinstaller --noconfirm --clean --onefile --windowed --name ODB_CAM_Renderer app.py
```

## 현재 Renderer 지원 범위

- ODB++ `.tgz`, `.tar.gz`, extracted job directory
- Step profile / STEP-REPEAT
- translation / rotation / mirror
- Standard symbol: round, square, rectangle
- Pad (`P`), Line (`L`), Surface (`S/OB/OS/OC/OE`)
- Positive / Negative polarity
- X/Y 독립 raster scale
- 8-bit grayscale Composite
- PNL-root hierarchy render
- Step visibility filter
- shared parse cache

지원하지 않는 Feature/Symbol은 임의로 근사하지 않고 warning으로 집계합니다.

## 테스트

```bash
python -m py_compile app.py app_core.py odb_cam_renderer.py hierarchy_renderer.py
python -m pytest -q
```

`.github/workflows/tests.yml`에서도 `main` push 및 PR 시 compile + pytest를 실행합니다.
