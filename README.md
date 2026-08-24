# ODB++ CAM Image Renderer

ODB++ `.tgz` 데이터를 열어 Step/Layer를 선택하고 AOI 비교검사용 CAM Reference Image를 미리보기 및 PNG로 렌더링하는 데스크톱 앱입니다.

## 주요 기능

1. 메뉴바 `파일 > ODB++ 파일 열기...` 또는 `파일 업로드` 버튼으로 ODB++ 파일 선택
2. 기본 파일 형식 `.tgz` (`.tar.gz`도 지원)
3. ODB++ Matrix에서 Step/Layer 목록 자동 추출
4. 단일 Layer 또는 여러 Layer를 합성한 Composite CAM Image 미리보기
5. PNL 좌표계 기준 Hierarchy Overlay 지원
   - PNL / STRIP / UNIT Step을 동시에 선택해 누적 표시
   - 선택한 Step이 소유한 Layer feature만 PNL 좌표계에 합성
   - STEP-REPEAT의 translation / rotation / mirror 적용
   - PNL / STRIP / UNIT Profile 테두리 색상 구분
   - 마우스 위치의 PNL 기준 X/Y 좌표(mm) 표시
6. 미리보기 확대/축소, 마우스 휠 Zoom, Drag Pan, Fit/100% 보기
7. `설정 > 미리보기 설정...`에서 미리보기 DPI 설정
8. 최종 출력 방식 2종 지원
   - 일반 DPI 렌더링
   - AOI 해상도(µm/pixel) 렌더링
9. AOI 모드에서 X/Y 해상도를 각각 지정 가능
10. 설비별 AOI 해상도 프로파일 저장/불러오기
11. Composite Layer별 Order / Operation(ADD, REPLACE, SUBTRACT) / GV(0~255) 설정
12. Composite preset 저장/불러오기
13. Job/Step/Layer/Profile 정보와 예상 출력 픽셀 크기 표시
14. Step/Feature/Profile parse cache 및 대형 Step adaptive preview DPI 적용

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

> Linux에서 Tkinter가 없는 경우 OS 패키지 `python3-tk`가 필요할 수 있습니다.

## Hierarchy Overlay

Preview 상단의 `Hierarchy Overlay`를 켜면 PNL을 root 좌표계로 사용합니다.

- `PNL`: PNL 자체 Profile/feature 표시
- `STRIP`: PNL 내부에 STEP-REPEAT된 STRIP Profile/feature 표시
- `UNIT`: 각 STRIP 내부에 STEP-REPEAT된 UNIT Profile/feature 표시

테두리는 Step level별로 서로 다른 색으로 표시됩니다. 마우스를 CAM 위에 올리면 우측에 PNL 기준 물리 좌표가 mm 단위로 표시됩니다.

예를 들어 `PNL + STRIP`만 선택하면 UNIT feature는 숨기고 PNL/STRIP 데이터만 같은 PNL 좌표계에서 확인할 수 있습니다. `UNIT`까지 선택하면 UNIT의 회로 feature가 동일한 좌표계 위에 추가됩니다.

이 좌표는 이후 AOI 검사영역의 Align origin을 찾기 위한 ODB PNL 좌표 기준으로 사용할 수 있습니다.

## 미리보기 성능 최적화

기존 Renderer는 동일 UNIT이 여러 번 STEP-REPEAT될 때 같은 `features` 파일과 `stephdr/profile`을 반복해서 읽고 파싱했습니다. 현재 버전은 다음 데이터를 메모리 캐시합니다.

- Layer feature text + symbol table
- STEP-REPEAT 정보
- Step profile contour

따라서 같은 ODB Job에서 Step/Layer를 변경하거나 다시 미리보기할 때 파일 I/O와 문자열 파싱이 크게 줄어듭니다.

또한 큰 PNL을 설정된 미리보기 DPI 그대로 rasterize하면 UI 확인에 불필요하게 큰 이미지가 만들어질 수 있어, 미리보기는 약 12MP를 기준으로 DPI를 자동 하향합니다. 사용자가 지정한 미리보기 DPI는 상한값으로 동작합니다.

이 최적화는 **미리보기에만 적용**되며 최종 PNG 저장의 DPI/AOI `µm/pixel` 값에는 영향을 주지 않습니다.

## Composite CAM 모드

`렌더링 대상`을 `Composite`로 선택하면 여러 ODB++ Layer를 순서대로 합성할 수 있습니다. 각 Layer는 다음 속성을 가집니다.

- `Order`: 합성 순서. 뒤의 Layer가 앞의 Layer 위에 적용됩니다.
- `ADD`: 현재 픽셀과 지정 GV 중 큰 값을 사용합니다.
- `REPLACE`: 해당 Layer 형상 영역을 지정 GV로 덮어씁니다.
- `SUBTRACT`: 해당 Layer 형상 영역을 GV 0으로 제거합니다.
- `GV`: 0~255 grayscale 값입니다.

예시:

```text
1. L1       / REPLACE / GV 255
2. UV_1-2   / REPLACE / GV 96
3. SM-L1    / ADD     / GV 160
```

Composite preset은 아래 파일에 저장됩니다.

```text
~/.odb_cam_renderer/composite_presets.json
```

Windows에서는 일반적으로 `%USERPROFILE%\.odb_cam_renderer\composite_presets.json` 입니다.

## AOI 해상도 모드

`출력 방식`을 `AOI 해상도`로 선택하면 X/Y 축을 `µm/pixel` 단위로 설정할 수 있습니다.

예시:

- X = 10 µm/pixel
- Y = 5 µm/pixel
- UNIT 크기 = 17 mm × 17 mm

DPI와의 관계는 다음과 같습니다.

```text
DPI = 25400 / (µm/pixel)
```

AOI 설비 프로파일은 사용자 홈 디렉터리의 아래 파일에 저장됩니다.

```text
~/.odb_cam_renderer/aoi_profiles.json
```

## 현재 Renderer 지원 범위

- ODB++ `.tgz`, `.tar.gz`, extracted job directory
- Step profile
- STEP-REPEAT (translation / rotation / mirror)
- Standard symbol: round, square, rectangle
- Pad (`P`)
- Line (`L`)
- Surface contour (`S`, `OB`, `OS`, `OC`, `OE`)
- Positive / Negative polarity
- X/Y 독립 raster scale
- 8-bit grayscale Composite render
- PNL-root hierarchy render / Step visibility filter
- shared parse cache

지원하지 않는 Feature/Symbol은 임의로 근사하지 않고 warning으로 집계합니다.

## 테스트

```bash
python -m pytest -q
```

`.github/workflows/tests.yml`에서도 push/PR 시 Python compile과 pytest를 실행하도록 구성되어 있습니다.
