# ODB++ CAM Image Renderer

ODB++ `.tgz` 데이터를 열어 Step/Layer를 선택하고 AOI 비교검사용 CAM Reference Image를 미리보기 및 PNG로 렌더링하는 데스크톱 앱입니다.

## 주요 기능

1. 메뉴바 `파일 > ODB++ 파일 열기...` 또는 `파일 업로드` 버튼으로 ODB++ 파일 선택
2. 기본 파일 형식 `.tgz` (`.tar.gz`도 지원)
3. ODB++ Matrix에서 Step/Layer 목록 자동 추출
4. 단일 Layer 또는 여러 Layer를 합성한 Composite CAM Image 미리보기
5. 미리보기 확대/축소, 마우스 휠 Zoom, Drag Pan, Fit/100% 보기
6. `설정 > 미리보기 설정...`에서 미리보기 DPI 설정
7. 최종 출력 방식 2종 지원
   - 일반 DPI 렌더링
   - AOI 해상도(µm/pixel) 렌더링
8. AOI 모드에서 X/Y 해상도를 각각 지정 가능
9. 설비별 AOI 해상도 프로파일 저장/불러오기
10. Composite Layer별 Order / Operation(ADD, REPLACE, SUBTRACT) / GV(0~255) 설정
11. Composite preset 저장/불러오기
12. Job/Step/Layer/Profile 정보와 예상 출력 픽셀 크기 표시

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

이 경우 회로는 밝은 흰색, Via/Hole 계열은 회색 계열로 표현할 수 있습니다. Composite preset은 아래 파일에 저장됩니다.

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

결과 이미지는 약 1700 × 3400 px 스케일로 생성됩니다. 실제 구현은 ODB++ profile boundary의 부동소수점 값과 raster ceiling 때문에 1 px 정도 커질 수 있습니다.

DPI와의 관계는 다음과 같습니다.

```text
DPI = 25400 / (µm/pixel)
```

AOI 설비 프로파일은 사용자 홈 디렉터리의 아래 파일에 저장됩니다.

```text
~/.odb_cam_renderer/aoi_profiles.json
```

Windows에서는 일반적으로 `%USERPROFILE%\.odb_cam_renderer\aoi_profiles.json` 입니다.

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
- 8-bit grayscale Composite render (Order / ADD / REPLACE / SUBTRACT / GV)

지원하지 않는 Feature/Symbol은 임의로 근사하지 않고 warning으로 집계합니다.

## CLI

일반 DPI:

```bash
python odb_cam_renderer.py sample.tgz --step unit --layer l1 --dpi 1200 --output unit_l1.png
```

AOI 해상도:

```bash
python odb_cam_renderer.py sample.tgz --step unit --layer l1 \
  --um-per-pixel-x 10 --um-per-pixel-y 5 --output unit_l1_aoi.png
```

Y 값을 생략하면 X와 동일한 해상도를 사용합니다.

## 테스트

```bash
python -m pytest -q
```

Renderer 테스트에는 standard symbol, profile, STEP transform, X/Y 독립 raster size, µm/pixel→DPI 변환, Composite GV 연산 검증이 포함됩니다.
