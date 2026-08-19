# ODB++ CAM Image Renderer

ODB++ `.tgz` 데이터를 열어 Step/Layer를 선택하고 AOI 비교검사용 CAM Reference Image를 미리보기 및 PNG로 렌더링하는 데스크톱 앱입니다.

## 주요 기능

1. ODB++ `.tgz` / `.tar.gz` 파일 로드
2. Step/Layer 자동 추출 및 CAM 미리보기
3. 미리보기 DPI 설정, 확대/축소, 마우스 휠 Zoom, Drag Pan, Fit/100% 보기
4. 최종 출력 방식 2종 지원
   - 일반 DPI 렌더링
   - AOI 해상도(`µm/pixel`) 렌더링
5. AOI 모드에서 X/Y 해상도를 각각 지정 가능
6. 설비별 AOI 해상도 프로파일 저장/불러오기
7. ODB Profile 크기와 예상 출력 pixel 크기 표시

## 실행

```bash
pip install -r requirements.txt
python app.py
```

## AOI 해상도 모드

`출력 방식`을 `AOI 해상도`로 선택하고 설비의 물리 해상도를 입력합니다.

예시:

```text
X = 10 µm/pixel
Y = 5 µm/pixel
UNIT = 17 mm × 17 mm
```

이 경우 결과 이미지는 약 `1700 × 3400 px` 스케일로 생성됩니다. ODB++ profile boundary의 부동소수점 값과 raster ceiling 때문에 실제 결과는 1 px 정도 커질 수 있습니다.

DPI 환산식:

```text
DPI = 25400 / (µm/pixel)
```

설비 프로파일은 사용자 홈 디렉터리에 저장됩니다.

```text
~/.odb_cam_renderer/aoi_profiles.json
```

Windows에서는 보통 `%USERPROFILE%\.odb_cam_renderer\aoi_profiles.json` 입니다.

## Renderer 지원 범위

- Step profile
- STEP-REPEAT
- Standard symbol: round, square, rectangle
- Pad (`P`), Line (`L`), Surface (`S/OB/OS/OC/OE`)
- Positive / Negative polarity
- X/Y 독립 raster scale

지원하지 않는 Feature/Symbol은 warning으로 집계합니다.

## CLI

일반 DPI:

```bash
python odb_cam_renderer.py sample.tgz --step unit --layer l1 --dpi 1200 --output unit_l1.png
```

AOI 해상도:

```bash
python odb_cam_renderer.py sample.tgz --step unit --layer l1 --um-per-pixel-x 10 --um-per-pixel-y 5 --output unit_l1_aoi.png
```

`--um-per-pixel-y`를 생략하면 X와 동일한 해상도를 사용합니다.

## 테스트

```bash
python -m pytest -q
```

AOI 기능 테스트에는 X/Y 독립 raster size와 `µm/pixel → DPI` 변환 검증이 포함됩니다.
