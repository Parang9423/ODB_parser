# ODB++ CAM Image Renderer

ODB++ `.tgz` 데이터를 열어 Step/Layer를 선택하고 CAM Reference Image를 미리보기 및 PNG로 렌더링하는 데스크톱 앱입니다.

## 주요 기능

1. 메뉴바 `파일 > ODB++ 파일 열기...` 또는 `파일 업로드` 버튼으로 ODB++ 파일 선택
2. 기본 파일 형식 `.tgz` (`.tar.gz`도 지원)
3. ODB++ Matrix에서 Step/Layer 목록 자동 추출
4. Layer 선택 시 저해상도 CAM Image 자동 미리보기
5. Job/Step/Layer/Profile 정보 표시
6. 출력 DPI를 지정하고 `렌더링 결과 저장`으로 PNG 생성
7. UI 멈춤을 줄이기 위해 압축 해제/미리보기/최종 렌더링을 worker thread에서 수행

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

## 사용 흐름

`파일 업로드 → .tgz 선택 → Step/Layer 선택 → CAM 미리보기 확인 → DPI 지정 → 렌더링 결과 저장`

기본 미리보기는 150 DPI, 최종 렌더링 기본값은 1200 DPI입니다.

## 현재 Renderer 지원 범위

- ODB++ `.tgz`, `.tar.gz`, extracted job directory (CLI core)
- Step profile
- STEP-REPEAT (translation / 90° rotation / mirror)
- Standard symbol: round, square, rectangle
- Pad (`P`)
- Line (`L`)
- Surface contour (`S`, `OB`, `OS`, `OC`, `OE`)
- Positive / Negative polarity

지원하지 않는 Feature/Symbol은 임의로 근사하지 않고 warning으로 집계합니다.

## CLI

GUI 외에 기존 CLI도 사용할 수 있습니다.

```bash
python odb_cam_renderer.py sample.tgz --step unit --layer l1 --dpi 1200 --output unit_l1.png
```

## 테스트

샘플 ODB++ TGZ가 프로젝트 상위 경로에 있을 때:

```bash
python -m pytest -q
```

현재 샘플 `UNIT/L1`은 Pad/Line/Surface를 warning 없이 렌더링하는 것을 검증합니다.
