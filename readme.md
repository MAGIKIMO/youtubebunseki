# YouTube Comment Sentiment Lab

YouTube 댓글을 수집해 AI 감정분석 모델로 여론 흐름, 위험 신호, 핵심 키워드를 시각화하는 Streamlit 대시보드입니다.  
이력서와 포트폴리오에서 설명하기 쉽도록 여러 영상 비교, 시간대별 추이, 부정 댓글 모니터링 기능을 한 화면에서 확인할 수 있게 구성했습니다.

## 주요 기능

- **AI 감정분석**: Transformers 기반 모델로 댓글을 긍정, 부정, 질문, 유머/비꼼, 중립 유형으로 분류
- **여러 영상 비교**: 복수의 YouTube URL을 입력해 댓글 반응과 위험도를 비교
- **위험도 모니터링**: 부정/비꼼 댓글 비율을 기반으로 위험 점수와 경보 표시
- **시간대별 추이**: 댓글 감정 변화와 간단한 미래 추세를 라인 차트로 시각화
- **키워드 분석**: 전체 댓글과 부정 댓글의 주요 단어, 워드클라우드 제공
- **포트폴리오형 UI**: Streamlit 기반 대시보드에 히어로 영역, 카드형 지표, 통일된 차트 스타일 적용

## 기술 스택

- Python
- Streamlit
- Pandas, NumPy
- Plotly
- Transformers, PyTorch
- YouTube Data API v3
- WordCloud, Matplotlib

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

또는 Windows에서 `streamlit` 명령이 바로 인식되지 않으면:

```bash
py -m streamlit run app.py
```

## 사용 방법

1. Google Cloud Console에서 YouTube Data API v3 키를 발급합니다.
2. 사이드바에 API 키를 입력합니다.
3. 분석할 YouTube 영상 URL을 한 줄에 하나씩 입력합니다.
4. 분석할 댓글 수를 선택하고 분석 시작 버튼을 누릅니다.
5. 결과 요약, 영상별 상세 분석, 키워드 분석, 댓글 테이블을 확인합니다.

## 포트폴리오 포인트

- 외부 API 연동부터 데이터 전처리, NLP 모델 추론, 시각화까지 하나의 서비스 흐름으로 구현
- 모델 로드 실패 시 백업 모델과 키워드 기반 분석으로 대체하는 안정성 고려
- 여러 영상 비교와 위험도 점수로 단순 감정분석을 실무형 모니터링 도구로 확장
- Streamlit 기본 UI를 커스텀 CSS와 공통 Plotly 테마로 개선

## 필요 조건

- Python 3.8 이상
- YouTube Data API v3 키
- 인터넷 연결

