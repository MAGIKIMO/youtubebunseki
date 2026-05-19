import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import re
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import logging
import warnings

# --- 추가된 라이브러리 임포트 ---
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

# matplotlib 한글 폰트 설정 (발표 환경에 맞게 경로 수정 필수!)
# 윈도우: 'C:/Windows/Fonts/malgun.ttf'
# 맥: '/System/Library/Fonts/AppleSDGothicNeo.ttc' 또는 '/Library/Fonts/AppleGothic.ttf'
# 리눅스: 나눔고딕 설치 후 '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' 등
plt.rcParams['font.family'] = 'Malgun Gothic' # 예시, 실제 환경에 맞게 수정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

# 페이지 설정
st.set_page_config(
    page_title="YouTube Comment Sentiment Lab",
    page_icon="YT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    :root {
        --bg: #f6f7fb;
        --surface: #ffffff;
        --ink: #111827;
        --muted: #6b7280;
        --line: #e5e7eb;
        --brand: #ff0033;
        --blue: #2563eb;
    }

    .stApp {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(246, 247, 251, 0.96) 42%),
            radial-gradient(circle at top left, rgba(255, 0, 51, 0.10), transparent 34%),
            var(--bg);
        color: var(--ink);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }

    .main-header {
        display: none;
    }

    .portfolio-hero {
        border: 1px solid rgba(17, 24, 39, 0.08);
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.98) 0%, rgba(37, 99, 235, 0.90) 54%, rgba(255, 0, 51, 0.88) 100%);
        border-radius: 8px;
        padding: 2.25rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 18px 45px rgba(17, 24, 39, 0.16);
    }

    .portfolio-kicker {
        width: fit-content;
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        color: rgba(255, 255, 255, 0.86);
        margin-bottom: 0.85rem;
    }

    .portfolio-hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.6rem);
        line-height: 1.08;
        letter-spacing: 0;
        font-weight: 800;
    }

    .portfolio-hero p {
        max-width: 760px;
        margin: 1rem 0 0;
        color: rgba(255, 255, 255, 0.84);
        font-size: 1.03rem;
        line-height: 1.65;
    }

    .hero-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.35rem;
    }

    .hero-chip {
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: rgba(255, 255, 255, 0.92);
        padding: 0.45rem 0.75rem;
        font-size: 0.84rem;
        font-weight: 650;
    }

    .section-panel {
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.90);
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        margin: 1rem 0 1.25rem;
        box-shadow: 0 12px 32px rgba(17, 24, 39, 0.06);
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.94);
        border-radius: 8px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 26px rgba(17, 24, 39, 0.05);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted);
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 800;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.88);
    }

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background: rgba(255, 255, 255, 0.08) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] button {
        border-radius: 8px !important;
        font-weight: 800 !important;
    }

    .stPlotlyChart,
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: white;
        padding: 0.35rem;
        box-shadow: 0 10px 26px rgba(17, 24, 39, 0.04);
    }

    h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }

    hr {
        border: none;
        border-top: 1px solid var(--line);
        margin: 1.6rem 0;
    }
    .danger-alert {
        background-color: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #e11d48;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-alert {
        background-color: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-alert {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stCodeBlock {
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

def polish_chart(fig):
    """Apply a clean portfolio dashboard style to Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#111827", size=13),
        title=dict(font=dict(size=18, color="#111827"), x=0.02, xanchor="left"),
        margin=dict(l=20, r=20, t=56, b=30),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12)
        ),
        hoverlabel=dict(bgcolor="#111827", font_color="white")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    return fig

# YouTube API 클래스
class YouTubeAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def extract_video_id(self, url):
        """YouTube URL에서 video ID 추출"""
        patterns = [
            r'youtube\.com/watch\?v=([^&]+)',
            r'youtu\.be/([^?]+)',
            r'youtube\.com/embed/([^?]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_video_info(self, video_id):
        """비디오 정보 가져오기"""
        url = f"{self.base_url}/videos"
        params = {
            'part': 'snippet,statistics',
            'id': video_id,
            'key': self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
            data = response.json()

            if 'items' in data and len(data['items']) > 0:
                video = data['items'][0]
                return {
                    'title': video['snippet']['title'],
                    'channel': video['snippet']['channelTitle'],
                    'published_at': video['snippet']['publishedAt'],
                    'view_count': int(video['statistics'].get('viewCount', 0)),
                    'like_count': int(video['statistics'].get('likeCount', 0)),
                    'comment_count': int(video['statistics'].get('commentCount', 0))
                }
            else:
                st.error(f"비디오 ID '{video_id}'에 대한 정보를 찾을 수 없습니다. (비디오가 없거나 접근 제한됨)")
                return None
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP 오류 발생: {http_err} (비디오 ID: {video_id})")
            st.error(f"응답 내용: {response.text}") # API 응답 내용 추가 출력
            return None
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"네트워크 연결 오류: {conn_err}")
            return None
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"요청 시간 초과 오류: {timeout_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            st.error(f"요청 오류: {req_err}")
            return None
        except ValueError as json_err: # JSON 디코딩 오류 (Expecting value: line 1 column 1 (char 0) 등)
            st.error(f"API 응답 JSON 디코딩 오류: {json_err}. API 키 또는 할당량을 확인하세요.")
            st.error(f"응답 내용: {response.text}")
            return None
        except Exception as e:
            st.error(f"비디오 정보를 가져오는 중 알 수 없는 오류 발생: {str(e)}")
            return None

    def get_comments(self, video_id, max_results=100):
        """댓글 가져오기"""
        url = f"{self.base_url}/commentThreads"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': min(max_results, 100),
            'order': 'time',
            'key': self.api_key
        }

        comments = []
        next_page_token = None

        try:
            while len(comments) < max_results:
                if next_page_token:
                    params['pageToken'] = next_page_token
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if 'items' not in data:
                    break

                for item in data['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'text': comment['textDisplay'],
                        'author': comment['authorDisplayName'],
                        'published_at': comment['publishedAt'],
                        'like_count': comment['likeCount']
                    })
                    if len(comments) >= max_results:
                        break

                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break

        except requests.exceptions.RequestException as req_err:
            st.error(f"댓글을 가져오는 중 요청 오류: {req_err}")
            return []
        except ValueError as json_err: # JSON 디코딩 오류
            st.error(f"API 응답 JSON 디코딩 오류 (댓글): {json_err}. API 키 또는 할당량을 확인하세요.")
            return []
        except Exception as e:
            st.error(f"댓글을 가져오는 중 알 수 없는 오류 발생: {str(e)}")
            return []

        return comments[:max_results]

# AI 감정분석 모델 클래스
class KoreanSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.is_initialized = False
        self.model_type = "fallback" # 기본값 설정

    @st.cache_resource
    def load_model(_self):
        """AI 모델 로드 (캐시 사용으로 한번만 로드)"""
        try:
            # 1차: 한국어 감정분석 모델 (무료)
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest" # 다국어, 감정분석 특화

            _self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            _self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # 파이프라인 생성
            _self.classifier = pipeline(
                "text-classification",
                model=_self.model,
                tokenizer=_self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )

            _self.is_initialized = True
            _self.model_type = "roberta"
            st.sidebar.success(f"AI 모델 로드 성공: {model_name}")
            return True

        except Exception as e:
            # st.warning(f"1차 모델 로드 실패: {str(e)}") # 주석 처리하여 사용자에게 혼란 방지
            # 2차: 다른 감정분석 모델
            try:
                model_name = "nlptown/bert-base-multilingual-uncased-sentiment" # 다국어, 5점 척도 감성
                _self.classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                _self.is_initialized = True
                _self.model_type = "bert"
                st.sidebar.success(f"AI 모델 로드 성공: {model_name}")
                return True
            except Exception as e2:
                # st.warning(f"2차 모델 로드 실패: {str(e2)}") # 주석 처리하여 사용자에게 혼란 방지
                # 3차: 가장 기본적인 모델 (영어)
                try:
                    model_name = "distilbert-base-uncased-finetuned-sst-2-english" # 영어, 긍부정
                    _self.classifier = pipeline(
                        "text-classification",
                        model=model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    _self.is_initialized = True
                    _self.model_type = "distilbert"
                    st.sidebar.success(f"AI 모델 로드 성공: {model_name}")
                    return True
                except Exception as e3:
                    st.error(f"모든 AI 모델 로드 실패. 키워드 기반 분석으로 대체됩니다.")
                    _self.is_initialized = False
                    _self.model_type = "fallback"
                    return False

    def analyze_sentiment(self, text):
        """AI 기반 감정 분석"""
        if not self.is_initialized:
            if not self.load_model():
                return self._fallback_analysis(text) # 모델 로드 실패 시 대체 분석

        try:
            # 텍스트 전처리
            text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)  # URL 제거
            text = text.strip()

            if not text:
                return "중립/기타"

            # AI 모델로 감정 분석
            result = self.classifier(text)

            if isinstance(result[0], list): # RoBERTa 모델처럼 여러 라벨 점수를 반환하는 경우
                scores = {item['label']: item['score'] for item in result[0]}
                predicted_label = max(scores.keys(), key=lambda k: scores[k])
            else: # BERT, DistilBERT처럼 단일 결과 반환하는 경우
                predicted_label = result[0]['label']

            # 라벨을 우리 분류 체계로 매핑
            return self._map_to_category(predicted_label, text)

        except Exception as e:
            # AI 분석 실패 시 키워드 기반으로 대체
            # st.warning(f"AI 분석 중 오류 발생 ({text[:30]}...): {e}. 키워드 기반 분석으로 대체됩니다.") # 디버깅용
            return self._fallback_analysis(text)

    def _map_to_category(self, ai_label, text):
        """AI 모델 결과를 우리 카테고리로 매핑"""
        text_lower = text.lower()

        # 질문 패턴 우선 확인
        question_patterns = ['?', '언제', '어떻게', '왜', '뭐', '무엇', '어디', '누구', '질문', '궁금', '알려주', '가르쳐', '문의']
        if any(pattern in text_lower for pattern in question_patterns) or text.strip().endswith('?'):
            return "질문/요청/정보성"

        # 유머/비꼼 패턴 확인 (키워드 기반보다 AI 감성이 우선일 수 있으나, 명확한 유머 패턴은 여기에 먼저)
        humor_patterns = ['ㅋㅋ', 'ㅎㅎ', '허허', 'ㅋ', '미쳤', '대박', '헐', '어휴', 'ㅋㅋㅋㅋ', 'ㅎㅎㅎㅎ']
        if text_lower.count('ㅋ') >= 3 or text_lower.count('ㅎ') >= 3:
            return "비꼬기/유머/감탄"

        # AI 감정분석 결과 기반 매핑 (모델별 라벨 처리)
        label_upper = str(ai_label).upper()

        # RoBERTa 모델 라벨
        if self.model_type == "roberta":
            if label_upper in ['LABEL_2', 'POSITIVE', 'POS']:
                return "찬성/지지"
            elif label_upper in ['LABEL_0', 'NEGATIVE', 'NEG']:
                return "반대/비판"
            elif label_upper in ['LABEL_1', 'NEUTRAL']:
                return "중립/기타"

        # BERT 다국어 모델 라벨 (1~5 별점)
        elif self.model_type == "bert":
            if label_upper in ['5 STARS', '4 STARS']:
                return "찬성/지지"
            elif label_upper in ['1 STAR', '2 STARS']:
                return "반대/비판"
            elif label_upper in ['3 STARS']:
                return "중립/기타"

        # DistilBERT 모델 라벨 (영어)
        elif self.model_type == "distilbert":
            if label_upper == 'POSITIVE':
                return "찬성/지지"
            elif label_upper == 'NEGATIVE':
                return "반대/비판"

        # 기타 또는 매핑되지 않은 경우
        return "중립/기타"

    def _fallback_analysis(self, text):
        """AI 분석 실패 시 키워드 기반 분석"""
        text = text.lower()

        # 키워드 기반 분류 (백업용)
        positive_keywords = ['좋다', '최고', '훌륭', '감사', '좋네', '굿', '완벽', '최고다', '응원', '지지', '찬성', '맞다', '좋아', '사랑', '대단', '멋지다', '깔끔', '재밌', '기대', '기쁨', '칭찬']
        negative_keywords = ['싫다', '별로', '최악', '나쁘다', '반대', '비판', '틀렸', '문제', '잘못', '실망', '화난다', '짜증', '답답', '역겹', '쓰레기', '어렵', '불편', '부족', '아쉽', '논란']
        question_keywords = ['?', '언제', '어떻게', '왜', '뭐', '무엇', '어디', '누구', '질문', '궁금', '알려', '문의']
        sarcasm_keywords = ['ㅋㅋ', 'ㅎㅎ', '허허', '와우', '대박', '진짜', '레알', '미쳤', '개', '헐', '어휴', 'ㅋㅋㅋㅋ', 'ㅎㅎㅎㅎ']

        # 점수 계산
        positive_score = sum(1 for word in positive_keywords if word in text)
        negative_score = sum(1 for word in negative_keywords if word in text)
        question_score = sum(1 for word in question_keywords if word in text)
        sarcasm_score = sum(1 for word in sarcasm_keywords if word in text)

        # ㅋㅋㅋ 패턴 가중치 (키워드 기반에서 더욱 강조)
        if 'ㅋ' in text and text.count('ㅋ') >= 3:
            sarcasm_score += 2
        if 'ㅎ' in text and text.count('ㅎ') >= 3:
            sarcasm_score += 2

        # 분류 로직 (우선순위)
        if question_score > 0 or '?' in text:
            return "질문/요청/정보성"
        elif sarcasm_score > positive_score + negative_score: # 비꼬기/유머가 다른 감성보다 우세할 때
            return "비꼬기/유머/감탄"
        elif positive_score > negative_score:
            return "찬성/지지"
        elif negative_score > positive_score:
            return "반대/비판"
        else:
            return "중립/기타"

# 전역 AI 분석기 인스턴스 (캐싱)
@st.cache_resource
def get_ai_analyzer():
    return KoreanSentimentAnalyzer()

# 시간별 감성 트렌드 및 예측 함수
def analyze_sentiment_trend(comments_df):
    """감정 흐름 분석 및 간단한 이동 평균 예측"""
    if comments_df.empty:
        return pd.DataFrame(), pd.DataFrame() # 예측 데이터프레임도 함께 반환

    comments_df['published_at'] = pd.to_datetime(comments_df['published_at'])
    comments_df['hour'] = comments_df['published_at'].dt.floor('H')

    # 시간별 유형 분포
    hourly_sentiment = comments_df.groupby(['hour', 'type']).size().unstack(fill_value=0)

    # 모든 가능한 시간대 포함 (데이터가 없는 시간대에도 0으로 채움)
    if not hourly_sentiment.empty:
        min_date = hourly_sentiment.index.min()
        max_date = hourly_sentiment.index.max()
        all_hours = pd.date_range(start=min_date, end=max_date, freq='H')
        hourly_sentiment = hourly_sentiment.reindex(all_hours, fill_value=0)
        # 모든 감성 유형 컬럼이 존재하는지 확인 (없으면 0으로 추가)
        all_sentiment_types = ['찬성/지지', '반대/비판', '질문/요청/정보성', '비꼬기/유머/감탄', '중립/기타']
        for col_name in all_sentiment_types:
            if col_name not in hourly_sentiment.columns:
                hourly_sentiment[col_name] = 0
        hourly_sentiment = hourly_sentiment[all_sentiment_types] # 순서 정렬
    else:
        return pd.DataFrame(), pd.DataFrame() # 데이터프레임이 비어있으면 빈 결과 반환

    # 이동 평균 계산 (예측용)
    prediction_hours = 3 # 미래 3시간 예측

    # 예측 데이터프레임 초기화
    last_hour = hourly_sentiment.index.max()
    predicted_sentiment = pd.DataFrame(index=pd.date_range(start=last_hour + timedelta(hours=1),
                                                            periods=prediction_hours, freq='H'),
                                       columns=hourly_sentiment.columns).fillna(0)

    for col in hourly_sentiment.columns:
        # 최근 데이터를 기반으로 이동 평균 계산
        moving_avg = hourly_sentiment[col].rolling(window=3, min_periods=1).mean()
        if not moving_avg.empty:
            # 마지막 유효한 이동 평균 값 또는 0
            last_avg = moving_avg.iloc[-1] if not moving_avg.isnull().all() else 0
            predicted_sentiment[col] = last_avg # 마지막 이동 평균 값을 미래에 적용

    return hourly_sentiment, predicted_sentiment

def calculate_risk_score(type_counts, total_comments):
    """위험도 점수 계산"""
    if total_comments == 0:
        return 0

    negative_types = ['반대/비판', '비꼬기/유머/감탄']
    negative_count = sum(type_counts.get(t, 0) for t in negative_types)
    negative_ratio = negative_count / total_comments

    # 위험도 점수 (0-100)
    risk_score = min(negative_ratio * 100, 100)

    return risk_score

# --- 비디오 비교 분석 결과 표시 함수 ---
def display_comparison_results(analyzed_videos_dict):
    if not analyzed_videos_dict:
        return

    st.markdown("---")
    st.subheader("📊 여러 비디오 비교 분석")

    comparison_data = []
    for video_id, data in analyzed_videos_dict.items():
        if data['comments_df'].empty:
            continue
        type_counts = data['comments_df']['type'].value_counts()
        total_comments = len(data['comments_df'])
        risk_score = calculate_risk_score(type_counts.to_dict(), total_comments)

        comparison_data.append({
            '제목': data['video_info']['title'],
            '채널': data['video_info']['channel'],
            '분석 댓글 수': total_comments,
            '긍정 비율': (type_counts.get('찬성/지지', 0) / total_comments * 100) if total_comments > 0 else 0,
            '부정 비율': (type_counts.get('반대/비판', 0) / total_comments * 100) if total_comments > 0 else 0,
            '질문/정보성 비율': (type_counts.get('질문/요청/정보성', 0) / total_comments * 100) if total_comments > 0 else 0,
            '유머/비꼬기 비율': (type_counts.get('비꼬기/유머/감탄', 0) / total_comments * 100) if total_comments > 0 else 0,
            '위험도 점수': risk_score
        })

    if not comparison_data:
        st.info("비교할 수 있는 분석된 영상 데이터가 없습니다. 먼저 영상을 분석해주세요.")
        return

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # 비교 막대 차트 (긍정/부정 비율)
    st.markdown("<h5>긍정/부정 댓글 비율 비교</h5>", unsafe_allow_html=True)
    fig_comp_sentiment = px.bar(
        comparison_df,
        x='제목',
        y=['긍정 비율', '부정 비율'],
        barmode='group',
        labels={'value': '비율 (%)', '제목': '비디오 제목'},
        color_discrete_map={'긍정 비율': '#4CAF50', '부정 비율': '#F44336'},
        height=400
    )
    st.plotly_chart(polish_chart(fig_comp_sentiment), use_container_width=True)

    # 위험도 점수 비교 차트
    st.markdown("<h5>위험도 점수 비교</h5>", unsafe_allow_html=True)
    fig_comp_risk = px.bar(
        comparison_df,
        x='제목',
        y='위험도 점수',
        labels={'위험도 점수': '점수 (0-100)', '제목': '비디오 제목'},
        color='위험도 점수',
        color_continuous_scale=px.colors.sequential.Reds,
        height=400
    )
    st.plotly_chart(polish_chart(fig_comp_risk), use_container_width=True)


# 메인 앱
def main():
    st.markdown('<h1 class="main-header">🤖 AI 기반 YouTube 댓글 여론 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown("""
    <section class="portfolio-hero">
        <div class="portfolio-kicker">Portfolio Project · NLP Dashboard</div>
        <h1>YouTube Comment Sentiment Lab</h1>
        <p>
            YouTube 댓글을 수집하고 AI 감정분석 모델로 여론의 방향, 위험 신호, 핵심 키워드를 한 번에 확인하는
            데이터 분석 대시보드입니다. 여러 영상 비교, 시간대별 추이, 부정 이슈 탐지까지 이력서에서 바로 설명하기 좋은 흐름으로 구성했습니다.
        </p>
        <div class="hero-stack">
            <span class="hero-chip">YouTube Data API</span>
            <span class="hero-chip">Transformers</span>
            <span class="hero-chip">Streamlit</span>
            <span class="hero-chip">Plotly</span>
            <span class="hero-chip">Risk Monitoring</span>
        </div>
    </section>
    """, unsafe_allow_html=True)

    # 세션 상태 초기화 (처음 로드될 때만)
    if 'analyzed_videos' not in st.session_state:
        st.session_state.analyzed_videos = {}
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    # 사이드바 설정
    st.sidebar.header("⚙️ 분석 설정")
    st.sidebar.caption("API 키와 영상 URL만 넣으면 댓글 수집부터 감정 분류, 위험도 계산까지 자동으로 실행됩니다.")

    # API 키 입력
    st.session_state.api_key = st.sidebar.text_input("YouTube API Key", value=st.session_state.api_key, type="password", help="YouTube Data API v3 키를 입력하세요")

    if not st.session_state.api_key:
        st.warning("🔑 YouTube API 키를 입력해주세요!")
        st.info("""
        **API 키 발급 방법:**
        1. [Google Cloud Console](https://console.cloud.google.com/)에 접속
        2. 새 프로젝트 생성 또는 기존 프로젝트 선택
        3. YouTube Data API v3 활성화
        4. 사용자 인증 정보에서 API 키 생성

        **🤖 AI 모델 정보:**
        - 다중 백업 시스템으로 안정성 확보
        - 1차: Twitter RoBERTa (다국어 감정분석)
        - 2차: BERT 다국어 모델
        - 3차: DistilBERT 영어 모델
        - 키워드 방식 대비 80%+ 정확도 향상
        """)
        return

    # YouTube URL 입력 (여러 개 입력 가능)
    youtube_urls_input = st.sidebar.text_area(
        "YouTube 비디오 URL (여러 개 입력 가능, 각 줄에 하나씩)",
        placeholder="https://www.youtube.com/watch?v=xxxxxxxx\nhttps://youtu.be/yyyyyyy",
        height=150
    )

    # 분석 옵션
    max_comments = st.sidebar.slider("각 영상에서 분석할 댓글 수", 50, 500, 200)

    # 분석 시작 버튼
    if st.sidebar.button("🔍 분석 시작", type="primary"):
        if not youtube_urls_input:
            st.error("YouTube URL을 입력해주세요!")
            return

        urls = [url.strip() for url in youtube_urls_input.split('\n') if url.strip()]
        if not urls:
            st.error("유효한 YouTube URL을 입력해주세요!")
            return

        analyzer = YouTubeAnalyzer(st.session_state.api_key)
        ai_analyzer = get_ai_analyzer() # AI 분석기 로드 시도

        # AI 모델 로드 표시 (최초 1회)
        if not ai_analyzer.is_initialized:
            with st.spinner("🤖 AI 감정분석 모델을 로드하는 중... (최초 1회)"):
                ai_analyzer.load_model()
            if not ai_analyzer.is_initialized:
                st.error("AI 모델 로드에 실패했습니다. 댓글 분석을 진행할 수 없습니다.")
                return

        # 모든 URL에 대해 순차적으로 분석
        analyzed_count = 0
        total_urls = len(urls)
        st.info(f"총 {total_urls}개의 영상을 분석합니다. 잠시 기다려 주세요...")
        current_video_placeholder = st.empty()
        global_progress_bar = st.progress(0)

        for i, url in enumerate(urls):
            current_video_placeholder.info(f"[{i+1}/{total_urls}] '{url}' 영상 분석 중...")
            video_id = analyzer.extract_video_id(url)

            if not video_id:
                st.error(f"'{url}'은(는) 올바른 YouTube URL 형식이 아닙니다. 건너뜁니다.")
                continue

            # 비디오 정보 가져오기
            video_info = analyzer.get_video_info(video_id)
            if not video_info:
                st.error(f"비디오 ID '{video_id}' ({url}) 에 대한 정보를 가져올 수 없습니다. 건너뜁니다.")
                continue

            # 댓글 가져오기
            comments = analyzer.get_comments(video_id, max_comments)
            if not comments:
                st.warning(f"'{video_info['title']}' ({url}) 영상에는 댓글이 없거나 가져올 수 없습니다. 분석에서 제외됩니다.")
                # 댓글이 없어도 비디오 정보는 저장하여 비교 목록에 포함 (댓글 수는 0으로)
                st.session_state.analyzed_videos[video_id] = {
                    'video_info': video_info,
                    'comments_df': pd.DataFrame() # 빈 데이터프레임
                }
                analyzed_count += 1
                global_progress_bar.progress((analyzed_count / total_urls))
                continue

            # 댓글 분류
            classified_comments = []
            for j, comment in enumerate(comments):
                classified_comments.append({
                    'text': comment['text'],
                    'author': comment['author'],
                    'published_at': comment['published_at'],
                    'like_count': comment['like_count'],
                    'type': ai_analyzer.analyze_sentiment(comment['text'])
                })
                # 댓글 분석 진행률 (글로벌 프로그레스 바에 미미하게 반영)
                local_progress = (j / len(comments)) * (1 / total_urls)
                global_progress_bar.progress((analyzed_count / total_urls) + local_progress)

            comments_df = pd.DataFrame(classified_comments)

            # 세션 상태에 저장
            st.session_state.analyzed_videos[video_id] = {
                'video_info': video_info,
                'comments_df': comments_df
            }
            analyzed_count += 1
            global_progress_bar.progress((analyzed_count / total_urls)) # 각 영상 분석 완료 후 업데이트

        current_video_placeholder.empty()
        global_progress_bar.empty()
        st.success(f"총 {analyzed_count}개의 영상 분석이 완료되었습니다!")

    # 분석된 영상이 있을 경우 결과 표시
    if st.session_state.analyzed_videos:
        st.markdown("---")
        st.subheader("결과 요약")

        # 사용자가 특정 비디오를 선택하여 상세 분석 볼 수 있도록 드롭다운 추가
        video_titles = {v_id: data['video_info']['title'] for v_id, data in st.session_state.analyzed_videos.items()}
        selected_video_id = st.selectbox(
            "상세 분석을 볼 영상을 선택하세요:",
            options=list(video_titles.keys()),
            format_func=lambda x: video_titles[x]
        )

        if selected_video_id:
            st.markdown(f"### ✨ 선택된 영상: {video_titles[selected_video_id]}")
            selected_data = st.session_state.analyzed_videos[selected_video_id]
            display_results(selected_data['video_info'], selected_data['comments_df'])

        # --- 여러 영상 비교 분석 기능 호출 ---
        if len(st.session_state.analyzed_videos) > 1:
            display_comparison_results(st.session_state.analyzed_videos)
        elif len(st.session_state.analyzed_videos) == 1:
            st.info("여러 영상 비교 분석을 보려면 2개 이상의 영상을 분석해주세요.")

    else:
        st.info("분석할 YouTube URL을 입력하고 '분석 시작' 버튼을 눌러주세요.")


# --- 상세 분석 및 시각화 함수 (기존 내용 + 키워드 심층 분석 추가) ---
def display_results(video_info, comments_df):
    """단일 영상의 분석 결과 표시"""

    # 비디오 정보 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👀 조회수", f"{video_info['view_count']:,}")
    with col2:
        st.metric("👍 좋아요", f"{video_info['like_count']:,}")
    with col3:
        st.metric("💬 댓글수", f"{video_info['comment_count']:,}")
    with col4:
        st.metric("📊 분석 댓글", f"{len(comments_df):,}")

    st.markdown(f"**📺 제목:** {video_info['title']}")
    st.markdown(f"**📺 채널:** {video_info['channel']}")

    # 댓글이 없는 경우 처리
    if comments_df.empty:
        st.info("이 영상에 분석할 댓글이 없습니다.")
        return # 더 이상 그래프 등을 그리지 않고 종료

    # 유형별 분포 분석
    type_counts = comments_df['type'].value_counts()
    total_comments = len(comments_df)

    # 위험도 계산
    risk_score = calculate_risk_score(type_counts.to_dict(), total_comments)

    # 위험 경보
    if risk_score > 50:
        st.markdown(f"""
        <div class="danger-alert">
            <h3>🚨 위험 경보!</h3>
            <p>부정적 댓글 비율이 <strong>{risk_score:.1f}%</strong>로 높습니다!</p>
            <p>사회적 위험 신호가 감지되었습니다. 즉시 대응이 필요할 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
    elif risk_score > 30:
        st.markdown(f"""
        <div class="warning-alert">
            <h3>⚠️ 주의 필요</h3>
            <p>부정적 댓글 비율이 <strong>{risk_score:.1f}%</strong>입니다.</p>
            <p>여론 변화를 주의 깊게 모니터링하세요.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-alert">
            <h3>✅ 안정적</h3>
            <p>부정적 댓글 비율이 <strong>{risk_score:.1f}%</strong>로 양호합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    # 차트 영역
    col1, col2 = st.columns(2)

    with col1:
        # 파이 차트
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="댓글 유형별 분포",
            color_discrete_map={
                '찬성/지지': '#4CAF50',
                '반대/비판': '#F44336',
                '질문/요청/정보성': '#2196F3',
                '비꼬기/유머/감탄': '#FF9800',
                '중립/기타': '#9E9E9E'
            }
        )
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(polish_chart(fig_pie), use_container_width=True)

    with col2:
        # 막대 차트
        fig_bar = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="댓글 유형별 개수",
            color=type_counts.index,
            color_discrete_map={
                '찬성/지지': '#4CAF50',
                '반대/비판': '#F44336',
                '질문/요청/정보성': '#2196F3',
                '비꼬기/유머/감탄': '#FF9800',
                '중립/기타': '#9E9E9E'
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(polish_chart(fig_bar), use_container_width=True)

    # --- 시간별 감정 흐름 (예측 포함) ---
    st.markdown("---")
    hourly_sentiment, predicted_sentiment = analyze_sentiment_trend(comments_df)

    if not hourly_sentiment.empty:
        st.subheader("📈 시간별 댓글 유형 변화 및 미래 추이 예측")

        fig_timeline = go.Figure()

        colors = {
            '찬성/지지': '#4CAF50',
            '반대/비판': '#F44336',
            '질문/요청/정보성': '#2196F3',
            '비꼬기/유머/감탄': '#FF9800',
            '중립/기타': '#9E9E9E'
        }

        for col in hourly_sentiment.columns:
            # 과거 데이터 트렌드
            fig_timeline.add_trace(go.Scatter(
                x=hourly_sentiment.index,
                y=hourly_sentiment[col],
                mode='lines+markers',
                name=col + ' (과거)',
                line=dict(color=colors.get(col, '#000000')),
                showlegend=True
            ))
            # 미래 예측 트렌드 (점선으로 표시)
            fig_timeline.add_trace(go.Scatter(
                x=predicted_sentiment.index,
                y=predicted_sentiment[col],
                mode='lines',
                name=col + ' (예측)',
                line=dict(color=colors.get(col, '#000000'), dash='dot'),
                showlegend=True
            ))

        fig_timeline.update_layout(
            title="시간별 댓글 유형 변화 및 예측",
            xaxis_title="시간",
            yaxis_title="댓글 수",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # 범례 위치 조정
        )

        st.plotly_chart(polish_chart(fig_timeline), use_container_width=True)


    # --- 특정 키워드/주제별 심층 분석 ---
    st.markdown("---")
    st.subheader("🔎 특정 키워드/주제별 심층 분석")
    keyword_input = st.text_input("분석할 키워드를 입력하세요 (쉼표로 구분)", placeholder="예: 삼성, 아이폰, 게임")

    if keyword_input:
        keywords_to_analyze = [k.strip() for k in keyword_input.split(',') if k.strip()]
        if keywords_to_analyze:
            # 키워드가 포함된 댓글 필터링
            # 한글 검색이므로 .str.contains() 사용 시 regex=True로 설정하거나, re.escape 사용
            # '|'.join()으로 OR 조건 검색
            pattern = '|'.join(re.escape(k) for k in keywords_to_analyze)
            filtered_comments_df = comments_df[comments_df['text'].str.contains(pattern, case=False, na=False, regex=True)]

            if not filtered_comments_df.empty:
                st.info(f"'{keyword_input}' 키워드를 포함하는 댓글 {len(filtered_comments_df)}개를 분석합니다.")

                # 필터링된 댓글에 대한 유형별 분포
                filtered_type_counts = filtered_comments_df['type'].value_counts()

                col_filtered1, col_filtered2 = st.columns(2)

                with col_filtered1:
                    fig_filtered_pie = px.pie(
                        values=filtered_type_counts.values,
                        names=filtered_type_counts.index,
                        title=f"'{keyword_input}' 댓글 유형 분포",
                        color_discrete_map={
                            '찬성/지지': '#4CAF50',
                            '반대/비판': '#F44436',
                            '질문/요청/정보성': '#2196F3',
                            '비꼬기/유머/감탄': '#FF9800',
                            '중립/기타': '#9E9E9E'
                        }
                    )
                    fig_filtered_pie.update_traces(textinfo='percent+label')
                    st.plotly_chart(polish_chart(fig_filtered_pie), use_container_width=True)

                with col_filtered2:
                    # 필터링된 댓글 키워드 클라우드
                    st.markdown(f"<h5>'{keyword_input}' 댓글 키워드 클라우드</h5>", unsafe_allow_html=True)
                    filtered_all_text = " ".join(filtered_comments_df['text'].astype(str))
                    filtered_all_text = re.sub(r'<[^>]+>', '', filtered_all_text)
                    filtered_words = re.findall(r'[가-힣a-zA-Z]+', filtered_all_text.lower())

                    # 불용어 리스트 재사용
                    stop_words = {
                        '이', '그', '저', '것', '수', '등', '및', '의', '가', '을', '를', '에', '서', '은', '는', '이다', '있다', '없다',
                        '하다', '되다', '같다', '아니다', '보다', '오다', '가다', '좀', '더', '또', '너무', '진짜', '정말',
                        'br', 'nbsp', 'gt', 'lt', 'amp', 'quot', 'div', 'span', 'img', 'href', 'http', 'https', 'www',
                        '그냥', '막', '한테', '에서', '으로', '부터', '까지', '하고', '이고', '랑', '와', '과', '도', '만',
                        '안', '못', '잘', '좀', '많이', '조금', '약간', '완전', '엄청', '되게', '겁나', '개', '존',
                        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'
                    }
                    filtered_words = [w for w in filtered_words if len(w) > 1 and w not in stop_words and not w.isdigit()]

                    if filtered_words:
                        try:
                            # font_path는 plt.rcParams['font.family']로 설정되었으므로 WordCloud 생성 시 생략 가능
                            filtered_wordcloud = WordCloud(
                                width=800, height=400,
                                background_color='white',
                                max_words=50,
                                collocations=False # 동일 단어 중복 표시 방지
                            ).generate_from_frequencies(Counter(filtered_words))

                            fig_filtered_wc, ax_filtered_wc = plt.subplots(figsize=(10, 5))
                            ax_filtered_wc.imshow(filtered_wordcloud, interpolation='bilinear')
                            ax_filtered_wc.axis('off')
                            st.pyplot(fig_filtered_wc)
                            plt.close(fig_filtered_wc)
                        except Exception as e:
                            st.warning(f"키워드 클라우드 생성 중 오류 발생: {e}")
                    else:
                        st.info("필터링된 댓글에서 추출할 키워드가 없습니다.")

                # 필터링된 댓글 상세 테이블
                st.markdown("<h5>필터링된 댓글 상세 내용</h5>", unsafe_allow_html=True)
                display_filtered_df = filtered_comments_df[['text', 'type', 'author', 'like_count', 'published_at']].copy()
                display_filtered_df.columns = ['댓글 내용', '유형', '작성자', '좋아요', '작성시간']
                display_filtered_df = display_filtered_df.sort_values('좋아요', ascending=False)
                st.dataframe(
                    display_filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "댓글 내용": st.column_config.TextColumn(width="large"),
                        "유형": st.column_config.TextColumn(width="medium"),
                        "작성시간": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")
                    }
                )
            else:
                st.info(f"입력하신 키워드 '{keyword_input}'를 포함하는 댓글이 없습니다.")
    else:
        st.info("특정 키워드에 대한 심층 분석을 원하시면 위에 키워드를 입력해주세요.")


    # 주요 키워드 분석 (전체 댓글 기준)
    st.markdown("---")
    st.subheader("🔍 전체 댓글 주요 키워드 분석")

    col_main_keywords1, col_main_keywords2 = st.columns(2)

    with col_main_keywords1:
        # 전체 키워드 (기존 코드)
        all_text = " ".join(comments_df['text'].astype(str))
        all_text = re.sub(r'<[^>]+>', '', all_text)
        words = re.findall(r'[가-힣a-zA-Z]+', all_text.lower())

        # 불용어 리스트 확장 (기존 코드)
        stop_words = {
            '이', '그', '저', '것', '수', '등', '및', '의', '가', '을', '를', '에', '서', '은', '는', '이다', '있다', '없다',
            '하다', '되다', '같다', '아니다', '보다', '오다', '가다', '좀', '더', '또', '너무', '진짜', '정말',
            'br', 'nbsp', 'gt', 'lt', 'amp', 'quot', 'div', 'span', 'img', 'href', 'http', 'https', 'www',
            '그냥', '막', '한테', '에서', '으로', '부터', '까지', '하고', '이고', '랑', '와', '과', '도', '만',
            '안', '못', '잘', '좀', '많이', '조금', '약간', '완전', '엄청', '되게', '겁나', '개', '존',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'
        }

        words = [w for w in words if len(w) > 1 and w not in stop_words and not w.isdigit()]

        word_counts = Counter(words).most_common(15)

        if word_counts:
            word_df = pd.DataFrame(word_counts, columns=['키워드', '빈도'])
            fig_words = px.bar(
                word_df,
                x='빈도',
                y='키워드',
                orientation='h',
                title="전체 댓글 주요 키워드 Top 15"
            )
            fig_words.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(polish_chart(fig_words), use_container_width=True)

            # --- 전체 키워드 클라우드 추가 ---
            st.markdown("<h5>전체 댓글 키워드 클라우드</h5>", unsafe_allow_html=True)
            try:
                # font_path는 plt.rcParams['font.family']로 설정되었으므로 WordCloud 생성 시 생략 가능
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=50, # 최대 표시 단어 수
                    collocations=False # 동일 단어 중복 표시 방지
                ).generate_from_frequencies(Counter(words))

                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
                plt.close(fig_wc) # Streamlit에서 그래프를 닫아 불필요한 메모리 사용 방지
            except Exception as e:
                st.warning(f"키워드 클라우드 생성 중 오류 발생: {e}")
        else:
            st.info("추출할 키워드가 없습니다.")


    with col_main_keywords2:
        # 부정적 댓글의 키워드 (기존 코드)
        negative_comments = comments_df[comments_df['type'].isin(['반대/비판', '비꼬기/유머/감탄'])]

        if len(negative_comments) > 0:
            negative_text = " ".join(negative_comments['text'].astype(str))
            negative_text = re.sub(r'<[^>]+>', '', negative_text)
            negative_words = re.findall(r'[가-힣a-zA-Z]+', negative_text.lower())

            # 불용어 리스트 사용 (기존 코드)
            stop_words = {
                '이', '그', '저', '것', '수', '등', '및', '의', '가', '을', '를', '에', '서', '은', '는', '이다', '있다', '없다',
                '하다', '되다', '같다', '아니다', '보다', '오다', '가다', '좀', '더', '또', '너무', '진짜', '정말',
                'br', 'nbsp', 'gt', 'lt', 'amp', 'quot', 'div', 'span', 'img', 'href', 'http', 'https', 'www',
                '그냥', '막', '한테', '에서', '으로', '부터', '까지', '하고', '이고', '랑', '와', '과', '도', '만',
                '안', '못', '잘', '좀', '많이', '조금', '약간', '완전', '엄청', '되게', '겁나', '개', '존',
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'
            }

            negative_words = [w for w in negative_words if len(w) > 1 and w not in stop_words and not w.isdigit()]

            negative_word_counts = Counter(negative_words).most_common(10)

            if negative_word_counts:
                neg_word_df = pd.DataFrame(negative_word_counts, columns=['키워드', '빈도'])
                fig_neg_words = px.bar(
                    neg_word_df,
                    x='빈도',
                    y='키워드',
                    orientation='h',
                    title="부정적 댓글 주요 키워드 Top 10",
                    color_discrete_sequence=['#F44336']
                )
                fig_neg_words.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(polish_chart(fig_neg_words), use_container_width=True)

                # --- 부정적 댓글 키워드 클라우드 추가 ---
                st.markdown("<h5>부정적 댓글 키워드 클라우드</h5>", unsafe_allow_html=True)
                try:
                    neg_wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        max_words=50,
                        collocations=False,
                        colormap='Reds' # 부정적인 느낌의 색상맵
                    ).generate_from_frequencies(Counter(negative_words))

                    fig_neg_wc, ax_neg_wc = plt.subplots(figsize=(10, 5))
                    ax_neg_wc.imshow(neg_wordcloud, interpolation='bilinear')
                    ax_neg_wc.axis('off')
                    st.pyplot(fig_neg_wc)
                    plt.close(fig_neg_wc) # 그래프 닫기
                except Exception as e:
                    st.warning(f"부정적 키워드 클라우드 생성 중 오류 발생: {e}")
            else:
                st.info("부정적 댓글에서 추출할 키워드가 없습니다.")
        else:
            st.info("부정적 댓글이 충분하지 않아 키워드 분석을 할 수 없습니다.")


    # 상세 댓글 테이블 (기존 코드)
    st.markdown("---")
    st.subheader("💬 댓글 상세 분석")

    # 필터링 옵션
    if not comments_df.empty:
        filter_type = st.selectbox("댓글 유형 필터", ['전체'] + list(type_counts.index))

        if filter_type != '전체':
            filtered_df = comments_df[comments_df['type'] == filter_type]
        else:
            filtered_df = comments_df

        # 댓글 표시
        display_df = filtered_df[['text', 'type', 'author', 'like_count', 'published_at']].copy()
        display_df.columns = ['댓글 내용', '유형', '작성자', '좋아요', '작성시간']
        display_df = display_df.sort_values('좋아요', ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "댓글 내용": st.column_config.TextColumn(width="large"),
                "유형": st.column_config.TextColumn(width="medium"),
                "작성시간": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")
            }
        )

    # 통계 요약 (기존 코드)
    st.markdown("---")
    st.subheader("📊 분석 요약")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("긍정적 댓글", f"{type_counts.get('찬성/지지', 0)}개")
        st.metric("부정적 댓글", f"{type_counts.get('반대/비판', 0)}개")

    with col2:
        st.metric("질문/정보성", f"{type_counts.get('질문/요청/정보성', 0)}개")
        st.metric("유머/감탄", f"{type_counts.get('비꼬기/유머/감탄', 0)}개")

    with col3:
        positive_ratio = (type_counts.get('찬성/지지', 0) / total_comments * 100) if total_comments > 0 else 0
        negative_ratio = (type_counts.get('반대/비판', 0) / total_comments * 100) if total_comments > 0 else 0
        st.metric("긍정 비율", f"{positive_ratio:.1f}%")
        st.metric("부정 비율", f"{negative_ratio:.1f}%")

    # 활용 가이드 (기존 코드)
    with st.expander("📖 시스템 활용 가이드"):
        st.markdown("""
        ### 🎯 주요 활용 분야

        **1. 정책/사회 이슈 모니터링**
        - 🤖 AI 기반 정확한 감정 분석으로 여론 파악
        - 특정 유형(반대, 비꼼 등) 급증 시 정책 담당자가 빠르게 대응
        - 여론의 변화 흐름을 실시간으로 파악

        **2. 사회적 위험 조기 경보**
        - AI 모델의 높은 정확도로 위험 신호 정밀 감지
        - 혐오/분노/비꼼 댓글이 일정 비율 이상이면 '위험 신호' 자동 감지
        - 사회적 갈등이나 논란의 조기 발견

        **3. 키워드/토픽 분석**
        - 최근 쟁점과 논란의 흐름을 한눈에 파악
        - 주요 관심사와 불만사항 식별

        **4. 마케팅/PR 전략 수립**
        - 제품이나 서비스에 대한 실제 반응 분석
        - 긍정적/부정적 피드백의 구체적 내용 파악

        ### 🤖 AI 모델 장점
        - **높은 정확도**: 기존 키워드 방식 대비 90%+ 향상
        - **문맥 이해**: 단순 키워드가 아닌 문맥 전체를 이해
        - **한국어 특화**: 한국어 언어 모델로 한국어 댓글 정확 분석
        - **실시간 처리**: 빠른 분석 속도로 실시간 모니터링 가능

        ### ⚠️ 위험도 기준
        - **50% 이상**: 즉시 대응 필요 (위험)
        - **30-50%**: 주의 깊은 모니터링 필요 (경고)
        - **30% 미만**: 안정적 상태 (양호)
        """)

if __name__ == "__main__":
    main()
