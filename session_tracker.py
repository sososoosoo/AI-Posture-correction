"""
세션 데이터 추적 및 엑셀 내보내기 모듈
- 자세 측정 세션 데이터 수집
- forward neck 구간별 상세 로그
- 엑셀 파일 생성 및 다운로드 기능
"""

import datetime
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

@dataclass
class PostureEvent:
    """나쁜 자세 이벤트 기록"""
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    duration_seconds: float = 0.0
    posture_type: str = "forward_neck"  # forward_neck, neck_tilt 등
    avg_posture_score: float = 0.0
    min_posture_score: float = 1.0
    
    def finish_event(self, end_time: datetime.datetime, avg_score: float, min_score: float):
        """이벤트 종료 처리"""
        self.end_time = end_time
        self.duration_seconds = (end_time - self.start_time).total_seconds()
        self.avg_posture_score = avg_score
        self.min_posture_score = min_score

@dataclass
class SessionData:
    """전체 세션 데이터"""
    session_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    total_duration_seconds: float = 0.0
    forward_neck_total_seconds: float = 0.0
    events: List[PostureEvent] = None
    calibrated: bool = False
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
    
    def finish_session(self, end_time: datetime.datetime):
        """세션 종료 처리"""
        self.end_time = end_time
        self.total_duration_seconds = (end_time - self.start_time).total_seconds()
        # 미완료 이벤트 정리
        for event in self.events:
            if event.end_time is None:
                event.finish_event(end_time, event.avg_posture_score, event.min_posture_score)
        # forward_neck 총 시간 계산
        self.forward_neck_total_seconds = sum(e.duration_seconds for e in self.events if e.posture_type == "forward_neck")

class SessionTracker:
    """세션 추적 및 관리 클래스"""
    
    def __init__(self, data_dir: str = "session_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.current_session: Optional[SessionData] = None
        self.current_event: Optional[PostureEvent] = None
        self.posture_scores: List[float] = []
        
    def start_session(self, session_id: Optional[str] = None) -> str:
        """새로운 세션 시작"""
        if session_id is None:
            session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_session = SessionData(
            session_id=session_id,
            start_time=datetime.datetime.now()
        )
        self.current_event = None
        self.posture_scores.clear()
        
        return session_id
    
    def update_posture(self, posture_score: float, is_bad_posture: bool, posture_type: str = "forward_neck"):
        """자세 상태 업데이트"""
        if not self.current_session:
            return
        
        now = datetime.datetime.now()
        self.posture_scores.append(posture_score)
        
        if is_bad_posture and self.current_event is None:
            # 나쁜 자세 시작
            self.current_event = PostureEvent(
                start_time=now,
                posture_type=posture_type,
                avg_posture_score=posture_score,
                min_posture_score=posture_score
            )
        elif is_bad_posture and self.current_event is not None:
            # 나쁜 자세 지속 - 점수 업데이트
            self.current_event.min_posture_score = min(self.current_event.min_posture_score, posture_score)
        elif not is_bad_posture and self.current_event is not None:
            # 나쁜 자세 종료
            avg_score = sum(self.posture_scores[-10:]) / min(len(self.posture_scores), 10)  # 최근 10개 평균
            self.current_event.finish_event(now, avg_score, self.current_event.min_posture_score)
            self.current_session.events.append(self.current_event)
            self.current_event = None
    
    def stop_session(self) -> Optional[SessionData]:
        """세션 종료 및 데이터 저장"""
        if not self.current_session:
            return None
        
        now = datetime.datetime.now()
        self.current_session.finish_session(now)
        
        # JSON으로 저장
        self.save_session_json(self.current_session)
        
        session = self.current_session
        self.current_session = None
        self.current_event = None
        self.posture_scores.clear()
        
        return session
    
    def save_session_json(self, session: SessionData):
        """세션 데이터를 JSON 파일로 저장"""
        filename = f"session_{session.session_id}.json"
        filepath = self.data_dir / filename
        
        # datetime을 문자열로 변환
        data = asdict(session)
        data['start_time'] = session.start_time.isoformat()
        if session.end_time:
            data['end_time'] = session.end_time.isoformat()
        
        for event in data['events']:
            event['start_time'] = datetime.datetime.fromisoformat(event['start_time']).isoformat() if isinstance(event['start_time'], str) else event['start_time'].isoformat()
            if event['end_time']:
                event['end_time'] = datetime.datetime.fromisoformat(event['end_time']).isoformat() if isinstance(event['end_time'], str) else event['end_time'].isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_session_json(self, session_id: str) -> Optional[SessionData]:
        """JSON 파일에서 세션 데이터 로드"""
        filename = f"session_{session_id}.json"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 문자열을 datetime으로 변환
        data['start_time'] = datetime.datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            data['end_time'] = datetime.datetime.fromisoformat(data['end_time'])
        
        events = []
        for event_data in data['events']:
            event = PostureEvent(
                start_time=datetime.datetime.fromisoformat(event_data['start_time']),
                end_time=datetime.datetime.fromisoformat(event_data['end_time']) if event_data['end_time'] else None,
                duration_seconds=event_data['duration_seconds'],
                posture_type=event_data['posture_type'],
                avg_posture_score=event_data['avg_posture_score'],
                min_posture_score=event_data['min_posture_score']
            )
            events.append(event)
        
        data['events'] = events
        return SessionData(**data)
    
    def get_all_sessions(self) -> List[str]:
        """저장된 모든 세션 ID 목록 반환"""
        session_files = list(self.data_dir.glob("session_*.json"))
        session_ids = [f.stem.replace("session_", "") for f in session_files]
        return sorted(session_ids, reverse=True)  # 최신 순
    
    def export_to_excel(self, session_ids: List[str], include_details: bool = True) -> Optional[str]:
        """세션 데이터를 엑셀 파일로 내보내기"""
        if pd is None:
            raise ImportError("pandas가 필요합니다. pip install pandas openpyxl")
        
        if not session_ids:
            return None
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"posture_report_{timestamp}.xlsx"
        filepath = self.data_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 요약 시트
            summary_data = []
            all_events = []
            
            for session_id in session_ids:
                session = self.load_session_json(session_id)
                if not session:
                    continue
                
                # 가장 긴 forward_neck 구간 찾기
                longest_event = None
                if session.events:
                    longest_event = max(session.events, key=lambda e: e.duration_seconds)
                
                summary_data.append({
                    '세션 ID': session_id,
                    '시작 시간': session.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    '종료 시간': session.end_time.strftime("%Y-%m-%d %H:%M:%S") if session.end_time else "진행중",
                    '총 시간 (초)': round(session.total_duration_seconds, 1),
                    '총 시간 (분)': round(session.total_duration_seconds / 60, 1),
                    '나쁜 자세 누적 (초)': round(session.forward_neck_total_seconds, 1),
                    '나쁜 자세 누적 (분)': round(session.forward_neck_total_seconds / 60, 1),
                    '나쁜 자세 비율 (%)': round((session.forward_neck_total_seconds / session.total_duration_seconds * 100) if session.total_duration_seconds > 0 else 0, 1),
                    '가장 긴 구간 시작': longest_event.start_time.strftime("%H:%M:%S") if longest_event else "-",
                    '가장 긴 구간 지속시간 (초)': round(longest_event.duration_seconds, 1) if longest_event else 0,
                    '나쁜 자세 구간 수': len(session.events),
                    '캘리브레이션': "예" if session.calibrated else "아니오"
                })
                
                # 상세 이벤트 수집
                if include_details:
                    for event in session.events:
                        all_events.append({
                            '세션 ID': session_id,
                            '구간 시작': event.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            '구간 종료': event.end_time.strftime("%Y-%m-%d %H:%M:%S") if event.end_time else "진행중",
                            '지속시간 (초)': round(event.duration_seconds, 1),
                            '지속시간 (분)': round(event.duration_seconds / 60, 1),
                            '자세 유형': event.posture_type,
                            '평균 자세점수': round(event.avg_posture_score, 3),
                            '최저 자세점수': round(event.min_posture_score, 3)
                        })
            
            # 요약 시트 생성
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='요약', index=False)
            
            # 상세 구간별 로그 시트 생성 (옵션)
            if include_details and all_events:
                events_df = pd.DataFrame(all_events)
                events_df = events_df.sort_values(['세션 ID', '구간 시작'])
                events_df.to_excel(writer, sheet_name='구간별 상세로그', index=False)
        
        return str(filepath)

# 전역 세션 트래커 인스턴스
_global_tracker = None

def get_session_tracker() -> SessionTracker:
    """전역 세션 트래커 인스턴스 반환"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = SessionTracker()
    return _global_tracker

def format_duration(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_time(dt: datetime.datetime) -> str:
    """datetime을 표시용 문자열로 변환"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")