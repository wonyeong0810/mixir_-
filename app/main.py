import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('skill_team_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Player(BaseModel):
    """플레이어 데이터 모델"""
    name: str
    skill_level: str  # '상', '중', '하' 중 하나

class SkillTeams(BaseModel):
    """실력별 팀 구성 모델"""
    high_team: List[Player]
    mid_team: List[Player]
    low_team: List[Player]

class TeamCreationConfig(BaseModel):
    """팀 생성 설정 모델"""
    spreadsheet_id: str = Field(..., min_length=10)
    range_name: str = Field(..., min_length=5)

class GoogleSheetsService:
    """Google Sheets 서비스 관리 클래스"""
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

    @staticmethod
    def get_credentials():
        """Google Sheets API 자격 증명 획득"""
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')

        try:
            if os.path.exists(token_path):
                with open(token_path, 'r') as token:
                    creds = Credentials.from_authorized_user_file(token_path, GoogleSheetsService.SCOPES)
            
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            
            if not creds:
                flow = Flow.from_client_secrets_file(
                    credentials_path, 
                    scopes=GoogleSheetsService.SCOPES
                )
                flow.run_local_server(port=0)
                creds = flow.credentials

                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            return creds
        except Exception as e:
            logger.error(f"Google Sheets 인증 오류: {e}")
            raise HTTPException(status_code=500, detail="Google Sheets 인증 실패")

def create_google_sheets_service():
    """Google Sheets 서비스 생성"""
    try:
        creds = GoogleSheetsService.get_credentials()
        return build('sheets', 'v4', credentials=creds)
    except Exception as e:
        logger.error(f"Google Sheets 서비스 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="Google Sheets 서비스 생성 실패")

def read_player_data(service, spreadsheet_id: str, range_name: str) -> List[Player]:
    """구글 시트에서 플레이어 데이터 읽어오기"""
    try:
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get('values', [])

        if not values or len(values) <= 1:
            raise HTTPException(status_code=404, detail="플레이어 데이터를 찾을 수 없음")

        players = []
        # 첫 번째 행(헤더) 제외하고 데이터 처리
        for row in values[1:]:
            try:
                # 데이터 검증 및 추출
                player = Player(
                    name=row[0].strip(),  # 이름
                    skill_level=row[1].strip()  # 실력 레벨
                )
                players.append(player)
            except (IndexError, ValueError) as e:
                logger.warning(f"데이터 형식 오류: {e}")

        if not players:
            raise HTTPException(status_code=400, detail="유효한 플레이어 데이터가 없음")

        return players

    except Exception as e:
        logger.error(f"플레이어 데이터 읽기 실패: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 읽기 오류: {str(e)}")

def divide_players_by_skill(players: List[Player]) -> SkillTeams:
    """
    플레이어를 실력별로 분류
    
    입력 규칙:
    1. 실력 레벨은 '상', '중', '하' 중 하나여야 함
    2. 같은 실력 레벨 내에서는 이름 순으로 정렬
    
    Args:
        players (List[Player]): 전체 플레이어 목록
    
    Returns:
        SkillTeams: 실력별로 분류된 팀
    """
    try:
        # 실력별 플레이어 분류
        skill_groups = {
            '상': sorted([p for p in players if p.skill_level == '상'], key=lambda x: x.name),
            '중': sorted([p for p in players if p.skill_level == '중'], key=lambda x: x.name),
            '하': sorted([p for p in players if p.skill_level == '하'], key=lambda x: x.name)
        }

        # SkillTeams 객체 생성
        skill_teams = SkillTeams(
            high_team=skill_groups['상'],
            mid_team=skill_groups['중'],
            low_team=skill_groups['하']
        )

        # 로깅
        logger.info(f"실력별 팀 생성 완료:")
        logger.info(f"상팀: {len(skill_teams.high_team)}명")
        logger.info(f"중팀: {len(skill_teams.mid_team)}명")
        logger.info(f"하팀: {len(skill_teams.low_team)}명")

        return skill_teams

    except Exception as e:
        logger.error(f"팀 분류 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="팀 분류 실패")

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="실력별 팀 생성 API",
    description="구글 시트 데이터 기반 실력별 팀 분류",
    version="1.0.0"
)

@app.post("/divide-teams", response_model=SkillTeams)
async def divide_teams(config: TeamCreationConfig):
    """
    Google Sheets에서 데이터를 읽어 실력별로 팀 분류
    
    Args:
        config (TeamCreationConfig): 스프레드시트 설정 정보
    
    Returns:
        SkillTeams: 실력별로 분류된 팀 정보
    """
    try:
        # Google Sheets 서비스 생성
        service = create_google_sheets_service()

        # 플레이어 데이터 읽기
        players = read_player_data(service, config.spreadsheet_id, config.range_name)

        # 실력별 팀 분류
        skill_teams = divide_players_by_skill(players)

        return skill_teams

    except Exception as e:
        logger.error(f"팀 분류 API 호출 중 오류: {e}")
        raise

# 서버 실행 설정
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=os.getenv("APP_HOST", "0.0.0.0"), 
        port=int(os.getenv("APP_PORT", 8000)),
        reload=True
    )
"""

이 코드의 주요 특징은 다음과 같습니다:

1. 실력별 팀 분류 기능
   - '상', '중', '하' 팀으로 명확하게 분류
   - 각 팀 내에서 이름 순으로 정렬
   - 실력 레벨에 따른 엄격한 분류

2. API 엔드포인트
   - `/divide-teams` 엔드포인트를 통해 실력별 팀 분류
   - 입력: 스프레드시트 ID, 데이터 범위
   - 출력: 상팀, 중팀, 하팀으로 구성된 팀 정보

3. 데이터 검증 및 오류 처리
   - 스프레드시트 데이터 유효성 검사
   - 상세한 로깅
   - 예외 상황에 대한 명확한 오류 메시지

"""