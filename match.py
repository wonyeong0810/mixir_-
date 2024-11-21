import torch
import math
import random

class TournamentBracket:
    def __init__(self, players, skills):
        """
        players: 선수 이름 리스트
        skills: 각 선수의 실력 점수 리스트 (0-100 사이의 값)
        """
        self.players = players
        self.skills = torch.tensor(skills, dtype=torch.float32)
        self.num_players = len(players)
        self.bracket_size = 2 ** math.ceil(math.log2(self.num_players))
        
    def seed_players(self):
        """선수들을 실력 기반으로 시드 배정"""
        # 실력 점수로 정렬하여 시드 배정
        seeded_indices = torch.argsort(self.skills, descending=True)
        seeded_players = [self.players[i] for i in seeded_indices]
        seeded_skills = self.skills[seeded_indices]
        
        # 부전승 처리를 위한 빈 자리 채우기
        while len(seeded_players) < self.bracket_size:
            seeded_players.append("BYE")
            seeded_skills = torch.cat([seeded_skills, torch.tensor([0.0])])
            
        return seeded_players, seeded_skills
    
    def generate_bracket(self):
        """대진표 생성"""
        seeded_players, seeded_skills = self.seed_players()
        rounds = []
        current_round = []
        
        # 첫 라운드 대진 생성
        for i in range(0, self.bracket_size, 2):
            match = {
                'player1': seeded_players[i],
                'player2': seeded_players[i + 1],
                'skill1': float(seeded_skills[i]),
                'skill2': float(seeded_skills[i + 1])
            }
            current_round.append(match)
        
        rounds.append(current_round)
        
        # 다음 라운드 대진 틀 생성
        num_rounds = int(math.log2(self.bracket_size))
        for r in range(1, num_rounds):
            current_round = []
            num_matches = self.bracket_size // (2 ** (r + 1))
            for i in range(num_matches):
                match = {
                    'player1': "TBD",
                    'player2': "TBD",
                    'skill1': 0.0,
                    'skill2': 0.0
                }
                current_round.append(match)
            rounds.append(current_round)
            
        return rounds
    
    def simulate_match(self, player1, player2, skill1, skill2):
        """경기 시뮬레이션 (실력 기반 승률 계산)"""
        if player1 == "BYE":
            return player2
        if player2 == "BYE":
            return player1
            
        # 실력차이에 따른 승률 계산
        skill_diff = abs(skill1 - skill2)
        stronger_player = player1 if skill1 > skill2 else player2
        weaker_player = player2 if skill1 > skill2 else player1
        
        # sigmoid 함수를 사용하여 승률 계산
        win_prob = torch.sigmoid(torch.tensor(skill_diff / 10.0))
        
        # 랜덤 숫자 생성하여 승자 결정
        if random.random() < win_prob:
            return stronger_player
        return weaker_player
    
    def simulate_tournament(self):
        """전체 토너먼트 시뮬레이션"""
        bracket = self.generate_bracket()
        
        for round_idx in range(len(bracket) - 1):
            current_round = bracket[round_idx]
            next_round = bracket[round_idx + 1]
            
            for match_idx, match in enumerate(current_round):
                winner = self.simulate_match(
                    match['player1'], 
                    match['player2'],
                    match['skill1'],
                    match['skill2']
                )
                
                # 다음 라운드에 승자 배정
                next_match_idx = match_idx // 2
                if match_idx % 2 == 0:
                    next_round[next_match_idx]['player1'] = winner
                else:
                    next_round[next_match_idx]['player2'] = winner
        
        return bracket

def print_bracket(bracket):
    """대진표 출력"""
    for round_idx, round_matches in enumerate(bracket):
        print(f"\nRound {round_idx + 1}:")
        for match_idx, match in enumerate(round_matches):
            print(f"Match {match_idx + 1}: {match['player1']} vs {match['player2']}")