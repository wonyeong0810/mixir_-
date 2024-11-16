import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

class TeamAssignmentNet(nn.Module):
    def __init__(self, num_participants, num_teams):
        super(TeamAssignmentNet, self).__init__()
        self.num_participants = num_participants
        self.num_teams = num_teams
        
        # 참가자의 특성(성별, 실력)을 임베딩
        self.embedding = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 팀 할당 확률을 출력
        self.assignment = nn.Sequential(
            nn.Linear(64, num_teams),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.assignment(x)

class TeamBalancer:
    def __init__(self, num_teams):
        self.num_teams = num_teams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_data(self, participants):
        """참가자 데이터를 텐서로 변환"""
        data = []
        for p in participants:
            gender_onehot = [1, 0] if p['gender'] == 'M' else [0, 1]
            skill_normalized = p['skill'] / 10.0
            data.append(gender_onehot + [skill_normalized])
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _calculate_balance_loss(self, assignments, features):
        """팀 밸런스 손실 함수 계산"""
        team_sizes = torch.sum(assignments, dim=0)
        
        # 1. 팀 크기 균형 (좀 더 유연하게 조정)
        # 가장 큰 팀과 가장 작은 팀의 크기 차이를 최소화
        size_diff = torch.max(team_sizes) - torch.min(team_sizes)
        size_loss = size_diff ** 2
        
        # 2. 실력 균형
        skills = features[:, 2].unsqueeze(1)
        team_skills = torch.mm(assignments.t(), skills)
        # 팀별 평균 실력 계산
        team_avg_skills = team_skills / (team_sizes.unsqueeze(1) + 1e-6)
        skill_mean = torch.mean(team_avg_skills)
        skill_loss = torch.sum((team_avg_skills - skill_mean)**2)
        
        # 3. 성별 균형
        genders = features[:, :2]
        team_genders = torch.mm(assignments.t(), genders)
        # 팀별 성별 비율 계산
        team_gender_ratios = team_genders / (team_sizes.unsqueeze(1) + 1e-6)
        gender_mean = torch.mean(team_gender_ratios, dim=0)
        gender_loss = torch.sum((team_gender_ratios - gender_mean.unsqueeze(0))**2)
        
        # 전체 손실 함수 (가중치 조정)
        # 팀 크기 차이에 대한 페널티는 줄이고, 실력과 성별 균형에 더 중점
        total_loss = size_loss * 0.5 + skill_loss * 2.0 + gender_loss * 2.0
        return total_loss

    def create_balanced_teams(self, participants):
        """최적의 팀 구성 생성"""
        features = self._prepare_data(participants)
        num_participants = len(participants)
        
        model = TeamAssignmentNet(num_participants, self.num_teams).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_loss = float('inf')
        best_assignments = None
        
        # 최적화 과정
        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            
            assignment_probs = model(features)
            
            # Gumbel-Softmax trick
            temperature = max(0.5, 1.0 - epoch/1000)
            noise = -torch.log(-torch.log(torch.rand_like(assignment_probs)))
            assignments = torch.softmax((torch.log(assignment_probs) + noise) / temperature, dim=1)
            
            loss = self._calculate_balance_loss(assignments, features)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_assignments = assignments.detach()
                
            if epoch % 100 == 0 and best_loss < 0.1:
                break
        
        # 최종 팀 할당
        final_assignments = torch.argmax(best_assignments, dim=1).cpu().numpy()
        
        # 결과 정리
        teams = {i: [] for i in range(self.num_teams)}
        for i, team_idx in enumerate(final_assignments):
            teams[team_idx.item()].append(participants[i])
            
        # 팀 통계 계산
        team_stats = {}
        for team_id, members in teams.items():
            total_skill = sum(member['skill'] for member in members)
            avg_skill = total_skill / len(members) if members else 0
            gender_count = defaultdict(int)
            for member in members:
                gender_count[member['gender']] += 1
            team_stats[team_id] = {
                'total_skill': total_skill,
                'average_skill': avg_skill,
                'gender_balance': dict(gender_count),
                'members': members
            }
            
        return team_stats

def main():
    print("=== 유연한 팀 인원수 밸런싱 시스템 ===")
    
    while True:
        try:
            num_teams = int(input("팀 갯수를 입력하세요: "))
            if num_teams <= 0:
                raise ValueError
            break
        except ValueError:
            print("유효한 팀 갯수를 입력해주세요.")
    
    while True:
        try:
            num_participants = int(input("참가자 수를 입력하세요: "))
            if num_participants <= 0:
                raise ValueError
            if num_participants < num_teams:
                print("참가자 수는 팀 수보다 많아야 합니다.")
                continue
            break
        except ValueError:
            print("유효한 참가자 수를 입력해주세요.")
    
    participants = []
    for i in range(num_participants):
        print(f"\n참가자 {i + 1}:")
        while True:
            name = input("이름: ").strip()
            if name:
                break
            print("이름을 입력해주세요.")
            
        while True:
            gender = input("성별 (M/F): ").strip().upper()
            if gender in ['M', 'F']:
                break
            print("성별은 M 또는 F로 입력해주세요.")
            
        while True:
            try:
                skill = int(input("실력 점수 (1-10): "))
                if 1 <= skill <= 10:
                    break
                print("실력은 1부터 10 사이의 숫자로 입력해주세요.")
            except ValueError:
                print("유효한 숫자를 입력해주세요.")
                
        participants.append({"name": name, "gender": gender, "skill": skill})

    try:
        balancer = TeamBalancer(num_teams)
        balanced_teams = balancer.create_balanced_teams(participants)
        
        print("\n=== 팀 구성 결과 ===")
        for team_id, stats in balanced_teams.items():
            print(f"\n팀 {team_id + 1}:")
            print(f"  인원수: {len(stats['members'])}명")
            print(f"  총 실력: {stats['total_skill']}")
            print(f"  평균 실력: {stats['average_skill']:.2f}")
            print(f"  성별 분포: 남성 {stats['gender_balance'].get('M', 0)}명, "
                  f"여성 {stats['gender_balance'].get('F', 0)}명")
            print("  멤버:")
            for member in stats['members']:
                print(f"    - {member['name']} (성별: {member['gender']}, "
                      f"실력: {member['skill']})")
    
    except Exception as e:
        print(f"\n오류: {e}")
        print("프로그램을 다시 실행해주세요.")

if __name__ == "__main__":
    main()