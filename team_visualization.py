import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import platform

os = platform.system()

# Windows
if os == 'Windows':
    plt.rc('font', family= 'Malgun Gothic')

class TeamAssignmentNet(nn.Module):
    def __init__(self, num_participants, num_teams):
        super(TeamAssignmentNet, self).__init__()
        self.num_participants = num_participants
        self.num_teams = num_teams
        
        self.embedding = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
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
        data = []
        for p in participants:
            gender_onehot = [1, 0] if p['gender'] == 'M' else [0, 1]
            skill_normalized = p['skill'] / 10.0
            data.append(gender_onehot + [skill_normalized])
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _calculate_balance_loss(self, assignments, features):
        team_sizes = torch.sum(assignments, dim=0)
        
        size_diff = torch.max(team_sizes) - torch.min(team_sizes)
        size_loss = size_diff ** 2
        
        skills = features[:, 2].unsqueeze(1)
        team_skills = torch.mm(assignments.t(), skills)
        team_avg_skills = team_skills / (team_sizes.unsqueeze(1) + 1e-6)
        skill_mean = torch.mean(team_avg_skills)
        skill_loss = torch.sum((team_avg_skills - skill_mean)**2)
        
        genders = features[:, :2]
        team_genders = torch.mm(assignments.t(), genders)
        team_gender_ratios = team_genders / (team_sizes.unsqueeze(1) + 1e-6)
        gender_mean = torch.mean(team_gender_ratios, dim=0)
        gender_loss = torch.sum((team_gender_ratios - gender_mean.unsqueeze(0))**2)
        
        total_loss = size_loss * 0.5 + skill_loss * 2.0 + gender_loss * 2.0
        return total_loss

    def visualize_teams(self, team_stats):
        """팀 구성 결과를 시각화"""
        # 1. 팀별 평균 실력 그래프
        plt.figure(figsize=(15, 10))
        
        # 1-1. 평균 실력 막대 그래프
        plt.subplot(2, 2, 1)
        team_names = [f'Team {i+1}' for i in range(self.num_teams)]
        avg_skills = [stats['average_skill'] for stats in team_stats.values()]
        plt.bar(team_names, avg_skills)
        plt.title('팀별 평균 실력')
        plt.ylim(0, 10)
        plt.ylabel('평균 실력')
        
        # 1-2. 성별 분포 스택 바 차트
        plt.subplot(2, 2, 2)
        male_counts = [stats['gender_balance'].get('M', 0) for stats in team_stats.values()]
        female_counts = [stats['gender_balance'].get('F', 0) for stats in team_stats.values()]
        
        width = 0.35
        plt.bar(team_names, male_counts, width, label='남성')
        plt.bar(team_names, female_counts, width, bottom=male_counts, label='여성')
        plt.title('팀별 성별 분포')
        plt.ylabel('인원 수')
        plt.legend()
        
        # 1-3. 팀별 실력 분포 박스플롯
        plt.subplot(2, 2, 3)
        team_skills = []
        for stats in team_stats.values():
            team_skills.append([member['skill'] for member in stats['members']])
        plt.boxplot(team_skills, labels=team_names)
        plt.title('팀별 실력 분포')
        plt.ylabel('실력')
        
        # 1-4. 팀별 인원수
        plt.subplot(2, 2, 4)
        team_sizes = [len(stats['members']) for stats in team_stats.values()]
        plt.bar(team_names, team_sizes)
        plt.title('팀별 인원수')
        plt.ylabel('인원 수')
        
        plt.tight_layout()
        plt.show()

    def create_balanced_teams(self, participants):
        features = self._prepare_data(participants)
        num_participants = len(participants)
        
        model = TeamAssignmentNet(num_participants, self.num_teams).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_loss = float('inf')
        best_assignments = None
        
        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            
            assignment_probs = model(features)
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
        
        final_assignments = torch.argmax(best_assignments, dim=1).cpu().numpy()
        
        teams = {i: [] for i in range(self.num_teams)}
        for i, team_idx in enumerate(final_assignments):
            teams[team_idx.item()].append(participants[i])
            
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
        
        # 시각화 결과 표시
        balancer.visualize_teams(balanced_teams)
    
    except Exception as e:
        print(f"\n오류: {e}")
        print("프로그램을 다시 실행해주세요.")

if __name__ == "__main__":
    main()