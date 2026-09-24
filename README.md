# RotorPy 기반 PN 대 E2E 요격 연구

환경·표적 궤적·PN·발사체는 `interception_env.py` 한 파일에 두고,
PPO, A2C, DDPG, TD3, SAC는 각각 `agent/`의 독립 파일에 둔다.
모든 알고리즘은 동일한 PN/E2E 환경과 평가 seed bank를 사용한다.

```text
config.py             공통 물리·보상·평가 조건
interception_env.py   RotorPy, 표적, PN, 단발 capture device, Gym 환경
agent/ppo.py          PPO 모델·학습·평가·checkpoint
train.py              학습 또는 sanity 실행
record.py             best/last/stage별 checkpoint의 NPZ·MP4 기록
runs/                 학습 실행 뒤 자동 생성되는 결과 폴더
```

`mode=pn`의 action은 `[LOS 좌우 발사각, LOS 상하 발사각]` 2개다. PN 유도와 발사 시점은 규칙 기반이다.
`mode=e2e`는 `[a_LOS, a_horizontal, a_vertical, LOS 좌우 발사각, LOS 상하 발사각]` 5개다.
두 방식은 동일한 RotorPy
quadrotor, 표적 분포, capture 조건, observation, reward를 쓴다.

관측은 정규화된 ground-truth
`[p_T-p_I, v_T-v_I, p_I, v_I, q_I, omega_I, a_I_previous, time, launch_used, gate_open, curriculum_stage]` (26차원)이다.
정상 논문 실험에는 timeout이 없다. hit, device miss, target sphere exit,
interceptor sphere exit만 `terminated=True`다. `debug_max_steps`를 켠 경우에만
개발용 `truncated=True`가 생기며 PPO는 그 실제 final observation으로 bootstrap한다.

## 설치와 실행

```bash
python -m pip install -r requirements.txt
python train.py --mode pn --physics-engine rotorpy --sanity
python train.py --mode pn --physics-engine rotorpy --seed 0
python train.py --mode e2e --physics-engine rotorpy --seed 0
```

`--total-steps`를 생략하면 수동으로 중지할 때까지 학습하며, `Ctrl+C`로 중지해도
`last.pt`를 저장한다. 유한한 smoke test에만 `--total-steps`를 지정한다.

빠른 문법·환경 점검만 할 때는 `--physics-engine simple`을 쓸 수 있지만 논문 결과에는
반드시 `--physics-engine rotorpy`를 쓴다. 같은 mode/seed 결과를 다시 만들려면
`--overwrite`를 명시한다.

```bash
python record.py --mode pn --seed 0 --scenario-seed 10000
```

MP4에는 `ffmpeg`가 필요하며, 없더라도 trajectory `.npz`는 먼저 저장된다.

타겟의 random성은 entry 위치 route방향 A_horizontal, A_vertical, ω_horizontal, ω_vertical ou에 의해서만 결정되고 \(A\sin(\omega t+\phi)\)의 파이는 0으로 둔다
이는 타겟이 공역에 들어오기 전까지는 기본 운동을 하다가 공역에 들어오고나서부터 회피기동을 한다고 해석할수 있다.
그리고 agent에서 reinforce는 뺌
==================================================
09/16 v1결과: 접근 reward를 주니까 pn agent들이 그냥 타겟을 잘 따라가기만 하고 쏘지를 않음. 심지어는 앞으로 가서 타겟 배웅 나갔다가 다시 유턴해서 타겟을 따라가기만 함. 가까이로 유도함으로 인해 받는 보상보다 그냥 시간이 지나면서 기본적으로 깎이는 페널티가 더 커야함
일반 step:
r = 0.5 * (d_previous - d_current) / 100 - 0.005

명중:
r_hit = 10 + 5 * clip(1 - capture_time / 25, 0, 1)

그물 miss:
r_miss = -12 + 4 * clip(1 - min_net_distance / 10, 0, 1)

미발사 상태로 타겟 탈출:
r_target_exit = -15

요격기 공역 이탈:
r_interceptor_exit = -15

가속도·jerk 페널티는 제거했지만 물리적 제한은 유지한다.
==========================================================
0918 v2결과: pn에서는 여전히 마중나갔다가 요격 안하고 뒤꽁무니만 쫓는 현상이 나타남. target_exit으로 끝남 -> 아예 접근으로 인한 보상을 없애야 할것으로 예상됨
e2e에서는 요격기가 이상한 곳으로 가서 intercepter_exit으로 끝남. 학습이 부족해서인 거 같기는 한데 학습 부족 문제가 아니라면, "어차피 쏴도 못맞출거  에피소드를 일찍 끝내서 누적 페널티를 적게 받자" 정책을 택한 거 같음.
그리고 내가 실수한게 지금까지 발사각과 e2e 가속도를 los기준이 아니라 world기준 각도와 가속도를 출력하도록 하고 있었음. 이것들을 los기준으로 바꾸었음.

상대 방위각:
action = -1 → -180도
action =  0 → LOS 정면
action = +1 → +180도

상대 고도각:
action = -1 → LOS보다 아래로 90도
action =  0 → LOS 정면
action = +1 → LOS보다 위로 90도

그리고 지금까지는
physics_substeps = control_dt / physics_dt
                 = 0.05 / 0.01
                 = 5번
이렇게 했었는데 학습이 너무 느려서 self.physics_dt = self.control_dt / 3.0로 바꿈. 이제는 substep이 5가 아니라 3임.
일단 jerk제한은 그대로 둠. 주말이라 각 경우에 대해 step을 300만이 아니라 500만으로 늘림.
===================================================
09/21 구현 사항
학습 실패의 주요 원인을 희소한 명중 보상, 조기 발사, 불완전한 관측값 및 발사 후 무의미한 transition으로 판단하여 다음과 같이 환경을 수정했다.
- Observation에 정규화된 요격기 위치, 직전 가속도 및 episode 시간을 추가했다.
- 발사 후에는 agent action과 요격기 운동·보상 계산을 중단한다.
- 발사 후 환경 내부에서 타겟과 탄도 그물만 hit/miss까지 시뮬레이션하고 하나의 terminal transition을 반환한다.
- 아래의 발사 gate·RL 발사 시점·residual 설계는 09/22 분석 뒤 현재의 규칙 기반 발사 시점과 직접 발사각 action으로 대체했다.
- 빗나간 경우에도 min_net_distance가 작을수록 보상을 받도록 40m 범위의 연속 near-miss 보상을 적용한다.
- 평가 CSV에 gate 진입률, 발사율·시점·거리, 최소 그물 거리 및 종료 원인별 비율을 기록한다.
최종 평가에서는 실제 조건인 capture radius 2m와 전체 타겟 기동 조건을 그대로 사용한다.
그리고 rotorpy버전이 서버랑 로컬이랑 달라서 서버 업데이트 함
Python 3.10.20
RotorPy 2.1.3
NumPy 2.2.6
SciPy 1.15.3
=======================================
## 09/22 실패 분석 반영

- PN은 유도와 발사 시점을 규칙 기반으로 고정하고, RL은 LOS 기준 좌우 ±30도·상하 ±20도의 발사 방향을 직접 결정한다. 발사는 `거리 <= 15 m`이고 `closing speed >= 5 m/s`인 첫 step에 실행한다.
- E2E stage 0은 RL 유도와 규칙 기반 탄도 조준을 사용한다. 평가 성공률 80%를 2회 연속 만족하면 stage 1로 넘어가 RL이 유도와 발사 방향을 함께 결정한다.
- 사용되지 않는 발사각 action은 PPO/A2C의 policy loss에서 제외하고 DDPG/TD3/SAC에서는 0으로 masking한다. 단계 전환 시 off-policy replay buffer를 비운다.
- 단계별 최고 모델은 `stage0_best.pt`, `stage1_best.pt`로 저장하고, 평가 CSV에는 rule/RL 조준 사용률을 기록한다.

## 09/23 PPO 수치 안정화

- `-log_prob(old action)`을 entropy로 사용하던 오류를 Gaussian entropy로 수정했다.
- tanh-Gaussian log probability의 Jacobian을 수치적으로 안정적인 식으로 계산한다.
- PPO/A2C의 `log_std`를 `[-5, 1]`, gradient norm을 `0.5`로 제한한다.
==========================================
## 0924
현재 상태 다시 정리
PN:한 단계만 사용
Stage 1: rl_los_aim
유도              = PN 규칙 기반
발사 시점         = 규칙 기반
발사 방향         = RL
action            = [LOS 좌우각, LOS 상하각]

발사조건
거리 ≤ 15 m
closing speed ≥ 5 m/s

E2E:
Stage 0: guidance_with_rule_aim
유도              = RL
발사 시점         = 규칙 기반
발사 방향         = 탄도해 규칙
학습 action       = 가속도 3개
발사각 2개        = masking

평가 성공률 80% 이상을 2회 연속 달성하면
Stage 1: rl_los_aim
유도              = RL
발사 시점         = 규칙 기반
발사 방향         = RL
학습 action       = 가속도 3개 + 발사각 2개

###  수정 사항

- E2E 발사 step에서 실제로 사용되지 않는 가속도 action까지 학습되던 mask 오류를 수정했다.
- Stage 0은 최소 200만 step 학습하고, 성공률과 gate 진입률이 모두 90% 이상인 평가를 5회 연속 통과해야 Stage 1로 전환한다.
- Stage 1 진입 전에 analytic ballistic angle로 발사각 출력을 짧게 imitation 학습한다. 이후 유도 network를 동결하고 발사각만 학습한다.
- Stage 1을 최소 100만 step 학습한 뒤 같은 90%·5회 조건을 만족하면 Stage 2로 전환한다. Stage 2에서는 전체 network를 learning rate 1e-4로 joint fine-tuning한다.
- 단계 전환 시 critic과 optimizer를 초기화하며, off-policy agent는 이전 단계의 replay buffer도 비운다. `stage2_best.pt` 저장과 `last.pt` 재개 학습을 지원한다.
- PN은 모든 agent에서 동일한 PN 유도, 15 m·closing speed 5 m/s gate, 2차원 LOS 발사각, 보상과 평가 seed를 사용한다.
