# RotorPy 기반 PN 대 E2E 요격 연구

첫 구현 범위는 `PN + PPO`와 `E2E + PPO`다. 환경·표적 궤적·PN·발사체는
`interception_env.py` 한 파일에 두고, PPO의 분포·rollout·GAE·update는
`agent/ppo.py` 한 파일에 둔다. PPO 검증 뒤 같은 환경 API에 다른 알고리즘을 추가한다.

```text
config.py             공통 물리·보상·평가 조건
interception_env.py   RotorPy, 표적, PN, 단발 capture device, Gym 환경
agent/ppo.py          PPO 모델·학습·평가·checkpoint
train.py              학습 또는 sanity 실행
record.py             best/last checkpoint의 NPZ·MP4 기록
runs/                 학습 실행 뒤 자동 생성되는 결과 폴더
```

`mode=pn`의 action은 `[azimuth, elevation, fire]` 3개다. PN이
interceptor desired acceleration을 만든다. `mode=e2e`는 앞에 world-frame
desired acceleration 3개가 더 붙어 action이 6개다. 두 방식은 동일한 RotorPy
quadrotor, 표적 분포, capture 조건, observation, reward를 쓴다.

관측은 정규화된 ground-truth
`[p_T-p_I, v_T-v_I, v_I, q_I, omega_I, launch_used]` (17차원)이다.
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
09/16: 접근 reward를 주니까 pn agent들이 그냥 타겟을 잘 따라가기만 하고 쏘지를 않음. 심지어는 앞으로 가서 타겟 배웅 나갔다가 다시 유턴해서 타겟을 따라가기만 함. 가까이로 유도함으로 인해 받는 보상보다 그냥 시간이 지나면서 기본적으로 깎이는 페널티가 더 커야함
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
