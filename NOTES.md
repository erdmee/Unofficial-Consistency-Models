# Notes

이런저런 고려해볼만한 사항들. 코드만 봐서는 안 드러나는 결정 근거와 논의 사항을 모아둠.

---

## 1. 논문과 실제 구현의 다름

### 1.1 Loss weighting λ(t_n) — EDM weighting 사용

- **논문 본문 (Song et al. 2023, Section 3)**: `λ(t_n)`을 generic weighting function으로 정의, default = 1 (uniform).
- **논문 Appendix C (hyperparameter table)**: 실제 실험에서는 Karras weighting을 사용한다고 명시.
- **오피셜 OpenAI 코드 (`cm/karras_diffusion.py`)**: `weight_schedule="karras"`가 default. 식은 `1/σ² + 1/σ_data²`.
- **우리 코드 (`cm/training/losses.py`)**: 같은 식을 사용 (`snrs + 1/sigma_data²`).

#### 왜 default(λ=1) 안 쓰고 EDM weight를 쓰나
1. **σ 레벨별 gradient scale 균등화**: `1/σ²` 항이 작은 σ에서 큰 가중치를 줘서, 모든 σ 레벨이 학습 신호에 비슷하게 기여하도록 함. λ=1을 쓰면 큰 σ가 gradient를 지배함.
2. **EDM과의 일관성**: CM은 EDM의 preconditioning (c_skip/c_out/c_in)을 그대로 차용 (`cm/models/precond.py`). weighting도 같은 EDM 컨벤션을 따르는 게 자연스러움.
3. **σ → 0 안정화**: `+1/σ_data²` 항이 weight 발산을 막는 floor 역할.

#### 결론
논문 본문의 λ=1은 *수식 유도의 일반성을 위한 표기*이고, SOTA 재현을 노린다면 EDM weighting이 정답. 일단 그대로 두고, 나중에 ablation 필요해지면 `weight_schedule` 인자로 분기.

### 1.2 σ schedule 정렬 방향 — ascending으로 통일

- **Karras et al. 2022 (EDM) Eq. 5**: `t_0 = σ_max, ..., t_{N-1} = σ_min` (descending). sampler가 high → low로 내려가니까 자연스러움.
- **오피셜 OpenAI 코드 (`get_sigmas_karras`)**: descending. 끝에 `append_zero` 붙여서 sampler 종단 처리.
- **CM 논문 (Song et al. 2023)**: `t_0 = σ_min, t_n < t_{n+1}` (ascending). Algorithm 2/3의 의사코드가 이 컨벤션 기준으로 쓰여 있음.
- **우리 코드 (`cm/diffusion/karras_schedule.py`)**: **ascending** 채택. `sigmas[n] = t_n`이 paper notation과 1:1 매칭됨.

#### 왜 ascending으로 갔나
1. **paper 의사코드와 1:1 매칭**: `losses.py`에서 `t_n = sigmas[indices]`, `t_n_plus_1 = sigmas[indices + 1]`이 그대로 paper Algorithm 표기와 일치. 인덱스 플립 없음.
2. **오피셜의 `t / t2` naming 회피**: OpenAI 코드는 `t / t2`를 쓰는데 어느 쪽이 high/low인지 이름만 봐선 모름. paper notation `t_n / t_{n+1}`로 가면 명확.
3. **online이 high, target이 low**: ascending 기준 `online_model(x_t_n_plus_1, t_n_plus_1)`, `target_model(x_t_n, t_n)`이 자연스럽게 읽힘.

#### 트레이드오프
- ➖ Karras 2022/OpenAI 코드와 sigmas 인덱스를 줄 단위로 비교할 때 헷갈릴 수 있음 (그쪽은 descending이라 `sigmas[0]`이 σ_max).
- ➕ CM 본문 재현이 목표라면 CM 논문 컨벤션을 따르는 게 정합성↑.

#### 결론
CM 논문 컨벤션 우선. sampler 짤 때 종단 0이 필요해지면 호출부에서 `torch.cat([sigmas, sigmas.new_zeros(1)])`로 추가하면 됨 (`append_zero`를 schedule에 내장하지 않음).

---
