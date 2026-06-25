# Input Ensemble Linear Independence Verification

## 결론

- 검증 대상: Weyl-Heisenberg 40개, Exact Haar D8 40개, 총 80개 ensemble
- Numerical rank tolerance: `1e-10`
- Rank failure count: `0`
- Haar 기존 CSV `gram_rank_numeric` 교차검증 불일치: `0`
- 결론: 두 benchmark 모두 요청 범위에서 `rank(G)=M`을 만족했다.

## 산출물

- Raw CSV: `raw/input_ensemble_gram_rank_results.csv`
- Summary JSON: `summaries/input_ensemble_gram_rank_summary.json`
- Weyl-Heisenberg figure: `../figures/weyl_heisenberg_gram_rank_diagnostics.png`
- Exact Haar D8 figure: `../figures/exact_haar_d8_gram_rank_diagnostics.png`

## 검증 방법

각 실험 row의 원래 seed와 benchmark 생성 코드를 재사용해 input state ensemble을 재생성했다. 재생성한 state matrix `Psi`로 Gram matrix `G = Psi Psi^dagger`를 만들고, `numpy.linalg.matrix_rank(G, tol=1e-10)`이 ensemble 크기 `M`과 같은지 확인했다.

## 수치 정밀도와 Tolerance

Gram matrix, eigenvalue, rank 계산은 재생성한 state를 `numpy.complex128`로 변환한 뒤 수행했다. 따라서 rank 판정은 NumPy/LAPACK의 double precision 계산에 기반하며, `float64` machine epsilon은 약 `2.220e-16`이다. Exact Haar D8 원래 runner도 Gram rank 진단에 `np.linalg.matrix_rank(gram, tol=1e-10)`을 사용하므로, 이번 독립 검증에서도 같은 기준을 써서 기존 CSV의 `gram_rank_numeric`과 직접 비교할 수 있게 했다.

`1e-10`은 절대 tolerance다. 여기서는 모든 input state가 normalize되어 `trace(G)=M`이고, Gram eigenvalue의 자연스러운 스케일이 `O(1)`이므로 절대 기준이 해석 가능하다. 이 값은 double precision round-off보다 충분히 크지만, 관측된 최소 eigenvalue보다는 훨씬 작다. 가장 작은 `lambda_min(G)`도 Weyl-Heisenberg에서 `0.000416716` (`lambda_min/tol = 4.16716e+06`), Exact Haar D8에서 `0.00273256` (`lambda_min/tol = 2.73256e+07`)였다. 즉 rank를 잃으려면 현재 최소 eigenvalue가 tolerance 기준까지 수백만 배 이상 작아져야 한다.

Condition number도 같은 결론을 지지한다. Weyl-Heisenberg의 최대 `kappa(G)=lambda_max/lambda_min`은 `5078.25`이고 `kappa*eps ~= 1.128e-12`이다. Exact Haar D8의 최대 condition number는 `1159.09`이고 `kappa*eps ~= 2.574e-13`이다. 두 값 모두 `1/tol = 1e10`보다 훨씬 작고, `kappa*eps`도 rank threshold보다 작다. 따라서 이번 rank 판정은 ill-conditioning 때문에 생긴 우연한 full-rank 판정으로 보기 어렵다.

## Weyl-Heisenberg Benchmark

이 benchmark는 세 qubit, 즉 Hilbert space dimension `d=8`에서 하나의 fiducial pure state를 만든 뒤, discrete Weyl-Heisenberg displacement orbit에서 `M`개의 state를 고르는 방식으로 구성된다. 각 instance마다 먼저 `|000>`에서 시작해 qubit `q=0,1,2`에 대해 독립적인 각도 `theta_q, phi_q ~ Uniform[-pi, pi]`를 고정하고, 각 qubit에 `RY(theta_q)` 다음 `RZ(phi_q)`를 적용한다. 그 다음 entangling layer로 `CNOT(0,1)`, `CNOT(1,2)`, `CZ(2,0)`를 적용하여 fiducial state `|phi>`를 얻는다.

이후 `Z_8 x Z_8` phase-space에서 서로 다른 `M`개의 label `(a_j,b_j)`를 uniform하게 비복원 추출한다. 각 label에 대해 generalized Pauli shift/phase operator를 사용하여 `|psi_j> = S X_8^{a_j} Z_8^{b_j} |phi>`를 만든다. 여기서 `Z_8|x> = omega^x|x>` (`omega = exp(2 pi i/8)`)이고 `X_8|x> = |x+1 mod 8>`이다. 마지막의 고정 unitary `S`는 qubit `0,2`에는 Hadamard, qubit `1`에는 `RZ(pi/2)`를 적용한 뒤 `CZ(0,1)`, `CZ(1,2)`를 적용하는 scrambler다. 이 `S`는 모든 state에 동일하게 작용하므로 Gram matrix spectrum과 linear independence 여부를 보존한다.

따라서 검증한 ensemble은 임의 fiducial state의 finite Weyl-Heisenberg orbit에서 고른 `M`개의 순수상태 ensemble이며, seed는 fiducial state와 phase-space label set을 재현하기 위해서만 사용되었다.

![Weyl-Heisenberg Gram rank diagnostics](../figures/weyl_heisenberg_gram_rank_diagnostics.png)

- checked: `40`
- rank failures: `0`
- minimum lambda_min(G): `0.000416716`

## Exact Haar D8 Benchmark

이 benchmark는 같은 세 qubit Hilbert space, 즉 `C^8`에서 Haar-random pure state를 직접 샘플링한다. 각 instance마다 먼저 길이 12의 nested state list를 만든다. 각 state는 complex Gaussian vector `z in C^8`를 성분별로 `Re z_k, Im z_k ~ N(0,1)`에서 독립적으로 뽑은 뒤 `|psi> = z / ||z||_2`로 정규화하여 얻는다. complex Gaussian을 정규화한 분포는 complex unit sphere 위의 Haar measure와 일치하므로, 이 절차는 `d=8` Haar pure-state ensemble을 생성한다.

실험에서 `M=5,...,8`을 바꿀 때는 같은 instance의 nested list에서 처음 `M`개 state를 취한다. 즉 `M=5` ensemble은 `M=6,7,8` ensemble의 prefix가 되도록 구성되어 있으며, 이번 검증도 이 nested 구조를 그대로 재현했다. 현재 결과셋에서는 별도의 global phase fixing을 사용하지 않았고, 회로 실행 단계에서는 주어진 state vector를 정확한 state-preparation unitary로 준비한다.

![Exact Haar D8 Gram rank diagnostics](../figures/exact_haar_d8_gram_rank_diagnostics.png)

- checked: `40`
- rank failures: `0`
- minimum lambda_min(G): `0.00273256`

## M별 요약

### Weyl-Heisenberg

| M | count | rank failures | min lambda_min | min lambda_min / tol | max lambda_min | max condition number |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 10 | 0 | 0.0488626 | 4.88626e+08 | 0.384501 | 45.0775 |
| 6 | 10 | 0 | 0.037891 | 3.7891e+08 | 0.286492 | 66.6713 |
| 7 | 10 | 0 | 0.000416716 | 4.16716e+06 | 0.149156 | 5078.25 |
| 8 | 10 | 0 | 0.00161945 | 1.61945e+07 | 0.0426875 | 1375.36 |

### Exact Haar D8

| M | count | rank failures | min lambda_min | min lambda_min / tol | max lambda_min | max condition number |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 10 | 0 | 0.0760603 | 7.60603e+08 | 0.336507 | 31.588 |
| 6 | 10 | 0 | 0.0352499 | 3.52499e+08 | 0.208033 | 63.4204 |
| 7 | 10 | 0 | 0.0143791 | 1.43791e+08 | 0.0875477 | 187.7 |
| 8 | 10 | 0 | 0.00273256 | 2.73256e+07 | 0.0322068 | 1159.09 |
