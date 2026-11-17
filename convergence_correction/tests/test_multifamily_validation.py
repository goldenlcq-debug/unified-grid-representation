from rl_poweradj.multifamily_sampler import TRAIN_FAMILIES
from rl_poweradj.multifamily_validation import rank_validation


def test_checkpoint_rank_prefers_family_macro_and_then_worst_family():
    def result(successes):
        return {"cases": [
            {"family": family, "initial_converged": False,
             "converged": index < successes[family], "steps": 1 if index < successes[family] else 150}
            for family in TRAIN_FAMILIES for index in range(10)
        ]}
    balanced = dict.fromkeys(TRAIN_FAMILIES, 7)
    lopsided = dict(zip(TRAIN_FAMILIES, (10, 10, 8, 0)))
    good_rank, detail = rank_validation(result(balanced), horizon=150, episode=100)
    weak_rank, _ = rank_validation(result(lopsided), horizon=150, episode=99)
    assert good_rank > weak_rank
    assert detail["macro_success_rate"] == 0.7
    assert detail["worst_family_success_rate"] == 0.7
