import json
from types import SimpleNamespace

import pytest

from rl_poweradj.evaluation_metrics import summarize_action_steps
from tools.summarize_evaluation_steps import summarize_result, main


def row(steps=1, success=True, initial=False, family="a", **extra):
    return dict(case_id=f"{family}-{steps}-{success}", family=family,
                initial_converged=initial, converged=success, steps=steps,
                reason="success" if success else "budget_exhausted", **extra)


def test_r04_counterexample():
    result = summarize_action_steps([row(), row(150, False)])
    assert result["mean_success_steps"] == result["median_success_steps"] == 1
    assert result["min_success_steps"] == result["max_success_steps"] == 1
    assert result["mean_all_required_steps"] == 75.5
    assert result["total_all_required_steps"] == 151
    assert result["failures"] == 1 and result["total_failed_steps"] == 150
    assert result["failure_steps_by_reason"]["budget_exhausted"] == {
        "cases": 1, "total_steps": 150, "mean_steps": 150.0}


@pytest.mark.parametrize("rows", [[], [row(0, initial=True)], [row(150, False)]])
def test_empty_success_population_is_null_not_zero(rows):
    result = summarize_action_steps(rows)
    for key in ("mean_success_steps", "median_success_steps", "min_success_steps", "max_success_steps"):
        assert result[key] is None
    assert result["successes"] == result["total_success_steps"] == 0
    assert result["mean_all_required_steps"] == (150 if rows and not rows[0]["initial_converged"] else None)
    json.dumps(result, allow_nan=False)


def test_median_boundary_and_excluded_cases():
    rows = [row(1), row(150), row(0, initial=True), {"disposition": "skipped"},
            {"disposition": "error", "reason": "solver_error", "steps": 90}]
    result = summarize_action_steps(rows)
    assert result["median_success_steps"] == result["mean_success_steps"] == 75.5
    assert result["max_success_steps"] == 150
    assert result["action_required"] == 2
    assert result["skipped_cases"] == result["initial_converged_cases"] == result["interrupted_cases"] == 1
    assert result["failures"] == 0


@pytest.mark.parametrize("steps", [-1, 1.5, float("nan"), True, None])
def test_invalid_action_counts_rejected(steps):
    with pytest.raises(ValueError, match="action count"):
        summarize_action_steps([row(steps)])


def test_no_actions_is_a_zero_step_policy_failure():
    case = row(0, False)
    case["reason"] = "no_authorized_action"
    result = summarize_action_steps([case])
    assert result["failures"] == 1 and result["mean_all_required_steps"] == 0


def test_offline_partial_report_uses_case_weighted_means():
    result = summarize_result({"status": "paused", "cases": [row(1), row(3), row(150, family="b")]})
    assert result["source_status"] == "paused"
    assert result["mean_success_steps"] == pytest.approx(154 / 3)
    assert result["families"]["a"]["mean_success_steps"] == 2
    assert result["families"]["b"]["mean_success_steps"] == 150


def test_offline_cli_never_overwrites_original_or_output(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"status": "complete", "cases": [row(), row(150, False)]}))
    original = source.read_bytes()
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["summary", "--input", str(source), "--output", str(output)])
    main()
    assert source.read_bytes() == original
    assert json.loads(output.read_text())["mean_all_required_steps"] == 75.5
    with pytest.raises(SystemExit):
        main()
    monkeypatch.setattr("sys.argv", ["summary", "--input", str(source), "--output", str(source)])
    with pytest.raises(SystemExit):
        main()
    assert source.read_bytes() == original


def test_evaluate_records_outputs_both_populations(monkeypatch):
    from rl_poweradj import graph_training as module

    cases = {"one": row(), "fail": row(150, False), "initial": row(0, initial=True)}

    class Env:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def reset(self, source, policy, expected_inputs=None):
            self.result = cases[source]
            self.done = self.result["initial_converged"]
            return None, dict(converged=self.done, steps=0, reason="initial_converged")
        def step(self, action):
            self.done = True
            return None, 0, self.result["converged"], not self.result["converged"], self.result

    monkeypatch.setattr(module, "GraphPowerAdjustmentEnv", Env)
    records = [dict(source=k, policy={}, case_id=k, grid_family_id="a") for k in cases]
    result = module.evaluate_records(SimpleNamespace(choose=lambda s: 0), None, None, records, 150)
    assert result["mean_success_steps"] == 1
    assert result["mean_required_steps"] == result["mean_all_required_steps"] == 75.5
    assert result["families"]["a"]["initial_converged_cases"] == 1


@pytest.mark.parametrize("resume", [False, True])
def test_published_output_includes_saved_prefix_and_per_family_metrics(tmp_path, monkeypatch, resume):
    from rl_poweradj import multifamily_evaluation as module
    models = tmp_path / "models"
    models.mkdir()
    model = models / "model.pt"
    model.write_bytes(b"mock")
    solver = tmp_path / "solver"
    solver.write_bytes(b"mock solver")
    publication = tmp_path / "selected.json"
    publication.write_text(json.dumps(dict(
        purpose="formal_multifamily_training", protocol="multi_family_joint_training_v1",
        selection="family_macro_validation_only", training_family_ids=module.TRAIN_FAMILIES,
        training_bus_counts=[118, 300, 2000, 3120], model=str(model),
        model_sha256=module.file_hash(model), solver_sha256=module.file_hash(solver), recipe={"horizon":150})))
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{}')
    rows = [row(), row(150, False), row(4, family="b"), row(0, initial=True, family="b")]
    records = [{"case_id":r["case_id"], "index":i} for i,r in enumerate(rows)]
    monkeypatch.setattr(module, "audit_external_manifest", lambda *a, **k: {"cases":records,"digest":"mock"})
    encoder = SimpleNamespace(verify_frozen=lambda: None)
    agent = SimpleNamespace(online=SimpleNamespace(eval=lambda:None), target=SimpleNamespace(eval=lambda:None), updates=0)
    monkeypatch.setattr(module.UnifiedFrozenEncoder, "from_project", lambda *a: encoder)
    monkeypatch.setattr(module.GraphDQN, "load", lambda *a, **k:(agent,None,None,None))
    monkeypatch.setattr(module, "state_hash", lambda *a:"unchanged")
    monkeypatch.setattr(module, "PSASPRunner", lambda *a:None)
    def evaluate(*args, **kwargs):
        for record in args[3]:kwargs["on_case"](rows[record["index"]])
    monkeypatch.setattr(module, "evaluate_records", evaluate)
    output=tmp_path / "result.json"
    if resume:
        protocol=dict(publication_sha256=module.file_hash(publication),model_sha256=module.file_hash(model),
            manifest_digest="mock",solver_sha256=module.file_hash(solver),phase="external_test",device="cpu",horizon=150,zero_updates=True)
        output.write_text(json.dumps({"protocol":protocol,"cases":rows[:1]}))
    result=module.evaluate_multifamily_published(tmp_path,publication,manifest,solver,output,
                                               phase="external_test",device="cpu",resume=resume)
    assert result["status"] == "complete"
    assert result["mean_success_steps"] == result["median_success_steps"] == 2.5
    assert result["mean_all_required_steps"] == pytest.approx(155/3)
    assert result["families"]["a"]["mean_success_steps"] == 1
    assert result["families"]["a"]["mean_all_required_steps"] == 75.5
    assert result["initial_converged_cases"] == 1
    assert json.loads(output.read_text())["total_failed_steps"] == 150
