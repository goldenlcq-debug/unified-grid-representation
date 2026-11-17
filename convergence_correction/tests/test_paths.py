import re

from rl_poweradj.paths import ProjectPaths


def _parents(cases):
    pattern = re.compile(r"^sample_(\d+)_\d+$")
    return {int(pattern.match(case.name).group(1)) for case in cases}


def test_repository_uses_grouped_lfsamples_without_parent_leakage():
    paths = ProjectPaths.from_root()
    assert paths.solver.is_file()
    train = paths.cases("train")
    validation = paths.cases("validation")
    test = paths.cases("test")
    converged = paths.cases("converged")
    assert (len(train), len(validation), len(test)) == (2416, 518, 518)
    assert len(train) + len(validation) + len(test) == 3452
    assert len(converged) == 188
    assert len(paths.cases("all")) == 3640
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    parent_splits = [_parents(split) for split in (train, validation, test)]
    assert parent_splits[0].isdisjoint(parent_splits[1])
    assert parent_splits[0].isdisjoint(parent_splits[2])
    assert parent_splits[1].isdisjoint(parent_splits[2])
    # Five parents have no usable bad child after relabeling their only
    # disturbed variant as converged; all 188 parents remain catalogued.
    assert len(set.union(*parent_splits)) == 183


def test_all_lfsample_splits_pass_active_gate_and_manifest_filter():
    paths = ProjectPaths.from_root()
    assert len(paths.reactive_cases("train")) == 2416
    assert len(paths.reactive_cases("validation")) == 518
    assert len(paths.reactive_cases("test")) == 518
