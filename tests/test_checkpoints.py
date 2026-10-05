import os

from myagy.core.checkpoints import Checkpoints


def test_snapshot_diff_and_restore(git_repo):
    root = Checkpoints.repo_root(str(git_repo))
    before = Checkpoints.snapshot(root)
    (git_repo / "a.txt").write_text("changed\n")
    (git_repo / "new.py").write_text("print(1)\n")
    after = Checkpoints.snapshot(root)
    assert before and after and before != after
    assert Checkpoints.changed(root, before, after) == {"a.txt": "M", "new.py": "A"}
    assert "+changed" in Checkpoints.diff(root, before, after)

    touched = Checkpoints.restore(root, before, after)
    assert sorted(touched) == ["a.txt", "new.py"]
    assert (git_repo / "a.txt").read_text() == "a\n"
    assert not (git_repo / "new.py").exists()


def test_restore_deleted_file(git_repo):
    root = Checkpoints.repo_root(str(git_repo))
    before = Checkpoints.snapshot(root)
    os.remove(git_repo / "a.txt")
    after = Checkpoints.snapshot(root)
    assert Checkpoints.changed(root, before, after) == {"a.txt": "D"}
    Checkpoints.restore(root, before, after)
    assert (git_repo / "a.txt").read_text() == "a\n"


def test_snapshot_does_not_touch_real_index(git_repo):
    import subprocess
    root = Checkpoints.repo_root(str(git_repo))
    (git_repo / "untracked.txt").write_text("x")
    Checkpoints.snapshot(root)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=git_repo, capture_output=True, text=True).stdout
    assert "?? untracked.txt" in status


def test_not_a_repo(tmp_path):
    assert Checkpoints.repo_root(str(tmp_path)) is None
