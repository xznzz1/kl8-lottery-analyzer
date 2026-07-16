# -*- coding: utf-8 -*-
import pytest

from src import config


def test_only_supports_kl8():
    assert list(config.LOTTERY_CONFIGS.keys()) == ["kl8"]
    kl8 = config.get_lottery_config("kl8")
    assert kl8.code == "kl8"
    assert kl8.red.sequence_len == 20


def test_get_lottery_config_invalid():
    with pytest.raises(ValueError):
        config.get_lottery_config("ssq")


def test_default_outputs_stay_inside_approved_project_directories():
    allowed = {"data_cache", "results", "reports"}
    for name in allowed:
        path = config._resolve_output_path(
            config.BASE_DIR / name, config.BASE_DIR / name
        )
        relative = path.relative_to(config.BASE_DIR)
        assert relative.parts[0] in allowed

    with pytest.raises(ValueError, match="输出路径必须位于"):
        config._resolve_output_path(config.BASE_DIR.parent / "outside", config.BASE_DIR)


def test_scoped_output_rejects_other_allowed_root_and_absolute_outside_path():
    default = config.BASE_DIR / "data_cache" / "graph_embeddings.npz"
    assert (
        config.resolve_scoped_output_path(
            "data_cache/models/graph_embeddings.npz", default, "data_cache"
        )
        == config.BASE_DIR / "data_cache" / "models" / "graph_embeddings.npz"
    )

    with pytest.raises(ValueError, match="必须位于data_cache内"):
        config.resolve_scoped_output_path(
            "results/graph_embeddings.npz", default, "data_cache"
        )
    with pytest.raises(ValueError, match="输出路径必须位于"):
        config.resolve_scoped_output_path(
            config.BASE_DIR.parent / "outside.npz", default, "data_cache"
        )


def test_ensure_runtime_directories(tmp_path):
    original_paths = config.PATHS.copy()
    for key in original_paths:
        config.PATHS[key] = tmp_path / key
    config.ensure_runtime_directories()
    for path in config.PATHS.values():
        assert path.exists()
        assert path.is_dir()
    config.PATHS.update(original_paths)


def test_name_path_matches_paths(tmp_path):
    original_paths = config.PATHS.copy()
    original_name_path = config.name_path.copy()
    config.PATHS["data"] = tmp_path / "data"
    config.ensure_runtime_directories()
    config.name_path = {
        code: {
            "name": cfg.name,
            "path": f"{(config.PATHS['data'] / code).as_posix()}/",
        }
        for code, cfg in config.LOTTERY_CONFIGS.items()
    }

    assert config.name_path["kl8"]["name"] == "快乐8"
    assert config.name_path["kl8"]["path"].endswith("/kl8/")

    config.PATHS.update(original_paths)
    config.name_path = original_name_path
