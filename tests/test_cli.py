from click.testing import CliRunner
from ais_shader.cli import cli, _default_output_path
from pathlib import Path

def test_default_output_path():
    assert _default_output_path(Path("foo.csv"), ".geoparquet") == Path("foo.geoparquet")
    assert _default_output_path(Path("foo.csv.zip"), ".geoparquet") == Path("foo.geoparquet")
    assert _default_output_path(Path("foo.geoparquet"), "-trajectorized.geoparquet") == Path("foo-trajectorized.geoparquet")
    assert _default_output_path(Path("foo-trajectorized.geoparquet"), "-lines.geoparquet") == Path("foo-lines.geoparquet")

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "trajectory" in result.output
    assert "convert" in result.output
    # Check that old root level commands are gone
    assert "trajectorize" not in result.output
    assert "generate-lines" not in result.output
    assert "generate-segments" not in result.output
    assert "convert-csv" not in result.output
    assert "convert-wkb" not in result.output
    assert "convert-ndjson" not in result.output

def test_convert_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
    assert "csv" in result.output
    assert "wkb" in result.output
    assert "ndjson" in result.output

def test_trajectory_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["trajectory", "--help"])
    assert result.exit_code == 0
    assert "compute" in result.output
    assert "to-linestring" in result.output
    assert "to-segment" in result.output

def test_trajectory_compute_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["trajectory", "compute", "--help"])
    assert result.exit_code == 0
    assert "--epoch-time" in result.output

def test_trajectory_to_segment_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["trajectory", "to-segment", "--help"])
    assert result.exit_code == 0
    assert "--epoch-time" in result.output

def test_events_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["events", "--help"])
    assert result.exit_code == 0
    assert "line-crossings" in result.output
    assert "polygon-entry-exit" in result.output
    assert "encounters" in result.output

def test_events_encounters_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["events", "encounters", "--help"])
    assert result.exit_code == 0
    assert "--max-distance" in result.output
    assert "--time-bin-minutes" in result.output
    assert "--merge-gap-minutes" in result.output
    assert "--metric-crs" in result.output
    assert "--event-file" in result.output

def test_behavior_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["behavior", "--help"])
    assert result.exit_code == 0
    assert "extract-windows" in result.output
    assert "compute-knn" in result.output
    assert "umap" in result.output
    assert "align" in result.output

def test_behavior_extract_windows_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["behavior", "extract-windows", "--help"])
    assert result.exit_code == 0
    assert "--cluster-type" in result.output
    assert "--vessel-id-col" in result.output
    assert "--time-col" in result.output

def test_behavior_umap_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["behavior", "umap", "--help"])
    assert result.exit_code == 0
    assert "--cluster-method" in result.output
    assert "--fit-sample-size" in result.output
    assert "--compatibility-layout" in result.output
    assert "--contamination" in result.output
    assert "--output-dir" in result.output


