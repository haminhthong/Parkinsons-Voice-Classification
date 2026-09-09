from parkinson_voice.report import create_portfolio_figures


def test_portfolio_figures_can_be_recreated(tmp_path):
    paths = create_portfolio_figures(output_dir=tmp_path)
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
