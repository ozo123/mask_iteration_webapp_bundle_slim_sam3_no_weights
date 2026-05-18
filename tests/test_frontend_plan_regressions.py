from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "web" / "mask_iteration_app" / "index_merged.html"


def _html() -> str:
    return FRONTEND.read_text(encoding="utf-8")


def test_special_toolbar_only_exposes_required_quality_actions():
    html = _html()

    assert 'id="markWrongTargetBtn"' in html
    assert 'id="markDifficultBtn"' in html
    assert 'id="markBlurryImageBtn"' in html
    assert "标记错误框" in html
    assert "标记难处理" in html
    assert "删除模糊图" in html
    assert 'id="deleteTargetBtn"' not in html
    assert 'id="deleteImageBtn"' not in html
    assert "删除当前框" not in html
    assert "删除整张图" not in html


def test_run_copy_import_uploads_selected_folder_instead_of_reading_runs_copy():
    html = _html()

    import_start = html.index("async function importRunCopyFolder()")
    import_end = html.index("async function importRunCopyFolderViaServerChunks", import_start)
    import_body = html[import_start:import_end]

    assert "importRunCopyFolderViaBatch(runCopyFiles, selection)" in import_body
    assert "importRunCopyFolderViaServerChunks(selection)" not in import_body
    assert "compat mode" not in import_body
    assert "fallback" not in import_body.lower()
