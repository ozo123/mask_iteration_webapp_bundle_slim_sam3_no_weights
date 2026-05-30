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
    import_end = html.index("async function importRunCopyFolderViaBatch", import_start)
    import_body = html[import_start:import_end]

    assert "importRunCopyFolderViaBatch(runCopyFiles, selection)" in import_body
    assert "importRunCopyFolderViaServerChunks" not in html
    assert "compat mode" not in import_body
    assert "fallback" not in import_body.lower()
    assert 'id="runCopyFolderUpload" type="file" webkitdirectory directory multiple' in html
    run_copy_input = html[html.index('id="runCopyFolderUpload"'):html.index(">", html.index('id="runCopyFolderUpload"'))]
    assert ".json" in run_copy_input
    assert ".png" in run_copy_input
    assert ".csv" not in run_copy_input


def test_run_copy_import_does_not_delete_selected_copy_folder():
    html = _html()

    import_start = html.index("async function importRunCopyFolderViaBatch")
    import_end = html.index("async function openCurrentTarget", import_start)
    import_body = html[import_start:import_end]
    read_loop_start = import_body.index("for (let index = 0; index < files.annotations.length; index += 1)")
    read_loop_body = import_body[
        read_loop_start:import_body.index("await flushPendingImports();", read_loop_start)
    ]

    assert "await flushPendingImports();" not in read_loop_body
    assert "requestPayload.replace_import_id = copyId" in import_body
    assert "requestPayload.reset_import_id = copyId" not in import_body
    assert "pendingImports.splice(0, IMPORT_BATCH_SIZE)" in import_body


def test_quality_marked_targets_stay_in_queue():
    html = _html()

    queue_start = html.index("function isQueueTarget")
    queue_end = html.index("function updateQueuedTargetStatus", queue_start)
    queue_body = html[queue_start:queue_end]

    assert 'status !== "delete"' in queue_body
    assert '"wrong"' not in queue_body
    assert '"difficult"' not in queue_body
    assert "updateQueuedTargetStatus(state.session.session_id, state.session.target_status)" in html


def test_text_prompt_has_current_target_status_button():
    html = _html()

    prompt_start = html.index('id="textPromptInput"')
    prompt_section = html[prompt_start:html.index('<div class="subtle">', prompt_start)]

    assert 'id="targetStatusBtn"' in prompt_section
    assert "function renderTargetStatusButton" in html
    assert "错误框" in html
    assert "难处理" in html
    assert "正常" in html
    assert ".target-status-button.status-normal" in html
    assert ".target-status-button.status-difficult" in html
    assert ".target-status-button.status-wrong" in html


def test_encoded_quality_mask_colors_do_not_affect_iteration_masks():
    html = _html()

    color_start = html.index("function resolveCurrentMaskColor")
    color_body = html[color_start:html.index("function renderMaskSourceStatus", color_start)]

    assert "difficultMask: [255, 211, 64]" in html
    assert "errorMask: [255, 106, 89]" in html
    assert 'status === "difficult"' in color_body
    assert "return COLORS.difficultMask" in color_body
    assert 'status === "delete" || status === "wrong"' in color_body
    assert "return COLORS.errorMask" in color_body
    assert "if (!isEncodedCurrentMask)" in color_body
    assert "return COLORS.currentMask" in color_body


def test_prompt_record_hover_highlights_canvas_items():
    html = _html()

    assert "function wirePromptRecordHover" in html
    assert "data-hover-kind" in html
    assert "state.hoverPrompt" in html
    assert "function isHoveredPrompt" in html
    assert "drawPromptLabel" in html


def test_locked_region_payload_uses_current_label():
    html = _html()

    save_start = html.index("async function saveLockedRegion")
    save_body = html[save_start:html.index("function removeTargetsAndStayNearCurrent", save_start)]

    assert "/lock-region" in save_body
    assert "label: state.currentLabel" in save_body


def test_prompt_visibility_toggles_are_split_from_locked_regions():
    html = _html()

    render_start = html.index("function renderCanvas")
    render_body = html[render_start:html.index("function renderAll", render_start)]
    show_points_start = render_body.index("if (state.showPoints)")
    show_points_body = render_body[show_points_start:render_body.index("}", show_points_start)]
    show_locked_start = render_body.index("if (state.showLockedRegions)")
    show_locked_body = render_body[show_locked_start:render_body.index("}", show_locked_start)]

    assert "drawPoints" in show_points_body
    assert "drawLineStrokes" in show_points_body
    assert "drawLockedRegions" not in show_points_body
    assert "drawLockedRegions" in show_locked_body
    assert 'id="togglePointsBtn"' in html
    assert 'id="toggleMaskBtn"' in html
    assert 'id="toggleLockedRegionsBtn"' in html


def test_mask_display_uses_non_lock_history_while_locked_regions_are_visible():
    html = _html()

    assert "function getDisplayMaskHistory" in html
    display_start = html.index("function getDisplayMaskHistory")
    display_body = html[display_start:html.index("function getCompareHistory", display_start)]
    assert "state.session.locked_regions" in display_body
    assert "!isLockedRegionHistory(item)" in display_body


def test_save_buttons_send_explicit_mask_modes_and_track_saved_state():
    html = _html()

    assert 'id="saveLockedOnlyBtn"' in html
    assert 'id="saveCurrentMaskBtn"' in html
    assert '"locked_only"' in html
    assert '"locked_union_mask"' in html
    assert "save-pending" in html
    assert "save-saved" in html


def test_locked_regions_disable_iteration_button():
    html = _html()

    buttons_start = html.index("function renderButtons")
    buttons_body = html[buttons_start:html.index("function toggleModeButtons", buttons_start)]

    assert "hasLockedRegions" in buttons_body
    assert "el.iterateBtn.disabled = !hasSession || state.busy || hasLockedRegions" in buttons_body
    assert "Manual mode is active while any locked region exists" in html


def test_locked_region_list_renders_foreground_background_labels():
    html = _html()

    list_start = html.index("function renderLockedRegionList")
    list_body = html[list_start:html.index("function renderHistoryList", list_start)]

    assert "Number(region.label) === 0 ? \"BG\" : \"FG\"" in list_body
    assert "Number(region.label) === 0 ? \"bg\" : \"fg\"" in list_body


def test_keyboard_shortcut_labels_are_visible_on_buttons():
    html = _html()

    button_expectations = {
        'id="prevTargetBtn"': "(A)",
        'id="nextTargetBtn"': "(D)",
        'id="iterateBtn"': "(Space)",
        'id="saveCurrentMaskBtn"': "(Enter)",
        'id="lockRegionToolBtn"': "(F)",
        'id="undoRegionPointBtn"': "(E)",
        'id="fgModeBtn"': "(1)",
        'id="bgModeBtn"': "(2)",
    }
    for marker, shortcut in button_expectations.items():
        start = html.index(marker)
        button_html = html[start:html.index("</button>", start)]
        assert shortcut in button_html


def test_mask_editing_toolbar_keeps_controls_accessible_on_small_viewports():
    html = _html()

    toolbar_start = html.index(".canvas-toolbar {")
    toolbar_block = html[toolbar_start:html.index("}", toolbar_start)]
    toolbar_button_start = html.index(".toolbar-grid button,")
    toolbar_button_block = html[toolbar_button_start:html.index("}", toolbar_button_start)]
    narrow_media_start = html.index("@media (max-width: 860px)")
    narrow_media_block = html[narrow_media_start:html.index("@media (max-height: 760px)", narrow_media_start)]
    mobile_media_start = html.index("@media (max-width: 680px)")
    mobile_media_block = html[mobile_media_start:html.index("</style>", mobile_media_start)]

    assert "overflow-y: auto" in toolbar_block
    assert "overflow-x: hidden" in toolbar_block
    assert "max-height: 100%" in toolbar_block
    assert "white-space: normal" in toolbar_button_block
    assert "overflow-wrap: anywhere" in toolbar_button_block
    assert ".toolbar-grid.two-col { grid-template-columns: 1fr; }" in narrow_media_block
    assert "grid-template-columns: 1fr" in mobile_media_block
    assert "max-height: 42vh" in mobile_media_block


def test_global_shortcuts_map_requested_keys_without_text_input_hijack():
    html = _html()

    handler_start = html.index("function handleGlobalKeyDown")
    handler_body = html[handler_start:html.index("function bindEvents", handler_start)]

    assert "shouldIgnoreGlobalShortcut(event)" in handler_body
    assert 'tagName === "input"' in html
    assert 'tagName === "textarea"' in html
    assert 'tagName === "select"' in html
    assert 'key === "1"' in handler_body
    assert 'key === "2"' in handler_body
    assert 'key === "a"' in handler_body
    assert 'key === "d"' in handler_body
    assert 'key === "f"' in handler_body
    assert 'key === "e"' in handler_body
    assert 'event.key === " " || event.key === "Spacebar"' in handler_body
    assert 'event.key === "Enter" || event.key === "NumpadEnter"' in handler_body
    assert "void handleIterate()" in handler_body
    assert "void handleSaveCurrentMask()" in handler_body
    assert "undoDrawingRegionPoint()" in handler_body
    assert "completeDrawingRegion" not in handler_body


def test_locked_region_draft_has_point_undo_button():
    html = _html()

    list_start = html.index("function renderLockedRegionList")
    list_body = html[list_start:html.index("function renderHistoryList", list_start)]

    assert 'id="undoRegionPointBtn"' in html
    assert "function undoDrawingRegionPoint" in html
    assert "data-region-undo" in list_body
    assert "el.undoRegionPointBtn.disabled = !hasSession || state.busy || !hasRegionDraft" in html


def test_locked_region_vertices_are_editable_after_commit():
    html = _html()

    assert "hoverLockedRegionPoint" in html
    assert "function getLockedRegionVertexAtEvent" in html
    assert "function beginLockedRegionPointDrag" in html
    assert "function finishLockedRegionPointDrag" in html
    assert "function deleteLockedRegionPoint" in html
    assert "/lock-region/update" in html
    assert 'state.drag.mode === "locked-region-point"' in html


def test_locked_region_editing_clamps_points_to_image_bounds():
    html = _html()

    client_start = html.index("function clientToImageXY")
    client_body = html[client_start:html.index("async function addPointAtEvent", client_start)]

    assert "allowOutside" in client_body
    assert "Number(target.image_width) - 1" in client_body
    assert "Number(target.image_height) - 1" in client_body
    assert "clientToImageXY(event, { allowOutside: true })" in html
    assert "function clampImagePoint" in html


def test_canvas_uses_crosshair_cursor_for_precise_clicking():
    html = _html()

    canvas_start = html.index("canvas {")
    canvas_block = html[canvas_start:html.index("}", canvas_start)]
    hover_start = html.index("canvas.locked-point-hover")
    hover_rule = html[hover_start:html.index("}", hover_start)]

    assert "cursor: crosshair" in canvas_block
    assert "cursor: crosshair" in hover_rule
    assert "canvas.dragging { cursor: grabbing; }" in html
