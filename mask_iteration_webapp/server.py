from __future__ import annotations

import json
import mimetypes
import traceback
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def create_handler(service, static_dir: Path):
    class MaskIterationRequestHandler(BaseHTTPRequestHandler):
        server_version = "MaskIterationWebApp/0.1"

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/" or path == "/index.html":
                    return self._serve_static("index.html")
                if path in {"/merged.html", "/index_merged.html"}:
                    return self._serve_static("index_merged.html")
                if path.startswith("/static/"):
                    relpath = path.removeprefix("/static/")
                    return self._serve_static(relpath)
                if path == "/api/bootstrap":
                    return self._send_json(HTTPStatus.OK, service.bootstrap_payload())
                if path.startswith("/api/targets/") and path.endswith("/image"):
                    target_key = unquote(path[len("/api/targets/") : -len("/image")]).strip("/")
                    return self._serve_image(target_key)
                if path.startswith("/api/sessions/") and path.endswith("/validate-tools/artifact"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/validate-tools/artifact")]).strip("/")
                    query = parse_qs(parsed.query)
                    relpath = str((query.get("relpath") or [""])[0]).strip()
                    return self._serve_validate_artifact(target_key, relpath)
                if path.startswith("/api/sessions/"):
                    target_key = unquote(path[len("/api/sessions/") :]).strip("/")
                    payload = service.get_existing_session_payload(target_key)
                    if payload is None:
                        return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Session not found"})
                    return self._send_json(HTTPStatus.OK, payload)
                return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown route: {path}"})
            except Exception as error:  # pragma: no cover
                self._send_error_response(error)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                payload = self._read_json_body()

                if path == "/api/import-targets":
                    return self._send_json(
                        HTTPStatus.OK,
                        service.import_targets(
                            image_file_name=str(payload.get("image_file_name", "")),
                            image_data_url=str(payload.get("image_data_url", "")),
                            annotation_file_name=str(payload.get("annotation_file_name", "")),
                            annotation_text=str(payload.get("annotation_text", "")),
                            import_session_id=payload.get("import_session_id"),
                            import_session_label=payload.get("import_session_label"),
                            image_set_id=payload.get("image_set_id"),
                            image_set_label=payload.get("image_set_label"),
                            annotation_state_id=payload.get("annotation_state_id"),
                            annotation_state_label=payload.get("annotation_state_label"),
                            image_relative_path=payload.get("image_relative_path"),
                            annotation_relative_path=payload.get("annotation_relative_path"),
                        ),
                    )

                if path == "/api/import-targets/batch":
                    return self._send_json(
                        HTTPStatus.OK,
                        service.import_targets_batch(payload.get("items")),
                    )

                if path == "/api/open-session":
                    target_key = str(payload.get("target_key", "")).strip()
                    return self._send_json(HTTPStatus.OK, service.open_session(target_key))

                if path == "/api/work-dataset/export":
                    return self._send_json(
                        HTTPStatus.OK,
                        service.export_work_dataset_copy(
                            image_set_id=str(payload.get("image_set_id", "")),
                            annotation_state_id=str(payload.get("annotation_state_id", "")),
                            export_name=payload.get("export_name"),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/points"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/points")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.add_point(
                            target_key=target_key,
                            x=float(payload["x"]),
                            y=float(payload["y"]),
                            label=int(payload["label"]),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/points/delete"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/points/delete")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.delete_point(target_key=target_key, point_id=str(payload["point_id"])),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/undo"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/undo")]).strip("/")
                    return self._send_json(HTTPStatus.OK, service.undo_point(target_key))

                if path.startswith("/api/sessions/") and path.endswith("/clear-points"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/clear-points")]).strip("/")
                    return self._send_json(HTTPStatus.OK, service.clear_points(target_key))

                if path.startswith("/api/sessions/") and path.endswith("/prompt-state"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/prompt-state")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.update_prompt_state(
                            target_key=target_key,
                            text_prompt=payload.get("text_prompt"),
                            line_strokes=payload.get("line_strokes"),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/lock-region"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/lock-region")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.lock_region(
                            target_key=target_key,
                            points=payload.get("points"),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/lock-region/delete"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/lock-region/delete")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.delete_locked_region(
                            target_key=target_key,
                            region_id=str(payload["region_id"]),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/delete-target"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/delete-target")]).strip("/")
                    return self._send_json(HTTPStatus.OK, service.delete_target(target_key))

                if path.startswith("/api/sessions/") and path.endswith("/delete-image"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/delete-image")]).strip("/")
                    return self._send_json(HTTPStatus.OK, service.delete_image(target_key))

                if path.startswith("/api/sessions/") and path.endswith("/mark-difficult"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/mark-difficult")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.mark_difficult_target(target_key, reason=payload.get("reason")),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/mark-blurry-image"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/mark-blurry-image")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.mark_blurry_image(target_key, reason=payload.get("reason")),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/validate-tools/validate"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/validate-tools/validate")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.run_validate_tools_validate(
                            target_key=target_key,
                            api_key=payload.get("api_key"),
                            base_url=payload.get("base_url"),
                            model=payload.get("model"),
                            strict_mode=bool(payload.get("strict_mode", False)),
                            review_mode=payload.get("review_mode"),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/validate-tools/visualize"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/validate-tools/visualize")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.run_validate_tools_visualize(
                            target_key=target_key,
                            use_latest_validation=bool(payload.get("use_latest_validation", True)),
                            review_mode=payload.get("review_mode"),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/validate-tools/full"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/validate-tools/full")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.run_validate_tools_full(
                            target_key=target_key,
                            api_key=payload.get("api_key"),
                            base_url=payload.get("base_url"),
                            model=payload.get("model"),
                            strict_mode=bool(payload.get("strict_mode", False)),
                            review_mode=payload.get("review_mode"),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/iterate"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/iterate")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.iterate(
                            target_key,
                            text_prompt=payload.get("text_prompt"),
                            line_strokes=payload.get("line_strokes"),
                        ),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/rollback"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/rollback")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.rollback(target_key=target_key, history_id=str(payload["history_id"])),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/history/delete"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/history/delete")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.delete_history_items(target_key=target_key, history_ids=payload.get("history_ids")),
                    )

                if path.startswith("/api/sessions/") and path.endswith("/metadata"):
                    target_key = unquote(path[len("/api/sessions/") : -len("/metadata")]).strip("/")
                    return self._send_json(
                        HTTPStatus.OK,
                        service.update_metadata(
                            target_key=target_key,
                            variant_name=payload.get("variant_name"),
                            notes=payload.get("notes"),
                        ),
                    )

                return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown route: {path}"})
            except Exception as error:  # pragma: no cover
                self._send_error_response(error)

        def log_message(self, format: str, *args) -> None:
            return

        def _read_json_body(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            if content_length <= 0:
                return {}
            body = self.rfile.read(content_length)
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))

        def _send_json(self, status: HTTPStatus, payload: dict, extra_headers: dict[str, str] | None = None) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, relative_path: str) -> None:
            candidate = (static_dir / relative_path).resolve()
            static_root = static_dir.resolve()
            if static_root not in candidate.parents and candidate != static_root:
                return self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "Forbidden"})
            if not candidate.exists() or not candidate.is_file():
                return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Static file not found"})
            mime_type, _ = mimetypes.guess_type(candidate.name)
            body = candidate.read_bytes()
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_image(self, target_key: str) -> None:
            image_path = service.get_target_image_path(target_key)
            if not image_path.exists():
                return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Image not found"})
            mime_type, _ = mimetypes.guess_type(image_path.name)
            body = image_path.read_bytes()
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_validate_artifact(self, target_key: str, relpath: str) -> None:
            artifact_path = service.get_validate_artifact_path(target_key, relpath)
            mime_type, _ = mimetypes.guess_type(artifact_path.name)
            body = artifact_path.read_bytes()
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_error_response(self, error: Exception) -> None:
            traceback.print_exc()
            payload = {
                "ok": False,
                "error": str(error),
                "error_type": error.__class__.__name__,
            }
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, payload)

    return MaskIterationRequestHandler


def run_server(service, host: str, port: int, static_dir: Path) -> None:
    handler_cls = create_handler(service, static_dir)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    print(f"[mask-iteration-webapp] serving on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[mask-iteration-webapp] stopped.")
    finally:
        httpd.server_close()
