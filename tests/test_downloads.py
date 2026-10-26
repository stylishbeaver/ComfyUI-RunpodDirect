"""Exercise queued downloads against a local server, without ComfyUI or a GPU.

Run with: python -m unittest discover -s tests -v (requires aiohttp).
"""

import asyncio
import hashlib
import importlib.util
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = bytes(range(256)) * 64


class LocalResolver(aiohttp.abc.AbstractResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        return [{"hostname": host, "host": "127.0.0.1", "port": port,
                 "family": socket.AF_INET, "proto": 0, "flags": 0}]

    async def close(self):
        pass


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "models" / "checkpoints"
        self.cache_dir = self.root / "cache"
        self.events = []
        self.requests = []
        self.range_support = True
        self.reject_auth = False
        self.hold_download = False
        self.download_started = asyncio.Event()
        self.release_download = asyncio.Event()

        async def send(event, data):
            self.events.append((event, data))

        self.server_instance = types.SimpleNamespace(routes=web.RouteTableDef(), send=send, loop=None)
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=self.server_instance)
        folders = types.ModuleType("folder_paths")
        folders.folder_names_and_paths = {"checkpoints": ([str(self.model_dir)], {".safetensors"})}
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy_mm = types.ModuleType("comfy.model_management")
        self.modules = patch.dict(sys.modules, {
            "server": server, "folder_paths": folders,
            "comfy": comfy, "comfy.model_management": comfy_mm,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {
            "RPD_DISABLE_CGROUP_RAM_PATCH": "1",
            "RUNPODDIRECT_CACHE_DIR": str(self.cache_dir),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        spec = importlib.util.spec_from_file_location("runpoddirect_test", ROOT / "__init__.py")
        self.plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.plugin)
        self.plugin.CHUNK_SIZE = 1024

        source = web.Application()
        source.router.add_route("*", "/{path:.*}", self.serve_model)
        self.source = TestServer(source)
        await self.source.start_server()
        self.addAsyncCleanup(self.source.close)
        app = web.Application()
        app.add_routes(self.server_instance.routes)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def asyncTearDown(self):
        self.release_download.set()
        task = self.plugin.current_download_task
        if task is not None:
            await asyncio.wait_for(task, 3)
        await asyncio.sleep(0)

    async def serve_model(self, request):
        self.requests.append((request.method, dict(request.headers)))
        if self.reject_auth:
            return web.Response(status=401)
        if request.method == "GET" and self.hold_download:
            self.download_started.set()
            await self.release_download.wait()
        headers = {"Accept-Ranges": "bytes"} if self.range_support else {}
        requested_range = request.headers.get("Range")
        if requested_range and self.range_support:
            start, end = map(int, requested_range.removeprefix("bytes=").split("-"))
            headers["Content-Range"] = f"bytes {start}-{end}/{len(PAYLOAD)}"
            return web.Response(body=PAYLOAD[start:end + 1], headers=headers, status=206)
        return web.Response(body=PAYLOAD, headers=headers)

    async def queue(self, name="model.safetensors", **extra):
        data = {"url": str(self.source.make_url("/model")),
                "save_path": "checkpoints", "filename": name, **extra}
        return await self.client.post("/server_download/start", json=data)

    async def outcome(self, name="model.safetensors"):
        download_id = "checkpoints/" + name
        for _ in range(300):
            state = self.plugin.active_downloads[download_id]
            if state["status"] in ("completed", "error", "cancelled"):
                task = self.plugin.current_download_task
                if task is not None:
                    await task
                return state
            await asyncio.sleep(0.01)
        self.fail("Download did not finish")

    def assert_completed_model(self):
        model = self.model_dir / "model.safetensors"
        cached = self.cache_dir / "checkpoints" / model.name
        self.assertTrue(model.is_symlink())
        self.assertEqual(model.resolve(), cached)
        self.assertEqual(cached.read_bytes(), PAYLOAD)
        self.assertFalse(Path(str(cached) + ".runpoddirect.part").exists())
        complete = [data for event, data in self.events if event == "server_download_complete"]
        self.assertEqual(complete[0]["path"], str(model))

    async def test_ranged_download_validates_then_links_cached_model(self):
        response = await self.queue(hash=hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual(response.status, 200)
        state = await self.outcome()
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["hash_verified"])
        self.assertEqual(len([r for r in self.requests if r[0] == "GET"]), 8)
        self.assert_completed_model()

    async def test_single_connection_download_uses_same_cache_and_link(self):
        self.range_support = False
        await self.queue()
        self.assertEqual((await self.outcome())["status"], "completed")
        self.assert_completed_model()

    async def test_bad_hash_never_exposes_model_and_removes_partial(self):
        await self.queue(hash="0" * 64)
        state = await self.outcome()
        self.assertEqual(state["status"], "error")
        self.assertIn("mismatch", state["error"])
        self.assertFalse(os.path.lexists(self.model_dir / "model.safetensors"))
        self.assertEqual(list(self.cache_dir.rglob("*.part")), [])
        self.assertEqual(list(self.cache_dir.rglob("*.safetensors")), [])

    async def test_broken_model_symlink_is_replaced_after_download(self):
        self.model_dir.mkdir(parents=True)
        (self.model_dir / "model.safetensors").symlink_to(self.root / "missing")
        await self.queue()
        self.assertEqual((await self.outcome())["status"], "completed")
        self.assert_completed_model()

    async def test_corrupt_cached_model_can_be_redownloaded(self):
        self.model_dir.mkdir(parents=True)
        cached = self.cache_dir / "checkpoints" / "model.safetensors"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"corrupt")
        (self.model_dir / cached.name).symlink_to(cached)
        await self.queue(hash=hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual((await self.outcome())["status"], "completed")
        self.assert_completed_model()

    async def test_existing_model_is_preserved(self):
        self.model_dir.mkdir(parents=True)
        model = self.model_dir / "model.safetensors"
        model.write_bytes(b"existing")
        self.assertEqual((await self.queue()).status, 400)
        self.assertEqual(model.read_bytes(), b"existing")
        self.assertEqual(self.requests, [])

    async def test_duplicate_request_does_not_remove_active_partial(self):
        self.hold_download = True
        await self.queue()
        await asyncio.wait_for(self.download_started.wait(), 3)
        partial = self.cache_dir / "checkpoints" / "model.safetensors.runpoddirect.part"
        self.assertTrue(partial.exists())
        self.assertEqual((await self.queue()).status, 409)
        self.assertTrue(partial.exists())
        self.release_download.set()
        self.assertEqual((await self.outcome())["status"], "completed")
        self.assert_completed_model()

    async def test_cancelled_download_never_exposes_model(self):
        self.hold_download = True
        await self.queue()
        await asyncio.wait_for(self.download_started.wait(), 3)
        response = await self.client.post("/server_download/cancel", json={"download_id": "checkpoints/model.safetensors"})
        self.assertEqual(response.status, 200)
        self.release_download.set()
        self.assertEqual((await self.outcome())["status"], "cancelled")
        self.assertFalse(os.path.lexists(self.model_dir / "model.safetensors"))
        self.assertEqual(list(self.cache_dir.rglob("*.part")), [])

    async def test_hf_auth_reaches_head_and_every_range_request(self):
        original_session = aiohttp.ClientSession

        def session(**kwargs):
            return original_session(connector=aiohttp.TCPConnector(resolver=LocalResolver()), **kwargs)

        url = f"http://huggingface.co:{self.source.port}/org/repo/resolve/main/model"
        with patch.object(aiohttp, "ClientSession", side_effect=session):
            await self.queue(url=url, token="test-token")
            self.assertEqual((await self.outcome())["status"], "completed")
        self.assertGreater(len(self.requests), 1)
        for _, headers in self.requests:
            self.assertEqual(headers.get("Authorization"), "Bearer test-token")

    async def test_hf_access_error_is_reported_without_retry(self):
        self.reject_auth = True
        original_session = aiohttp.ClientSession

        def session(**kwargs):
            return original_session(connector=aiohttp.TCPConnector(resolver=LocalResolver()), **kwargs)

        url = f"http://huggingface.co:{self.source.port}/org/repo/resolve/main/model"
        with patch.object(aiohttp, "ClientSession", side_effect=session):
            await self.queue(url=url)
            state = await self.outcome()
        self.assertEqual(state["status"], "error")
        self.assertIn("Authentication required", state["error"])
        self.assertEqual(len(self.requests), 1)

    async def test_non_hf_download_does_not_receive_hf_token(self):
        await self.queue(token="test-token")
        self.assertEqual((await self.outcome())["status"], "completed")
        self.assertTrue(self.requests)
        self.assertTrue(all("Authorization" not in headers for _, headers in self.requests))

    async def test_hf_token_aliases_are_visible_to_frontend(self):
        for name in self.plugin.HF_TOKEN_ENV_VARS:
            with self.subTest(variable=name), patch.dict(os.environ, {name: "test-token"}, clear=True):
                self.assertEqual(self.plugin._resolve_hf_token(), "test-token")
                response = await self.client.get("/server_download/hf_token_status")
                self.assertEqual(await response.json(), {"has_token": True})

    async def test_hf_auth_is_limited_to_hf_hosts(self):
        for url in ("https://huggingface.co/model", "https://cdn.huggingface.co/model", "https://hf.co/model"):
            self.assertTrue(self.plugin._is_hf_url(url))
        for url in ("https://nothuggingface.co/model", "https://huggingface.co.example.com/model", "https://example.com/huggingface.co/model"):
            self.assertFalse(self.plugin._is_hf_url(url))


if __name__ == "__main__":
    unittest.main()
