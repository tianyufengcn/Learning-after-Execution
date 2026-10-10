from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageOps

from .tikz import extract_tikz_document


@dataclass
class RenderResult:
    ok: bool
    png_path: str | None
    code_hash: str
    cache_key: str | None = None
    compiler: str | None = None
    elapsed_s: float = 0.0
    compile_s: float = 0.0
    error: str | None = None
    cache_hit: bool = False


def _trim_white(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    bg = Image.new("RGB", image.size, "white")
    bbox = ImageChops.difference(image, bg).getbbox()
    return image.crop(bbox) if bbox else image


class TikZRenderer:
    """Compile TikZ/LaTeX completions and rasterize the first PDF page.

    The renderer is a verifier, never a semantic code fixer. It provides:

    - per-rollout timeout isolation;
    - content-addressed success *and stable-failure* caching;
    - in-process duplicate suppression for identical concurrent rollouts;
    - atomic cache writes so concurrent worker threads do not leave partial files.
    """

    _CACHE_FORMAT_VERSION = 2

    def __init__(
        self,
        cache_dir: str,
        tmp_dir: str,
        texlive_bin: str | None = None,
        pdftoppm_bin: str | None = None,
        compilers: list[str] | None = None,
        compile_timeout_s: int = 30,
        image_size: int = 448,
        dpi: int = 150,
        keep_failed_logs: bool = True,
        cache_failures: bool = True,
        cache_transient_failures: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.tmp_dir = Path(tmp_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir = self.cache_dir / "failures"
        if keep_failed_logs:
            self.failure_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed_logs = bool(keep_failed_logs)
        self.cache_failures = bool(cache_failures)
        self.cache_transient_failures = bool(cache_transient_failures)
        self.compilers = compilers or ["pdflatex", "lualatex", "xelatex"]
        self.compile_timeout_s = int(compile_timeout_s)
        self.image_size = int(image_size)
        self.dpi = int(dpi)
        self.env = os.environ.copy()
        if texlive_bin:
            self.env["PATH"] = str(texlive_bin) + os.pathsep + self.env.get("PATH", "")
        self.pdftoppm = pdftoppm_bin or shutil.which("pdftoppm", path=self.env.get("PATH"))
        if not self.pdftoppm:
            raise FileNotFoundError("pdftoppm not found. Install poppler or set renderer.pdftoppm_bin")

        # Identical completions are common in grouped sampling. Ensure only one
        # thread compiles a given cache key in this process.
        self._locks_guard = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    @staticmethod
    def hash_code(code: str) -> str:
        return hashlib.sha1(code.encode("utf-8", errors="replace")).hexdigest()

    def _protocol_signature(self) -> str:
        payload = {
            "v": self._CACHE_FORMAT_VERSION,
            "image_size": self.image_size,
            "dpi": self.dpi,
            "compilers": self.compilers,
            "shell_escape": False,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _cache_key(self, doc: str) -> str:
        material = self._protocol_signature() + "\0" + doc
        return hashlib.sha1(material.encode("utf-8", errors="replace")).hexdigest()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _meta_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _png_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.png"

    def _atomic_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _cached(self, cache_key: str, code_hash: str) -> RenderResult | None:
        meta = self._meta_path(cache_key)
        if not meta.exists():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return None
        status = data.get("status")
        if status == "success":
            png = self._png_path(cache_key)
            if not png.exists() or png.stat().st_size <= 0:
                return None
            return RenderResult(
                True,
                str(png),
                code_hash,
                cache_key=cache_key,
                compiler=data.get("compiler"),
                elapsed_s=float(data.get("elapsed_s", 0.0)),
                cache_hit=True,
            )
        if status == "failed" and self.cache_failures:
            return RenderResult(
                False,
                None,
                code_hash,
                cache_key=cache_key,
                compiler=data.get("compiler"),
                elapsed_s=float(data.get("elapsed_s", 0.0)),
                error=data.get("error", "cached_failure"),
                cache_hit=True,
            )
        return None

    def _failure_is_cacheable(self, error: str) -> bool:
        if not self.cache_failures:
            return False
        stable = {"no_tikz_document", "compile_failed", "blank_render"}
        return self.cache_transient_failures or error in stable

    def _write_failure_meta(
        self,
        *,
        cache_key: str,
        code_hash: str,
        error: str,
        elapsed_s: float,
        compiler: str | None = None,
    ) -> None:
        if not self._failure_is_cacheable(error):
            return
        self._atomic_json(
            self._meta_path(cache_key),
            {
                "status": "failed",
                "error": error,
                "compiler": compiler,
                "elapsed_s": elapsed_s,
                "code_hash": code_hash,
                "protocol": self._protocol_signature(),
            },
        )

    def render(self, completion: str) -> RenderResult:
        start = time.monotonic()
        doc = extract_tikz_document(completion, wrap_tikzpicture=False)
        if not doc:
            code_hash = self.hash_code(completion)
            cache_key = self._cache_key(completion)
            cached = self._cached(cache_key, code_hash)
            if cached:
                return cached
            elapsed = time.monotonic() - start
            self._write_failure_meta(
                cache_key=cache_key,
                code_hash=code_hash,
                error="no_tikz_document",
                elapsed_s=elapsed,
            )
            return RenderResult(False, None, code_hash, cache_key=cache_key, elapsed_s=elapsed, error="no_tikz_document")

        code_hash = self.hash_code(doc)
        cache_key = self._cache_key(doc)
        cached = self._cached(cache_key, code_hash)
        if cached:
            return cached

        lock = self._lock_for(cache_key)
        with lock:
            cached = self._cached(cache_key, code_hash)
            if cached:
                return cached
            return self._render_uncached(doc, code_hash, cache_key, start)

    def _render_uncached(self, doc: str, code_hash: str, cache_key: str, start: float) -> RenderResult:
        with tempfile.TemporaryDirectory(prefix=f"i2t_{cache_key[:10]}_", dir=self.tmp_dir) as td:
            work = Path(td)
            tex = work / "figure.tex"
            tex.write_text(doc, encoding="utf-8")
            compiler_used = None
            last_log = ""
            pdf = work / "figure.pdf"
            compile_end = None
            for compiler in self.compilers:
                exe = shutil.which(compiler, path=self.env.get("PATH"))
                if not exe:
                    continue
                try:
                    proc = subprocess.run(
                        [
                            exe,
                            "-no-shell-escape",
                            "-interaction=nonstopmode",
                            "-halt-on-error",
                            "-file-line-error",
                            tex.name,
                        ],
                        cwd=work,
                        env=self.env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=self.compile_timeout_s,
                        check=False,
                    )
                    last_log = (proc.stdout or "")[-20000:]
                    if proc.returncode == 0 and pdf.exists() and pdf.stat().st_size > 0:
                        compiler_used = compiler
                        compile_end = time.monotonic()
                        break
                except subprocess.TimeoutExpired as exc:
                    last_log = f"timeout after {self.compile_timeout_s}s: {exc}"
                except Exception as exc:  # environment failures become reward failures, never trainer crashes
                    last_log = repr(exc)

            if compiler_used is None:
                elapsed = time.monotonic() - start
                if self.keep_failed_logs:
                    (self.failure_dir / f"{cache_key}.tex").write_text(doc, encoding="utf-8")
                    (self.failure_dir / f"{cache_key}.log").write_text(last_log, encoding="utf-8", errors="replace")
                self._write_failure_meta(
                    cache_key=cache_key,
                    code_hash=code_hash,
                    error="compile_failed",
                    elapsed_s=elapsed,
                )
                return RenderResult(False, None, code_hash, cache_key=cache_key, elapsed_s=elapsed, error="compile_failed")

            prefix = work / "rendered"
            try:
                proc = subprocess.run(
                    [self.pdftoppm, "-f", "1", "-singlefile", "-png", "-r", str(self.dpi), str(pdf), str(prefix)],
                    cwd=work,
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=self.compile_timeout_s,
                    check=False,
                )
                raw_png = work / "rendered.png"
                if proc.returncode != 0 or not raw_png.exists():
                    error = "pdf_to_png_failed"
                    elapsed = time.monotonic() - start
                    self._write_failure_meta(
                        cache_key=cache_key,
                        code_hash=code_hash,
                        error=error,
                        elapsed_s=elapsed,
                        compiler=compiler_used,
                    )
                    return RenderResult(False, None, code_hash, cache_key=cache_key, compiler=compiler_used, elapsed_s=elapsed, error=error)

                image = Image.open(raw_png).convert("RGB")
                image.load()
                image = _trim_white(image)
                if image.width < 2 or image.height < 2:
                    error = "blank_render"
                    elapsed = time.monotonic() - start
                    self._write_failure_meta(
                        cache_key=cache_key,
                        code_hash=code_hash,
                        error=error,
                        elapsed_s=elapsed,
                        compiler=compiler_used,
                    )
                    return RenderResult(False, None, code_hash, cache_key=cache_key, compiler=compiler_used, elapsed_s=elapsed, error=error)

                image = ImageOps.pad(
                    image,
                    (self.image_size, self.image_size),
                    color="white",
                    method=Image.Resampling.LANCZOS,
                )
                out_png = self._png_path(cache_key)
                tmp_png = out_png.with_name(f".{out_png.name}.{os.getpid()}.{threading.get_ident()}.tmp.png")
                image.save(tmp_png, "PNG")
                os.replace(tmp_png, out_png)
                elapsed = time.monotonic() - start
                self._atomic_json(
                    self._meta_path(cache_key),
                    {
                        "status": "success",
                        "compiler": compiler_used,
                        "elapsed_s": elapsed,
                        "code_hash": code_hash,
                        "protocol": self._protocol_signature(),
                    },
                )
                return RenderResult(
                    True,
                    str(out_png),
                    code_hash,
                    cache_key=cache_key,
                    compiler=compiler_used,
                    elapsed_s=elapsed,
                    compile_s=(compile_end - start) if compile_end is not None else elapsed,
                )
            except subprocess.TimeoutExpired:
                error = "raster_timeout"
                elapsed = time.monotonic() - start
                self._write_failure_meta(
                    cache_key=cache_key,
                    code_hash=code_hash,
                    error=error,
                    elapsed_s=elapsed,
                    compiler=compiler_used,
                )
                return RenderResult(False, None, code_hash, cache_key=cache_key, compiler=compiler_used, elapsed_s=elapsed, error=error)
            except Exception as exc:
                error = f"raster_error:{exc}"
                elapsed = time.monotonic() - start
                self._write_failure_meta(
                    cache_key=cache_key,
                    code_hash=code_hash,
                    error=error,
                    elapsed_s=elapsed,
                    compiler=compiler_used,
                )
                return RenderResult(False, None, code_hash, cache_key=cache_key, compiler=compiler_used, elapsed_s=elapsed, error=error)
