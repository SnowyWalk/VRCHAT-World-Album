import os, re, time, json, asyncio, pathlib
from typing import Dict, Any, Tuple, List
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# ===== 설정 =====
PORT = int(os.getenv("PORT", "8080"))
CACHE_TTL = int(os.getenv("CACHE_TTL", str(24 * 60 * 60)))  # 24시간
MANIFEST_TTL = int(os.getenv("MANIFEST_TTL", "60"))  # /api/worlds 캐시 TTL
USER_AGENT = os.getenv("USER_AGENT", "VRC-World-Site/1.3")
UPSTREAM_BASE = "https://api.vrchat.cloud/api/1/worlds/"
WORLD_ID_RE = re.compile(r"^wrld_[0-9a-f-]{8,}$", re.I)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# 썸네일 설정
THUMB_WIDTH = int(os.getenv("THUMB_WIDTH", "480"))
THUMB_QUALITY = int(os.getenv("THUMB_QUALITY", "82"))
USE_WEBP = os.getenv("THUMB_WEBP", "1") == "1"
VIEW_WIDTH = int(os.getenv("VIEW_WIDTH", "1600"))
VIEW_QUALITY = int(os.getenv("VIEW_QUALITY", "88"))

ROOT = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"
WORLDS_ROOT = STATIC_DIR / "worlds"
THUMBS_ROOT = STATIC_DIR / "_thumbs"
VIEWS_ROOT = STATIC_DIR / "_views"

# 캐시: worldId -> (expire_ts, data)
_cache: Dict[str, Tuple[float, Any]] = {}
_locks: Dict[str, asyncio.Lock] = {}
http_client: httpx.AsyncClient | None = None

# /api/worlds 결과 캐시
_manifest_cache = {"data": None, "etag": None, "exp": 0}
# txt 설명 캐시
_desc_cache: Dict[str, Tuple[float, str]] = {}

app = FastAPI(title="VRC World Site")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")

from starlette.middleware.base import BaseHTTPMiddleware
class CacheHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if path.startswith("/static/worlds/"):
            resp.headers.setdefault("Cache-Control", "public, max-age=1209600")
        return resp
app.add_middleware(CacheHeaderMiddleware)

@app.on_event("startup")
async def _startup():
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    WORLDS_ROOT.mkdir(parents=True, exist_ok=True)
    THUMBS_ROOT.mkdir(parents=True, exist_ok=True)
    VIEWS_ROOT.mkdir(parents=True, exist_ok=True)

@app.on_event("shutdown")
async def _shutdown():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None

# ---- 캐시/락 ----
def cache_get(key: str):
    v = _cache.get(key)
    if not v: return None
    exp, data = v
    if exp < time.time():
        return None
    return data

def cache_set(key: str, data: Any):
    _cache[key] = (time.time() + CACHE_TTL, data)

def lock_for(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock

# ---- 시간 파싱 ----
def iso_to_ts(s: str) -> float:
    if not s: return 0.0
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0

# ---- VRChat upstream ----
async def fetch_world_from_upstream(world_id: str):
    assert http_client is not None
    url = f"{UPSTREAM_BASE}{world_id}"
    r = await http_client.get(url)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail="upstream_error")
    return r.json()

async def get_world_with_cache(world_id: str):
    data = cache_get(world_id)
    if data is not None:
        return data
    lock = lock_for(world_id)
    async with lock:
        data2 = cache_get(world_id)
        if data2 is not None:
            return data2
        data3 = await fetch_world_from_upstream(world_id)
        cache_set(world_id, data3)
        return data3

# ---- 파일 시스템 ----
def list_world_dirs() -> List[pathlib.Path]:
    if not WORLDS_ROOT.exists(): return []
    return [p for p in WORLDS_ROOT.iterdir() if p.is_dir() and WORLD_ID_RE.match(p.name or "")]

def read_local_desc(world_dir: pathlib.Path) -> str:
    """폴더 내 txt 파일 1개를 읽어서 설명으로 사용. 없으면 ''. mtime 캐시."""
    try:
        txts = sorted([p for p in world_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
                      key=lambda p: p.name.lower())
        if not txts: return ""
        f = txts[0]
        mtime = f.stat().st_mtime
        key = str(f.resolve())
        cached = _desc_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        data = f.read_bytes()
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
            try:
                text = data.decode(enc).strip()
                _desc_cache[key] = (mtime, text)
                return text
            except Exception:
                continue
        text = data.decode("utf-8", "ignore").strip()
        _desc_cache[key] = (mtime, text)
        return text
    except Exception:
        return ""

def scan_world_dir(world_dir: pathlib.Path) -> dict:
    """한 번의 iterdir로 addedTs, 사진 목록, 로컬 txt 설명 수집."""
    info = {"addedTs": 0.0, "photos": [], "desc": "", "photo_count": 0}
    if not world_dir.exists(): return info

    latest_file_ts = 0.0
    image_files, txt_file = [], None
    try:
        for f in world_dir.iterdir():
            if not f.is_file(): continue
            ext = f.suffix.lower()
            latest_file_ts = max(latest_file_ts, float(f.stat().st_mtime))
            if ext == ".txt" and txt_file is None:
                txt_file = f
            elif ext in ALLOWED_EXT:
                image_files.append(f)
    except Exception:
        pass

    try:
        st = world_dir.stat()
        base = getattr(st, "st_birthtime", None)
        base = float(base) if base else float(st.st_mtime)
    except Exception:
        base = 0.0
    info["addedTs"] = max(base, latest_file_ts)

    wid = world_dir.name
    for p in sorted(image_files, key=lambda x: x.name.lower()):
        rel = p.relative_to(STATIC_DIR).as_posix()
        info["photos"].append({
            "full": "/static/" + rel,
            "thumb": f"/thumbs/{wid}/{p.name}",
            "view": f"/view/{wid}/{p.name}",
        })
    info["photo_count"] = len(info["photos"])
    info["desc"] = read_local_desc(world_dir) if txt_file else ""
    return info

# ---- 이미지 리사이즈 ----
def thumb_target_path(src: pathlib.Path) -> pathlib.Path:
    wid, stem = src.parent.name, src.stem
    ext = ".webp" if USE_WEBP else src.suffix.lower()
    dst_dir = THUMBS_ROOT / wid
    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir / f"{stem}{ext}"

def view_target_path(src: pathlib.Path) -> pathlib.Path:
    wid, stem = src.parent.name, src.stem
    ext = ".webp" if USE_WEBP else src.suffix.lower()
    dst_dir = VIEWS_ROOT / wid
    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir / f"{stem}{ext}"

def _ensure_resized(src: pathlib.Path, dst: pathlib.Path, width: int, quality: int, force_webp: bool):
    from PIL import Image, ImageOps
    try:
        if (not dst.exists()) or (src.stat().st_mtime > dst.stat().st_mtime):
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                w, h = im.size
                if w > width:
                    ratio = width / float(w)
                    im = im.resize((width, max(1, int(h * ratio))), Image.LANCZOS)
                params = {}
                if force_webp:
                    params = dict(format="WEBP", quality=quality, method=6)
                elif dst.suffix.lower() in (".jpg", ".jpeg"):
                    params = dict(format="JPEG", quality=quality, optimize=True)
                im.save(dst, **params)
    except Exception:
        return src
    return dst

@app.get("/thumbs/{world_id}/{filename:path}")
def get_thumb(world_id: str, filename: str):
    src = WORLDS_ROOT / world_id / filename
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404)
    if src.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400)
    dst = thumb_target_path(src)
    out = _ensure_resized(src, dst, THUMB_WIDTH, THUMB_QUALITY, USE_WEBP)
    media = "image/webp" if (USE_WEBP and src.suffix.lower() != ".webp") else None
    return FileResponse(out, media_type=media, headers={"Cache-Control":"public, max-age=604800"})

@app.get("/view/{world_id}/{filename:path}")
def get_view(world_id: str, filename: str):
    src = WORLDS_ROOT / world_id / filename
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404)
    if src.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400)
    dst = view_target_path(src)
    out = _ensure_resized(src, dst, VIEW_WIDTH, VIEW_QUALITY, USE_WEBP)
    media = "image/webp" if (USE_WEBP and src.suffix.lower() != ".webp") else None
    return FileResponse(out, media_type=media, headers={"Cache-Control":"public, max-age=1209600"})

# ---- API ----
@app.get("/healthz")
async def healthz():
    return {"ok": True, "cache_items": len(_cache), "ttl": CACHE_TTL}

@app.get("/api/world")
async def api_world(id: str = Query(...)):
    world_id = id.strip()
    if not WORLD_ID_RE.match(world_id):
        raise HTTPException(status_code=400)
    meta = await get_world_with_cache(world_id)
    world_dir = WORLDS_ROOT / world_id
    scan = scan_world_dir(world_dir)
    merged = {
        "id": world_id,
        "name": meta.get("name"),
        "description": scan["desc"],
        "authorName": meta.get("authorName"),
        "favorites": meta.get("favorites"),
        "visits": meta.get("visits"),
        "capacity": meta.get("recommendedCapacity") or meta.get("capacity"),
        "heat": meta.get("heat"),
        "popularity": meta.get("popularity"),
        "imageUrl": meta.get("imageUrl") or meta.get("thumbnailImageUrl"),
        "thumbnailImageUrl": meta.get("thumbnailImageUrl"),
        "updated_at": meta.get("updated_at"),
        "updatedTs": iso_to_ts(meta.get("updated_at") or ""),
        "publicationDate": meta.get("publicationDate"),
        "url": f"https://vrchat.com/home/world/{world_id}",
        "tags": meta.get("tags", []),
        "photos": scan["photos"],
        "addedTs": scan["addedTs"],
        "authorId": meta.get("authorId"),
    }
    return JSONResponse(content=merged, headers={"Cache-Control": f"public, max-age={CACHE_TTL}"})

@app.get("/api/photos")
async def api_photos(id: str = Query(...), limit: int = Query(None, ge=1, le=200)):
    world_id = id.strip()
    if not WORLD_ID_RE.match(world_id):
        raise HTTPException(status_code=400)
    world_dir = WORLDS_ROOT / world_id
    scan = scan_world_dir(world_dir)
    photos = scan["photos"]
    if limit is not None:
        photos = photos[:limit]
    return JSONResponse(content={"photos": photos}, headers={"Cache-Control": "public, max-age=300"})

async def _build_worlds_manifest_async() -> Tuple[dict, str]:
    dirs = list_world_dirs()
    items: List[dict] = []
    for d in dirs:
        wid = d.name
        scan = scan_world_dir(d)
        try:
            meta = await get_world_with_cache(wid)
        except HTTPException:
            meta = {}
        items.append({
            "id": wid,
            "name": meta.get("name") or wid,
            "authorName": meta.get("authorName"),
            "authorId": meta.get("authorId"),
            "favorites": meta.get("favorites"),
            "visits": meta.get("visits"),
            "capacity": meta.get("recommendedCapacity") or meta.get("capacity"),
            "heat": meta.get("heat"),
            "popularity": meta.get("popularity"),
            "imageUrl": meta.get("imageUrl") or meta.get("thumbnailImageUrl"),
            "thumbnailImageUrl": meta.get("thumbnailImageUrl"),
            "updated_at": meta.get("updated_at"),
            "updatedTs": iso_to_ts(meta.get("updated_at") or ""),
            "publicationDate": meta.get("publicationDate"),
            "url": f"https://vrchat.com/home/world/{wid}",
            "tags": meta.get("tags", []),
            "description": scan["desc"],
            "addedTs": scan["addedTs"],
            "photoCount": scan["photo_count"],
            "previewThumb": (scan["photos"][0]["thumb"] if scan["photo_count"] else None),
        })
    items.sort(key=lambda x: x.get("addedTs") or 0.0, reverse=True)
    import hashlib
    hasher = hashlib.sha1()
    for it in items:
        base = f'{it["id"]}:{int(it.get("addedTs") or 0)}:{int(it.get("updatedTs") or 0)}:{int(it.get("photoCount") or 0)}'
        hasher.update(base.encode("utf-8"))
    etag = '"' + hasher.hexdigest() + '"'
    return {"worlds": items}, etag

@app.get("/api/worlds")
async def api_worlds(request: Request):
    now = time.time()
    mc = _manifest_cache
    if mc["data"] is not None and mc["exp"] > now:
        inm = request.headers.get("if-none-match")
        if inm and mc["etag"] and inm == mc["etag"]:
            return Response(status_code=304, headers={"ETag": mc["etag"], "Cache-Control": f"public, max-age={MANIFEST_TTL}"})
        return JSONResponse(mc["data"], headers={"ETag": mc["etag"], "Cache-Control": f"public, max-age={MANIFEST_TTL}"})
    payload, etag = await _build_worlds_manifest_async()
    _manifest_cache["data"] = payload
    _manifest_cache["etag"] = etag
    _manifest_cache["exp"] = now + MANIFEST_TTL
    inm = request.headers.get("if-none-match")
    if inm and inm == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": f"public, max-age={MANIFEST_TTL}"})
    return JSONResponse(payload, headers={"ETag": etag, "Cache-Control": f"public, max-age={MANIFEST_TTL}"})

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)
