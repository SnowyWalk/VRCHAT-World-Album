import os, re, time, json, asyncio, pathlib, io
from typing import Dict, Any, Tuple, List
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# ===== 설정 =====
PORT = int(os.getenv("PORT", "8080"))
CACHE_TTL = int(os.getenv("CACHE_TTL", str(24 * 60 * 60)))  # 24시간
USER_AGENT = os.getenv("USER_AGENT", "VRC-World-Site/1.3")
UPSTREAM_BASE = "https://api.vrchat.cloud/api/1/worlds/"
WORLD_ID_RE = re.compile(r"^wrld_[0-9a-f-]{8,}$", re.I)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# 썸네일 설정
THUMB_WIDTH = int(os.getenv("THUMB_WIDTH", "480"))   # 썸네일 가로(px)
THUMB_QUALITY = int(os.getenv("THUMB_QUALITY", "82"))
USE_WEBP = os.getenv("THUMB_WEBP", "1") == "1"       # webp로 저장
VIEW_WIDTH = int(os.getenv("VIEW_WIDTH", "1600")) # 라이트박스용 가로 px
VIEW_QUALITY = int(os.getenv("VIEW_QUALITY", "88"))

ROOT = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"
WORLDS_ROOT = STATIC_DIR / "worlds"
THUMBS_ROOT = STATIC_DIR / "_thumbs"  # 생성/캐시 위치
VIEWS_ROOT = STATIC_DIR / "_views"

# 캐시: worldId -> (expire_ts, data)
_cache: Dict[str, Tuple[float, Any]] = {}
_locks: Dict[str, asyncio.Lock] = {}
http_client: httpx.AsyncClient | None = None

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
        headers={"Accept":"application/json","User-Agent":USER_AGENT},
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

# ---- 파일 시스템 스캔 ----
def list_world_dirs() -> List[pathlib.Path]:
    if not WORLDS_ROOT.exists(): return []
    return [p for p in WORLDS_ROOT.iterdir() if p.is_dir() and WORLD_ID_RE.match(p.name or "")]

def dir_timestamp(p: pathlib.Path) -> float:
    st = p.stat()
    base = getattr(st, "st_birthtime", None)
    base = float(base) if base else float(st.st_mtime)
    latest_file_ts = 0.0
    try:
        for f in p.iterdir():
            if f.is_file():
                latest_file_ts = max(latest_file_ts, float(f.stat().st_mtime))
    except Exception:
        pass
    return max(base, latest_file_ts)

def list_photo_files(world_dir: pathlib.Path) -> List[pathlib.Path]:
    files = []
    try:
        for f in sorted(world_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
                files.append(f)
    except Exception:
        pass
    return files

def read_local_desc(world_dir: pathlib.Path) -> str:
    """
    월드 폴더 내 .txt 파일을 하나 골라 내용을 설명으로 사용.
    없으면 빈 문자열 반환.
    """
    try:
        if not world_dir.exists():
            return ""
        # 폴더 안의 .txt 파일 중 사전순으로 첫 번째 선택
        cands = sorted(
            [p for p in world_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
            key=lambda p: p.name.lower()
        )
        if not cands:
            return ""
        data = cands[0].read_bytes()
        # 인코딩 추정 (utf-8 우선, 그 다음 흔한 한글 인코딩)
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
            try:
                return data.decode(enc).strip()
            except Exception:
                continue
        # 마지막 폴백: 손실 감수하고 utf-8로 디코드
        return data.decode("utf-8", "ignore").strip()
    except Exception:
        return ""

# ---- 썸네일 생성/서빙 ----
def thumb_target_path(src: pathlib.Path) -> pathlib.Path:
    wid = src.parent.name
    stem = src.stem # 확장자 제거 이름
    ext = ".webp" if USE_WEBP else src.suffix.lower()
    dst_dir = THUMBS_ROOT / wid
    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir / f"{stem}{ext}"
    
def view_target_path(src: pathlib.Path) -> pathlib.Path:
    wid = src.parent.name
    stem = src.stem
    ext = ".webp" if USE_WEBP else src.suffix.lower()
    dst_dir = VIEWS_ROOT / wid
    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir / f"{stem}{ext}"

def make_thumb_if_needed(src: pathlib.Path) -> pathlib.Path:
    dst = thumb_target_path(src)
    return _ensure_resized(src, dst, THUMB_WIDTH, THUMB_QUALITY, USE_WEBP)

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
        # 실패 시 원본 복사 경로로 대체
        return src
    return dst

@app.get("/thumbs/{world_id}/{filename:path}")
def get_thumb(world_id: str, filename: str):
    # filename 예: 01.jpg
    src = WORLDS_ROOT / world_id / filename
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail="not found")
    if src.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="unsupported file type")
    dst = make_thumb_if_needed(src)
    media = "image/webp" if (USE_WEBP and src.suffix.lower() != ".webp") else None
    return FileResponse(dst, media_type=media, headers={"Cache-Control":"public, max-age=604800"})
#     return FileResponse(dst, headers={"Cache-Control": "public, max-age=604800"})  # 7일

@app.get("/view/{world_id}/{filename:path}")
def get_view(world_id: str, filename: str):
    src = WORLDS_ROOT / world_id / filename
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail="not found")
    if src.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="unsupported file type")
    dst = view_target_path(src)
    out = _ensure_resized(src, dst, VIEW_WIDTH, VIEW_QUALITY, USE_WEBP)
    media = "image/webp" if (USE_WEBP and src.suffix.lower() != ".webp") else None
    return FileResponse(out, media_type=media, headers={"Cache-Control": "public, max-age=1209600"}) # 14일

# ---- API ----
@app.get("/healthz")
async def healthz():
    return {"ok": True, "cache_items": len(_cache), "ttl": CACHE_TTL}

@app.get("/api/world")
async def api_world(id: str = Query(..., description="VRChat world id (wrld_...)")):
    world_id = id.strip()
    if not WORLD_ID_RE.match(world_id):
        raise HTTPException(status_code=400, detail="invalid world id")
    meta = await get_world_with_cache(world_id)
    world_dir = WORLDS_ROOT / world_id
    added_ts = dir_timestamp(world_dir) if world_dir.exists() else 0.0
    photos = []
    for p in list_photo_files(world_dir):
        rel = p.relative_to(STATIC_DIR).as_posix()             # /static/worlds/...
        full = "/static/" + rel
        thumb = f"/thumbs/{world_id}/{p.name}"
        photos.append({"full": full, "thumb": thumb, "view": f"/view/{world_id}/{p.name}",})
    local_desc = read_local_desc(world_dir)

    merged = {
        "id": world_id,
        "name": meta.get("name"),
        "description": local_desc, # meta.get("description"),
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
        "photos": photos,
        "addedTs": added_ts,
        "authorId": meta.get("authorId"),
    }
    return JSONResponse(content=merged, headers={"Cache-Control": f"public, max-age={CACHE_TTL}"})

@app.get("/api/worlds")
async def api_worlds():
    dirs = list_world_dirs()
    sortable = [(dir_timestamp(d), d) for d in dirs]
    sortable.sort(key=lambda t: t[0], reverse=True)

    results: List[dict] = []
    for added_ts, d in sortable:
        wid = d.name
        try:
            meta = await get_world_with_cache(wid)
        except HTTPException:
            meta = {"name": wid, "description": "", "authorName": None, "favorites": 0,
                    "visits": 0, "capacity": None, "heat": None, "popularity": None,
                    "imageUrl": None, "thumbnailImageUrl": None, "updated_at": None,
                    "publicationDate": None, "tags": []}

        photos = []
        for p in list_photo_files(d):
            rel = p.relative_to(STATIC_DIR).as_posix()
            photos.append({
                "full": "/static/" + rel,
                "thumb": f"/thumbs/{wid}/{p.name}",
                "view": f"/view/{wid}/{p.name}",
            })
        local_desc = read_local_desc(d)

        results.append({
            "id": wid,
            "name": meta.get("name") or wid,
            "description": local_desc, # meta.get("description") or "",
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
            "url": f"https://vrchat.com/home/world/{wid}",
            "tags": meta.get("tags", []),
            "photos": photos,
            "addedTs": added_ts,
            "authorId": meta.get("authorId"),
        })

    return JSONResponse(content={"worlds": results}, headers={"Cache-Control": "no-store"})

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)