"""Frame rendering, TTS generation, timeline composition, and MP4 muxing."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

from models import SlideContent


PLAIN_PALETTE: dict[str, tuple[int, int, int]] = {
    "bg": (255, 255, 255),
    "title": (28, 47, 88),
    "subtitle": (72, 90, 120),
    "body": (47, 64, 96),
}


def _normalize_space(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    return " ".join(str(text).split())


def _estimate_duration(text: str, *, min_seconds: float, words_per_second: float) -> float:
    words = len([w for w in text.split() if _normalize_space(w)])
    wps = max(0.5, float(words_per_second))
    return max(float(min_seconds), float(words) / wps)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load font."""
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    xy: tuple[int, int],
    max_width: int,
    line_height: int,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> int:
    """Render wrapped text lines inside a bounded width and return final Y position."""
    def _split_long_word(word: str) -> list[str]:
        # Hard-wrap very long tokens so they cannot overflow the available width.
        if int(draw.textlength(word, font=font)) <= max_width:
            return [word]
        parts: list[str] = []
        current = ""
        for ch in word:
            trial = f"{current}{ch}"
            if current and int(draw.textlength(trial, font=font)) > max_width:
                parts.append(current)
                current = ch
            else:
                current = trial
        if current:
            parts.append(current)
        return parts

    expanded_words: list[str] = []
    for raw in text.split():
        expanded_words.extend(_split_long_word(raw))

    lines: list[str] = []
    current: list[str] = []
    for word in expanded_words:
        trial = " ".join(current + [word]).strip()
        box = draw.textbbox((0, 0), trial, font=font)
        if box[2] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def _fit_title_lines(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    max_width: int,
    max_lines: int,
    max_size: int,
    min_size: int,
) -> tuple[ImageFont.ImageFont, list[str]]:
    """Pick a title font/line-wrap pair that fits width and line-count constraints."""
    normalized = _normalize_space(text)
    if not normalized:
        return _load_font(max_size), [""]

    words = normalized.split()
    fallback_font = _load_font(min_size)
    fallback_lines = [normalized]

    for size in range(int(max_size), int(min_size) - 1, -2):
        font = _load_font(size)
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            trial = " ".join(current + [word]).strip()
            if int(draw.textlength(trial, font=font)) <= max_width or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        if lines and len(lines) <= max_lines:
            return font, lines
        if lines:
            fallback_font = font
            fallback_lines = lines

    # Clamp to max lines using ellipsis when title is still too long.
    trimmed = fallback_lines[:max_lines]
    if trimmed:
        last = trimmed[-1]
        ell = "..."
        while last and int(draw.textlength(f"{last}{ell}", font=fallback_font)) > max_width:
            last = last[:-1].rstrip()
        trimmed[-1] = f"{last}{ell}" if last else ell
    return fallback_font, trimmed or [normalized]


def _draw_fitted_title(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    x: int,
    y: int,
    max_width: int,
    fill: tuple[int, int, int],
    max_size: int,
    min_size: int,
) -> int:
    """Draw a wrapped title that fits and return the next free y coordinate."""
    font, lines = _fit_title_lines(
        draw,
        text=text,
        max_width=max_width,
        max_lines=2,
        max_size=max_size,
        min_size=min_size,
    )
    spacing = max(6, int(getattr(font, "size", min_size)) // 5)
    y_cursor = y
    for line in lines:
        draw.text((x, y_cursor), line, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        y_cursor += int(bbox[3] - bbox[1]) + spacing
    return y_cursor


def _draw_fitted_subtitle(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    x: int,
    y: int,
    max_width: int,
    fill: tuple[int, int, int],
    max_size: int,
    min_size: int,
) -> int:
    """Draw one compact subtitle line and return next free y coordinate."""
    normalized = _normalize_space(text)
    if not normalized:
        return y
    font, lines = _fit_title_lines(
        draw,
        text=normalized,
        max_width=max_width,
        max_lines=1,
        max_size=max_size,
        min_size=min_size,
    )
    y_cursor = y
    for line in lines[:1]:
        draw.text((x, y_cursor), line, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        y_cursor += int(bbox[3] - bbox[1]) + max(4, int(getattr(font, "size", min_size)) // 6)
    return y_cursor


def _paste_contained(base: Image.Image, image_path: Path, box: tuple[int, int, int, int]) -> None:
    img = Image.open(image_path).convert("RGB")
    left, top, right, bottom = box
    max_w = max(1, right - left)
    max_h = max(1, bottom - top)
    ratio = min(max_w / img.width, max_h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    x = left + (max_w - new_w) // 2
    y = top + (max_h - new_h) // 2
    base.paste(resized, (x, y))


def _render_frame(
    *,
    out_path: Path,
    slide: SlideContent,
    image_path: Path | None,
    width: int,
    height: int,
    style: str = "academic",
) -> None:
    """Render one slide frame image for recording/video export."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    palette = PLAIN_PALETTE
    img = Image.new("RGB", (width, height), color=palette["bg"])
    draw = ImageDraw.Draw(img)

    body_font = _load_font(34 if width >= 1800 else 24)
    title_bottom = _draw_fitted_title(
        draw,
        text=slide.title,
        x=56,
        y=34,
        max_width=max(200, width - 112),
        fill=palette["title"],
        max_size=44 if width >= 1800 else 34,
        min_size=24 if width >= 1800 else 18,
    )
    subtitle_bottom = _draw_fitted_subtitle(
        draw,
        text=getattr(slide, "subtitle", ""),
        x=56,
        y=title_bottom + 4,
        max_width=max(200, width - 112),
        fill=palette["subtitle"],
        max_size=28 if width >= 1800 else 20,
        min_size=16 if width >= 1800 else 13,
    )

    has_image = image_path is not None and image_path.exists()
    content_top = max(150, subtitle_bottom + 22)
    if has_image:
        left_box = (60, content_top, int(width * 0.52), height - 90)
        right_box = (int(width * 0.56), content_top, width - 60, height - 130)
    else:
        left_box = (60, content_top, width - 60, height - 90)
        right_box = None

    y = left_box[1] + 28
    for bullet in slide.bullets:
        y = _draw_wrapped_text(
            draw,
            text=f"- {bullet}",
            xy=(left_box[0] + 24, y),
            max_width=(left_box[2] - left_box[0] - 48),
            line_height=46 if width >= 1800 else 30,
            fill=palette["body"],
            font=body_font,
        )
        y += 12

    if has_image and right_box is not None:
        _paste_contained(img, image_path, (right_box[0] + 18, right_box[1] + 18, right_box[2] - 18, right_box[3] - 18))

    img.save(out_path)


def _ffmpeg_bin() -> str | None:
    """Resolve ffmpeg binary path from imageio-ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _ffprobe_bin(ffmpeg_bin: str | None) -> str | None:
    """Resolve ffprobe binary path matching the active ffmpeg runtime."""
    candidate = shutil.which("ffprobe")
    if candidate:
        return candidate
    if ffmpeg_bin:
        sibling = Path(ffmpeg_bin).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)
    return None


def _h264_level_for(width: int, height: int, fps: int) -> str:
    pixels = max(1, int(width)) * max(1, int(height))
    rate = max(1, int(fps))
    if pixels <= 640 * 480 and rate <= 30:
        return "3.0"
    if pixels <= 1280 * 720 and rate <= 30:
        return "3.1"
    if pixels <= 1280 * 720 and rate <= 60:
        return "3.2"
    if pixels <= 1920 * 1080 and rate <= 30:
        return "4.0"
    if pixels <= 1920 * 1080 and rate <= 60:
        return "4.2"
    return "5.1"


def _audio_duration_seconds(path: Path, *, ffprobe_bin: str | None) -> float:
    """Read media duration in seconds using ffprobe, defaulting to 0 on failure."""
    if ffprobe_bin is None or not path.exists():
        return 0.0
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore").strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def _media_has_audio_stream(path: Path, *, ffprobe_bin: str | None) -> bool | None:
    """Detect whether a media file contains an audio stream via ffprobe."""
    if ffprobe_bin is None or not path.exists():
        return None
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore").strip().lower()
        return out == "audio"
    except Exception:
        return None


def _media_has_audio_stream_ffmpeg(path: Path, *, ffmpeg_bin: str | None) -> bool | None:
    """Detect audio stream availability via ffmpeg stderr probing."""
    if ffmpeg_bin is None or not path.exists():
        return None
    cmd = [ffmpeg_bin, "-hide_banner", "-i", str(path)]
    try:
        probe = subprocess.run(cmd, capture_output=True, text=True, check=False)
        stderr = (probe.stderr or "").lower()
        return "audio:" in stderr
    except Exception:
        return None


def _prepare_tts_text(text: str) -> str:
    """Normalize narration text into a cleaner form for TTS engines."""
    t = _normalize_space(text)
    if not t:
        return "This slide summarizes the key concept."
    replacements = {
        "->": " leads to ",
        "=>": " results in ",
        "<->": " relates to ",
        "≈": " approximately ",
        "∂": " partial derivative ",
        "∇": " gradient ",
        "α": " alpha ",
        "β": " beta ",
        "γ": " gamma ",
        "θ": " theta ",
    }
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    # Replace dense separators that often sound robotic in synthetic speech.
    for ch in ("|", "/", "\\", ";"):
        t = t.replace(ch, ". ")
    t = _normalize_space(t)
    if t and t[-1] not in ".!?":
        t = f"{t}."
    return t


async def _edge_tts_save(
    *,
    text: str,
    voice: str,
    out_path: Path,
    rate: str = "-10%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> None:
    import edge_tts

    communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    await communicator.save(str(out_path))


def _elevenlabs_tts_save(*, text: str, voice_id: str, out_path: Path) -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for voiceover-engine=elevenlabs.")
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
    output_format = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip() or "mp3_44100_128"
    endpoint_base = (
        os.environ.get("ELEVENLABS_TTS_ENDPOINT", "https://api.elevenlabs.io/v1/text-to-speech")
        .strip()
        .rstrip("/")
    )
    url = f"{endpoint_base}/{voice_id}?output_format={quote_plus(output_format)}"
    payload = {
        "text": text,
        "model_id": model_id,
    }
    stability = os.environ.get("ELEVENLABS_STABILITY")
    similarity = os.environ.get("ELEVENLABS_SIMILARITY_BOOST")
    if stability is not None or similarity is not None:
        voice_settings: dict[str, float] = {}
        try:
            if stability is not None:
                voice_settings["stability"] = float(stability)
            if similarity is not None:
                voice_settings["similarity_boost"] = float(similarity)
        except Exception:
            voice_settings = {}
        if voice_settings:
            payload["voice_settings"] = voice_settings
    req = Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    try:
        with urlopen(req, timeout=180) as response:
            blob = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"ElevenLabs request failed: {exc}") from exc
    if not blob:
        raise RuntimeError("ElevenLabs returned empty audio payload.")
    out_path.write_bytes(blob)


def _generate_audio_tracks(
    *,
    slides: list[SlideContent],
    audio_dir: Path,
    engine: str,
    voice: str,
) -> dict[int, Path]:
    """Generate one audio track per slide using the selected TTS engine."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    if engine == "none":
        return {}

    tracks: dict[int, Path] = {}
    if engine == "edge-tts":
        try:
            import edge_tts  # noqa: F401
        except Exception as exc:
            raise RuntimeError("edge-tts package is required for voiceover-engine=edge-tts.") from exc
        requested_voice = (voice or "").strip() or "en-US-JennyNeural"
        fallback_voice = "en-US-JennyNeural"
        edge_rate = (os.environ.get("EDGE_TTS_RATE", "-10%") or "-10%").strip()
        edge_pitch = (os.environ.get("EDGE_TTS_PITCH", "+0Hz") or "+0Hz").strip()
        edge_volume = (os.environ.get("EDGE_TTS_VOLUME", "+0%") or "+0%").strip()
        for slide in slides:
            out = audio_dir / f"slide_{slide.slide_number:02d}.mp3"
            tts_text = _prepare_tts_text(slide.voiceover)
            try:
                asyncio.run(
                    _edge_tts_save(
                        text=tts_text,
                        voice=requested_voice,
                        out_path=out,
                        rate=edge_rate,
                        pitch=edge_pitch,
                        volume=edge_volume,
                    )
                )
            except ValueError as exc:
                msg = str(exc)
                if "Invalid voice" not in msg or requested_voice == fallback_voice:
                    raise
                asyncio.run(
                    _edge_tts_save(
                        text=tts_text,
                        voice=fallback_voice,
                        out_path=out,
                        rate=edge_rate,
                        pitch=edge_pitch,
                        volume=edge_volume,
                    )
                )
            tracks[slide.slide_number] = out
        return tracks

    if engine == "espeak":
        if shutil.which("espeak") is None:
            raise RuntimeError("espeak not found in PATH.")
        for slide in slides:
            out = audio_dir / f"slide_{slide.slide_number:02d}.wav"
            subprocess.run(
                ["espeak", "-w", str(out), slide.voiceover],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tracks[slide.slide_number] = out
        return tracks

    if engine == "elevenlabs":
        if not os.environ.get("ELEVENLABS_API_KEY", "").strip():
            raise RuntimeError("ELEVENLABS_API_KEY is required for voiceover-engine=elevenlabs.")
        voice_id = (voice or "").strip() or os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        if not voice_id:
            raise RuntimeError("Set VOICE or ELEVENLABS_VOICE_ID for voiceover-engine=elevenlabs.")
        for slide in slides:
            out = audio_dir / f"slide_{slide.slide_number:02d}.mp3"
            _elevenlabs_tts_save(text=_prepare_tts_text(slide.voiceover), voice_id=voice_id, out_path=out)
            tracks[slide.slide_number] = out
        return tracks

    raise RuntimeError(f"Unsupported voiceover engine: {engine}")


def _render_video(
    *,
    recording_dir: Path,
    timeline: list[dict[str, Any]],
    width: int,
    height: int,
    fps: int,
) -> tuple[Path | None, str | None]:
    """Render a silent slide video timeline from frame images."""
    ffmpeg = _ffmpeg_bin()
    if ffmpeg is None:
        return None, "ffmpeg not available"

    clips_dir = recording_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    h264_level = _h264_level_for(width, height, fps)

    for item in timeline:
        slide_no = int(item["slide_number"])
        frame = Path(str(item["frame_png"]))
        duration = float(item["duration_seconds"])
        clip = clips_dir / f"clip_{slide_no:02d}.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(max(1, int(fps))),
            "-i",
            str(frame),
            "-t",
            f"{max(0.2, duration):.3f}",
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            h264_level,
            "-an",
            "-movflags",
            "+faststart",
            str(clip),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_paths.append(clip)

    concat_list = recording_dir / "concat_list.txt"
    concat_lines: list[str] = []
    for path in clip_paths:
        escaped = str(path).replace("'", "'\\''")
        concat_lines.append(f"file '{escaped}'\n")
    concat_list.write_text("".join(concat_lines), encoding="utf-8")
    out_video = recording_dir / "recording.mp4"
    cmd_reencode = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        h264_level,
        "-r",
        str(max(1, int(fps))),
        "-an",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    subprocess.run(cmd_reencode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    compat_video = recording_dir / "recording_compat.mp4"
    cmd_compat = [
        ffmpeg,
        "-y",
        "-i",
        str(out_video),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        h264_level,
        "-r",
        str(max(1, int(fps))),
        "-an",
        "-movflags",
        "+faststart",
        str(compat_video),
    ]
    subprocess.run(cmd_compat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return compat_video, None


def _compose_narration_track(
    *,
    ffmpeg: str,
    recording_dir: Path,
    timeline: list[dict[str, Any]],
) -> Path | None:
    """Concatenate per-slide narration tracks into one continuous audio asset."""
    audio_items: list[Path] = []
    for item in timeline:
        audio_file = item.get("audio_file")
        if not audio_file:
            continue
        path = Path(str(audio_file))
        if not path.exists():
            continue
        audio_items.append(path)

    if not audio_items:
        return None

    out_audio = recording_dir / "narration_mix.m4a"
    cmd: list[str] = [ffmpeg, "-y"]
    for path in audio_items:
        cmd.extend(["-i", str(path)])

    filters: list[str] = []
    concat_inputs: list[str] = []
    for idx in range(len(audio_items)):
        filters.append(
            f"[{idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=1.0[a{idx}]"
        )
        concat_inputs.append(f"[a{idx}]")
    filters.append(
        f"{''.join(concat_inputs)}concat=n={len(audio_items)}:v=0:a=1[aout]"
    )

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-b:a",
            "192k",
            str(out_audio),
        ]
    )
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_audio


def _build_timeline_objects(
    *,
    timeline: list[dict[str, Any]],
    voiceover_audio: Path | None,
    voiceover_duration_seconds: float | None,
) -> list[dict[str, Any]]:
    """Build timeline objects."""
    objects: list[dict[str, Any]] = []
    for item in timeline:
        objects.append(
            {
                "type": "video",
                "track": "video_main",
                "start": float(item.get("start_seconds", 0.0)),
                "duration": float(item.get("duration_seconds", 0.0)),
                "asset_id": f"slide-{int(item.get('slide_number', 0)):02d}",
                "asset_path": str(item.get("frame_png", "")),
                "volume": 0.1,
            }
        )

    if voiceover_audio is not None:
        objects.append(
            {
                "type": "audio",
                "track": "audio_voiceover",
                "start": 0.0,
                "duration": max(0.0, float(voiceover_duration_seconds or 0.0)),
                "asset_id": "voiceover-main",
                "asset_path": str(voiceover_audio),
            }
        )

    return objects


def _compose_narration_track_concat_demuxer(
    *,
    ffmpeg: str,
    recording_dir: Path,
    timeline: list[dict[str, Any]],
) -> Path | None:
    """Fallback narration concatenation using ffmpeg concat demuxer."""
    audio_paths: list[Path] = []
    for item in timeline:
        audio_file = item.get("audio_file")
        if not audio_file:
            continue
        path = Path(str(audio_file))
        if path.exists():
            audio_paths.append(path)
    if not audio_paths:
        return None

    list_path = recording_dir / "audio_concat_list.txt"
    lines = []
    for path in audio_paths:
        escaped = str(path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'\n")
    list_path.write_text("".join(lines), encoding="utf-8")

    out_audio = recording_dir / "narration_concat.m4a"
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "192k",
        str(out_audio),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_audio


def _mux_video_with_audio(
    *,
    ffmpeg: str,
    input_video: Path,
    input_audio: Path,
    output_video: Path,
) -> Path:
    """Mux narration audio onto the rendered video stream."""
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(input_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_video


def build_recording(
    *,
    slides: list[SlideContent],
    image_by_slide: dict[int, Path],
    recording_dir: Path,
    engine: str,
    voice: str,
    style: str,
    width: int,
    height: int,
    fps: int,
    min_slide_seconds: float,
    words_per_second: float,
) -> dict[str, Any]:
    """Build timeline, audio tracks, and final recording artifacts for the deck."""
    recording_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = recording_dir / "frames"
    audio_dir = recording_dir / "audio"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for slide in slides:
        frame_path = frames_dir / f"slide_{slide.slide_number:02d}.png"
        _render_frame(
            out_path=frame_path,
            slide=slide,
            image_path=image_by_slide.get(slide.slide_number),
            width=max(640, int(width)),
            height=max(360, int(height)),
            style=style,
        )

    audio_tracks = _generate_audio_tracks(
        slides=slides,
        audio_dir=audio_dir,
        engine=engine,
        voice=voice,
    )
    if engine != "none" and len(audio_tracks) != len(slides):
        raise RuntimeError(
            f"Voiceover requested, but audio generation is incomplete: got {len(audio_tracks)} tracks for {len(slides)} slides."
        )

    ffprobe = _ffprobe_bin(_ffmpeg_bin())
    timeline: list[dict[str, Any]] = []
    cursor = 0.0
    for slide in slides:
        audio_path = audio_tracks.get(slide.slide_number)
        duration = 0.0
        if audio_path is not None:
            duration = _audio_duration_seconds(audio_path, ffprobe_bin=ffprobe)
            if engine != "none" and duration <= 0.0:
                raise RuntimeError(
                    f"Slide {slide.slide_number}: generated audio is empty/invalid ({audio_path})."
                )
        if duration <= 0.0:
            duration = _estimate_duration(
                slide.voiceover,
                min_seconds=min_slide_seconds,
                words_per_second=words_per_second,
            )
        frame = frames_dir / f"slide_{slide.slide_number:02d}.png"
        item = {
            "slide_number": slide.slide_number,
            "start_seconds": round(cursor, 3),
            "duration_seconds": round(duration, 3),
            "end_seconds": round(cursor + duration, 3),
            "frame_png": str(frame),
            "audio_file": str(audio_path) if audio_path is not None else None,
            "narration": slide.voiceover,
        }
        timeline.append(item)
        cursor += duration

    timeline_path = recording_dir / "timeline.json"
    timeline_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    base_video_path, warning = _render_video(
        recording_dir=recording_dir,
        timeline=timeline,
        width=max(640, int(width)),
        height=max(360, int(height)),
        fps=max(1, int(fps)),
    )
    warnings: list[str] = []
    if warning:
        warnings.append(str(warning))
    ffmpeg = _ffmpeg_bin()
    narration_audio: Path | None = None
    video_path: Path | None = base_video_path
    if engine != "none":
        if base_video_path is None:
            raise RuntimeError("Video rendering failed before narration mux.")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to mux narration audio.")
        narration_audio = _compose_narration_track(ffmpeg=ffmpeg, recording_dir=recording_dir, timeline=timeline)
        ffprobe_tmp = _ffprobe_bin(ffmpeg)
        narration_duration = (
            _audio_duration_seconds(narration_audio, ffprobe_bin=ffprobe_tmp)
            if narration_audio is not None and narration_audio.exists()
            else 0.0
        )
        if narration_duration <= 0.0:
            narration_audio = _compose_narration_track_concat_demuxer(
                ffmpeg=ffmpeg, recording_dir=recording_dir, timeline=timeline
            )
            narration_duration = (
                _audio_duration_seconds(narration_audio, ffprobe_bin=ffprobe_tmp)
                if narration_audio is not None and narration_audio.exists()
                else 0.0
            )
        if narration_audio is None or not narration_audio.exists() or narration_duration <= 0.0:
            raise RuntimeError("Could not compose a valid narration track.")
        with_voice = recording_dir / "recording_with_voice.mp4"
        video_path = _mux_video_with_audio(
            ffmpeg=ffmpeg,
            input_video=Path(base_video_path),
            input_audio=narration_audio,
            output_video=with_voice,
        )
    if video_path is not None:
        canonical_video = recording_dir / "recording.mp4"
        current_video = Path(video_path)
        if current_video != canonical_video:
            shutil.copyfile(current_video, canonical_video)
            video_path = canonical_video

    ffprobe_final = _ffprobe_bin(ffmpeg)
    has_audio_stream = _media_has_audio_stream(Path(video_path), ffprobe_bin=ffprobe_final) if video_path else None
    if has_audio_stream is None and video_path is not None:
        has_audio_stream = _media_has_audio_stream_ffmpeg(Path(video_path), ffmpeg_bin=ffmpeg)
    if engine != "none" and video_path is not None and has_audio_stream is False:
        # Emergency remux: rebuild final file from the known base visual stream + narration track.
        if ffmpeg is None or base_video_path is None or narration_audio is None or not narration_audio.exists():
            raise RuntimeError("final video has no audio stream")
        forced_video = recording_dir / "recording_with_voice_forced.mp4"
        video_path = _mux_video_with_audio(
            ffmpeg=ffmpeg,
            input_video=Path(base_video_path),
            input_audio=narration_audio,
            output_video=forced_video,
        )
        has_audio_stream = _media_has_audio_stream(Path(video_path), ffprobe_bin=ffprobe_final)
        if has_audio_stream is None:
            has_audio_stream = _media_has_audio_stream_ffmpeg(Path(video_path), ffmpeg_bin=ffmpeg)
        if has_audio_stream is False:
            raise RuntimeError("final video has no audio stream")
    voiceover_duration = (
        _audio_duration_seconds(narration_audio, ffprobe_bin=ffprobe_final)
        if narration_audio is not None and ffprobe_final is not None
        else None
    )
    timeline_objects = _build_timeline_objects(
        timeline=timeline,
        voiceover_audio=narration_audio if narration_audio is not None and narration_audio.exists() else None,
        voiceover_duration_seconds=voiceover_duration,
    )
    timeline_objects_path = recording_dir / "timeline_objects.json"
    timeline_objects_path.write_text(json.dumps({"objects": timeline_objects}, indent=2), encoding="utf-8")

    return {
        "timeline_json": str(timeline_path),
        "timeline_objects_json": str(timeline_objects_path),
        "recording_video": str(video_path) if video_path else None,
        "recording_warning": "; ".join(warnings) if warnings else None,
        "audio_dir": str(audio_dir),
        "audio_tracks_count": len(audio_tracks),
        "voiceover_engine_used": engine,
        "video_has_audio_stream": has_audio_stream,
        "voiceover_audio_file": str(narration_audio) if narration_audio is not None else None,
        "voiceover_audio_duration_seconds": voiceover_duration,
    }
