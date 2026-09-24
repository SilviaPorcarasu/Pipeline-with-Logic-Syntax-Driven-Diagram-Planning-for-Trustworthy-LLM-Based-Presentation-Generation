"""Typed dataclasses describing requests, outlines, slide content, and layout coordinates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PresentationRequest:
    topic: str
    slides_count: int
    style: str = "academic"
    include_images: bool = True
    include_voiceover: bool = True


@dataclass
class SlideOutline:
    slide_number: int
    type: str
    title: str
    goal: str


@dataclass
class SlideContent:
    slide_number: int
    title: str
    bullets: list[str]
    voiceover: str
    image_suggestion: str
    layout_hint: str
    subtitle: str = ""


@dataclass
class LayoutSpec:
    title_x: float
    title_y: float
    title_w: float
    title_h: float
    body_x: float
    body_y: float
    body_w: float
    body_h: float
    image_x: float | None = None
    image_y: float | None = None
    image_w: float | None = None
    image_h: float | None = None
