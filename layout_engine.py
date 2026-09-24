"""Rule-based layout selection and coordinate presets for slide rendering."""

from __future__ import annotations

from dataclasses import asdict

from models import LayoutSpec, SlideContent, SlideOutline


# Slide size: 13.333 x 7.5 inches (widescreen)
LAYOUTS: dict[str, LayoutSpec] = {
    # Title slide — centered, big
    "title": LayoutSpec(
        title_x=1.0,
        title_y=2.0,
        title_w=11.3,
        title_h=1.2,
        body_x=1.5,
        body_y=3.6,
        body_w=10.3,
        body_h=2.2,
    ),
    # Section divider — centered
    "section_divider": LayoutSpec(
        title_x=1.0,
        title_y=2.4,
        title_w=11.3,
        title_h=1.1,
        body_x=1.5,
        body_y=3.8,
        body_w=10.3,
        body_h=1.8,
    ),
    # Text only — full width
    "text_only": LayoutSpec(
        title_x=0.6,
        title_y=0.3,
        title_w=12.1,
        title_h=0.9,
        body_x=0.8,
        body_y=1.6,
        body_w=11.7,
        body_h=5.4,
    ),
    # Text left, diagram right — diagram takes ~60% of width
    # 50/50 split: text left, diagram right
    "text_left_image_right": LayoutSpec(
        title_x=0.4,
        title_y=0.2,
        title_w=12.5,
        title_h=0.85,
        body_x=0.5,
        body_y=1.4,
        body_w=5.5,
        body_h=5.5,
        image_x=6.1,
        image_y=1.1,
        image_w=7.0,
        image_h=6.1,
    ),
    # Conclusion — centered with more space
    "conclusion": LayoutSpec(
        title_x=1.0,
        title_y=0.6,
        title_w=11.3,
        title_h=0.9,
        body_x=1.2,
        body_y=1.9,
        body_w=10.9,
        body_h=4.6,
    ),
}


def choose_layout(
    outline: SlideOutline,
    content: SlideContent,
    *,
    include_images: bool,
) -> str:
    """Select the layout preset for a slide based on type/content/image availability."""
    if outline.type == "title":
        return "title"
    if outline.type == "conclusion":
        return "conclusion"
    if outline.type == "section":
        return "section_divider"

    if include_images and content.image_suggestion:
        return "text_left_image_right"

    return "text_only"


def get_layout(layout_name: str) -> LayoutSpec:
    """Return a layout preset by name with text-only fallback."""
    return LAYOUTS.get(layout_name, LAYOUTS["text_only"])


def layout_dict(layout_name: str) -> dict[str, float | None]:
    """Return layout coordinates as a serializable dictionary."""
    return asdict(get_layout(layout_name))
