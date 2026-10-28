#!/usr/bin/env python3
"""Render the authored README diagrams using only Python's standard library.

The compact layouts rearrange the same contracts for a narrow viewport. This
script does not inspect assets, load models or run any pipeline stage.
"""

import argparse
from html import escape
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "diagrams"
SANS = "Arial, Helvetica, sans-serif"
MONO = "Menlo, Consolas, monospace"
THEMES = {
    "light": {
        "bg": "#f6f8fc", "panel": "#ffffff", "ink": "#182235", "muted": "#526176",
        "line": "#ccd5e2", "route": "#526176", "rose": "#9f1239", "rose_bg": "#fff1f2",
        "blue": "#1d4ed8", "blue_bg": "#eff6ff", "teal": "#0f766e", "teal_bg": "#f0fdfa",
        "amber": "#92400e", "amber_bg": "#fffbeb", "grid": "#e1e7ef",
    },
    "dark": {
        "bg": "#0d1117", "panel": "#161e2b", "ink": "#eaf0f8", "muted": "#a7b3c4",
        "line": "#344358", "route": "#8796ad", "rose": "#fda4af", "rose_bg": "#381724",
        "blue": "#93c5fd", "blue_bg": "#172848", "teal": "#5eead4", "teal_bg": "#10352f",
        "amber": "#fcd34d", "amber_bg": "#3d2b12", "grid": "#273448",
    },
}
RECOVERY_ROWS = (
    ("No committed reference", "No trusted image commit yet", "Generate reference + image QA"),
    ("Image committed", "3D bundle incomplete", "Check image QA; rebuild 3D"),
    ("Geometry complete", "Required QA missing or stale", "Inspect the existing exports"),
    ("Current QA passes", "Final release still needs accounting", "Audit every saved target"),
)
RELEASE_RULES = (
    "Default policy eligible", "No recorded override", "No quarantine",
    "Complete deliverable", "All required current QA",
)
DESCRIPTIONS = {
    "local-pipeline": (
        "The atlas-managed offline pipeline freezes policy-screened manifest rows and settings. "
        "Qwen-Image-2.1 produces an RGBA reference: native alpha is preserved, while opaque output "
        "uses BiRefNet or chroma matting. Image QA and the required safety screen gate TRELLIS.2-4B, "
        "which uses DINOv3 conditioning and a trained structure decoder. Its shared decoded mesh "
        "and voxel attributes feed both a textured Y-up GLB and native Z-up Gaussians, usually SPZ "
        "with SPLAT fallback. Mesh and Gaussian QA are separate required gates. A separate audit "
        "determines release membership. The arrows describe data flow, not parallel execution."
    ),
    "offline-recovery": (
        "The latest saved Start, Resume or Pause command controls the managed worker. Pause saves "
        "paused intent before stopping owned children; Status only reads saved state. Resume "
        "verifies the saved plan, artifact hashes and current QA. Missing references are generated; "
        "a committed image with incomplete geometry keeps its image and rebuilds the complete 3D "
        "stage; complete geometry with missing QA is inspected; current passes still need positive "
        "release membership for every saved target. Resource waits recheck automatically. A crashed "
        "worker can restart within the logged-in session if intent is running. After reboot, user "
        "login is also required. Unresolved prerequisites, "
        "changed snapshots, exhausted failures and quarantine need attention. An interrupted sampling "
        "stage restarts; completed hash-verified commits are reused."
    ),
    "artifact-evidence": (
        "Release eligibility has three distinct layers. Committed files include a PNG with matching "
        "image metadata and a GLB, recognized Gaussian container and geometry metadata. Image "
        "metadata binds the configured model and PNG hash; geometry records the actual source PNG "
        "hash. Current image, mesh and Gaussian QA must bind the inspected file hashes and current "
        "QA contract, including the required safety screen. Positive release membership additionally "
        "requires default policy eligibility, no override or quarantine and a complete deliverable. "
        "The local release manifest records included IDs and file hashes. The controller completes "
        "only when every saved target is included. Visual fidelity, metric scale, collision quality "
        "and physics suitability need separate review."
    ),
}


class Canvas:
    def __init__(self, width, height, theme, title, description):
        self.width, self.height, self.p = width, height, THEMES[theme]
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<!-- Source: scripts/render_readme_diagrams.py. -->',
            '<defs>',
        ]
        for color in ("route", "rose", "blue", "teal", "amber"):
            self.parts.append(
                f'<marker id="arrow-{color}" viewBox="0 0 8 8" refX="8" refY="4" '
                'markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
                f'<path d="M0 0L8 4L0 8Z" fill="{self.p[color]}"/></marker>')
        self.parts += [
            '<pattern id="alpha" width="12" height="12" patternUnits="userSpaceOnUse">',
            f'<rect width="12" height="12" fill="{self.p["panel"]}"/>',
            f'<path d="M0 0H6V6H0ZM6 6H12V12H6Z" fill="{self.p["grid"]}"/>',
            '</pattern></defs>',
        ]
        self.rect(0, 0, width, height, "bg", radius=20)
        self.rect(0.75, 0.75, width - 1.5, height - 1.5, "none", "line", radius=20)

    def color(self, value):
        return self.p.get(value, value)

    def rect(self, x, y, width, height, fill="panel", stroke=None, radius=14):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
            f'fill="{self.color(fill)}" stroke="{self.color(stroke) if stroke else "none"}" stroke-width="1.5"/>')

    def text(self, x, y, value, size=20, color="ink", weight=400, mono=False, anchor="start"):
        family = MONO if mono else SANS
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{self.color(color)}" text-anchor="{anchor}">'
            f'{escape(value)}</text>')

    def lines(self, x, y, values, size=18, color="muted", step=26, **kwargs):
        for index, value in enumerate(values):
            self.text(x, y + index * step, value, size, color, **kwargs)

    def path(self, data, color="route", width=2, fill="none", dashed=False, arrow=False, opacity=1):
        self.parts.append(
            f'<path d="{data}" fill="{self.color(fill)}" stroke="{self.color(color)}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"'
            + (' stroke-dasharray="6 6"' if dashed else '')
            + (f' marker-end="url(#arrow-{color})"' if arrow else '') + '/>')

    def route(self, points, color="route", arrow=True, dashed=False):
        data = "M" + "L".join(f"{x} {y}" for x, y in points)
        self.path(data, color, 2, arrow=arrow, dashed=dashed)

    def dot(self, x, y, radius=4, color="rose", fill=None):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{self.color(fill or color)}" '
            f'stroke="{self.color(color)}" stroke-width="1.5"/>')

    def pill(self, x, y, width, label, color="blue", size=18, height=36):
        self.rect(x, y, width, height, color + "_bg", color, radius=height / 2)
        self.text(x + width / 2, y + height / 2 + size * 0.34, label, size, color, 600, anchor="middle")

    def panel(self, x, y, width, height, color=None):
        self.rect(x, y, width, height, color + "_bg" if color else "panel", color or "line")
        for offset in (20, 29, 38):
            self.path(f"M{x + offset} {y}v5", color or "line", 1.5)

    def header(self, number, label, title, subtitle, compact=False):
        margin = 24 if compact else 36
        for index in range(5):
            offset = index * 11
            self.path(f"M{self.width - 126 - offset} 18C{self.width - 180 - offset} 62 "
                      f"{self.width - 40 - offset} 62 {self.width - 78 - offset} 119",
                      "rose", 1, opacity=0.14)
        self.text(margin, 39, f"RELIEF ATLAS  /  {number}  /  {label}", 14, "rose", 700)
        titles = title if isinstance(title, tuple) else (title,)
        self.lines(margin, 82, titles, 30 if compact else 32, "ink", step=36, weight=700)
        self.lines(margin, 82 + len(titles) * 36, subtitle, 18 if compact else 20, step=25)

    def icon(self, kind, x, y, size=48, color="blue"):
        self.parts.append(f'<g transform="translate({x} {y}) scale({size / 48})" aria-hidden="true">')
        if kind == "reference":
            self.rect(1, 1, 46, 46, "url(#alpha)", "line", 6)
            self.path("M14 31V22H29V34H14ZM18 22V16H26V22M29 27H36V18H31", color, 2.5,
                      color + "_bg")
            self.dot(18, 35, 3, color)
            self.dot(28, 35, 3, color)
        elif kind == "voxel":
            self.path("M24 3L44 14V35L24 46L4 35V14ZM4 14L24 25L44 14M24 25V46",
                      color, 2, color + "_bg")
            self.path("M14 9L34 20V41M34 9L14 20V41M4 25L24 36L44 25", color, 1)
        elif kind == "mesh":
            self.path("M10 13L29 5L43 20L38 38L16 44L4 29ZM10 13L24 25L29 5M24 25L43 20"
                      "M24 25L38 38M24 25L16 44M24 25L4 29M10 13L4 29", color, 1.8, color + "_bg")
            for px, py in ((10, 13), (29, 5), (43, 20), (38, 38), (16, 44), (4, 29), (24, 25)):
                self.dot(px, py, 2, color)
        elif kind == "gaussians":
            for px, py, angle, radius in ((13, 13, -25, 7), (27, 9, 30, 6), (37, 20, -30, 7),
                                          (23, 23, 15, 10), (10, 29, 50, 7), (30, 37, -15, 9),
                                          (16, 40, 10, 5), (41, 34, 50, 5)):
                self.parts.append(f'<ellipse cx="{px}" cy="{py}" rx="{radius}" ry="3.5" '
                                  f'transform="rotate({angle} {px} {py})" fill="{self.color(color + "_bg")}" '
                                  f'stroke="{self.color(color)}" stroke-width="1.5"/>')
                self.dot(px, py, 1.4, color)
        elif kind == "shield":
            self.path("M24 4L41 11V24C41 35 32 42 24 46C16 42 7 35 7 24V11Z", color, 2,
                      color + "_bg")
            self.path("M15 24L22 31L34 17", color, 3)
        else:
            self.path("M10 3H29L39 13V45H10ZM29 3V14H39", color, 2, color + "_bg")
            self.path("M16 21H32M16 28H32M16 35H27", color, 2)
        self.parts.append('</g>')

    def finish(self):
        return "\n".join(self.parts + ['</svg>', ''])


def pipeline(theme, compact):
    c = Canvas(480 if compact else 1120, 1328 if compact else 836, theme,
               "One item, one traceable pipeline", DESCRIPTIONS["local-pipeline"])
    if compact:
        c.header("01", "LOCAL PIPELINE", ("One item, one", "traceable pipeline"),
                 ("Reference → shared decode → two outputs.",), True)
        c.panel(24, 176, 432, 112, "rose")
        c.text(48, 210, "Keep one saved selection", 24, weight=700)
        c.lines(48, 240, ("Policy-screened IDs, source prompts,", "recorded provenance and settings."))
        c.route(((240, 288), (240, 316)), "rose")
        c.panel(24, 316, 432, 176)
        c.text(48, 346, "REFERENCE", 14, "blue", 700)
        c.text(48, 382, "Qwen-Image-2.1", 26, weight=700)
        c.lines(48, 418, ("RGBA PNG + image metadata", "Native alpha; matte opaque output."))
        c.icon("reference", 374, 396, 58)
        c.route(((240, 492), (240, 520)))
        c.panel(24, 520, 432, 108, "blue")
        c.icon("shield", 48, 548, 48)
        c.text(116, 560, "Image QA + safety", 24, weight=700)
        c.text(116, 590, "Required before reconstruction", 18, "muted")
        c.route(((240, 628), (240, 656)))
        c.panel(24, 656, 432, 188)
        c.text(48, 686, "RECONSTRUCTION", 14, "blue", 700)
        c.text(48, 722, "TRELLIS.2-4B", 26, weight=700)
        c.lines(48, 758, ("DINOv3 conditioning", "+ trained structure decoder"))
        c.icon("voxel", 374, 734, 58)
        c.text(48, 820, "Decoded mesh + voxel attributes", 19, "blue", 600)
        c.route(((240, 844), (240, 872), (12, 872), (12, 1120)), "blue", arrow=False)
        c.text(48, 892, "TWO REPRESENTATIONS / ONE DECODE", 14, "blue", 700)
        c.route(((12, 970), (24, 970)), "blue")
        c.route(((12, 1120), (24, 1120)), "blue")
        c.panel(24, 912, 432, 128)
        c.text(48, 950, "Textured GLB", 24, weight=700)
        c.text(48, 979, "Texture bake · Y-up", 18, "muted")
        c.icon("mesh", 380, 940, 48)
        c.pill(48, 994, 128, "Mesh QA", "blue", height=32)
        c.panel(24, 1064, 432, 144)
        c.text(48, 1102, "Gaussian container", 24, weight=700)
        c.lines(48, 1132, ("SPZ / SPLAT · native Z-up", "Synthesized from decoded voxels."), step=24)
        c.icon("gaussians", 380, 1094, 48)
        c.pill(246, 1166, 182, "Gaussian QA", "blue", height=30)
        c.text(24, 1250, "Reference + both exports + metadata", 20, weight=600)
        c.lines(24, 1280, ("Next: current evidence and a separate", "release-membership audit."), step=24)
    else:
        c.header("01", "LOCAL PIPELINE", "One item, one traceable pipeline",
                 ("Reference image → shared 3D decode → two deliverables.",))
        c.panel(36, 150, 1048, 102, "rose")
        c.icon("file", 60, 174, 52, "rose")
        c.text(136, 187, "Freeze the permitted selection", 24, weight=700)
        c.text(136, 219, "Item IDs, original prompts, provenance, settings", 19, "muted")
        c.pill(782, 174, 272, "Same scope at every phase", "rose", 18, 48)
        c.route(((198, 252), (198, 288)), "rose")
        c.panel(36, 288, 324, 216)
        c.text(60, 320, "REFERENCE", 14, "blue", 700)
        c.text(60, 357, "Qwen-Image-2.1", 26, weight=700)
        c.icon("reference", 60, 388, 62)
        c.text(140, 408, "RGBA PNG", 22, weight=600)
        c.lines(140, 438, ("Preserve native alpha", "Matte opaque output"), step=25)
        c.route(((360, 396), (390, 396)))
        c.panel(390, 320, 220, 152, "blue")
        c.text(414, 351, "GATE 01", 14, "blue", 700)
        c.text(414, 386, "Image QA", 24, weight=700)
        c.lines(414, 418, ("Framing + alpha", "Safety screen"), step=25)
        c.route(((610, 396), (640, 396)))
        c.panel(640, 288, 444, 216)
        c.text(664, 320, "RECONSTRUCTION", 14, "blue", 700)
        c.text(664, 357, "TRELLIS.2-4B", 26, weight=700)
        c.lines(664, 394, ("DINOv3 conditioning", "+ trained structure decoder"), size=19)
        c.icon("voxel", 998, 372, 62)
        c.rect(660, 448, 404, 38, "blue_bg", radius=8)
        c.text(862, 473, "Decoded mesh + voxel attributes", 20, "blue", 600, anchor="middle")
        c.route(((862, 504), (862, 536), (284, 536), (284, 564)), "blue")
        c.route(((862, 536), (828, 536), (828, 564)), "blue")
        c.dot(862, 536, 4, "blue")
        c.panel(36, 564, 496, 176)
        c.text(60, 595, "TEXTURE BAKE", 14, "blue", 700)
        c.text(60, 630, "Textured GLB", 24, weight=700)
        c.icon("mesh", 60, 653, 52)
        c.lines(132, 674, ("PBR textures", "Y-up coordinates"), size=19)
        c.pill(366, 658, 142, "Mesh QA", "blue", height=44)
        c.panel(572, 564, 512, 176)
        c.text(596, 595, "GAUSSIAN EXPORT", 14, "blue", 700)
        c.text(596, 630, "Gaussian container", 24, weight=700)
        c.icon("gaussians", 596, 653, 52)
        c.lines(668, 674, ("SPZ / SPLAT", "From voxels · Z-up"), size=19)
        c.pill(900, 658, 160, "Gaussian QA", "blue", height=44)
        c.text(36, 782, "PNG + GLB + Gaussian container + metadata", 20, weight=600)
        c.text(36, 811, "Next: current QA evidence and a separate release-membership audit.", 18, "muted")
    return c.finish()


def recovery(theme, compact):
    c = Canvas(480 if compact else 1120, 1512 if compact else 992, theme,
               "Recovery follows saved evidence", DESCRIPTIONS["offline-recovery"])
    if compact:
        c.header("02", "OFFLINE RECOVERY", ("Recovery follows", "saved evidence"),
                 ("The latest saved command wins.",), True)
        for x, width, title, lines, color in (
            (24, 204, "Start / Resume", ("Save running", "intent"), "rose"),
            (240, 102, "Pause", ("Save,", "then stop"), "amber"),
            (354, 102, "Status", ("Read", "only"), "blue"),
        ):
            c.panel(x, 176, width, 112, color)
            c.text(x + 12, 211, title, 20, weight=700)
            c.lines(x + 12, 246, lines, step=24)
        c.rect(24, 312, 432, 88, "rose_bg", radius=12)
        c.lines(44, 340, ("Manual pause survives login.", "Running intent allows crash recovery.", "After reboot, log in to recover."), step=24)
        c.panel(24, 424, 432, 104, "rose")
        c.text(48, 460, "Verify saved plan,", 23, weight=700)
        c.text(48, 490, "artifact hashes + current QA", 23, weight=700)
        c.route(((240, 528), (240, 558)), "rose")
        c.text(24, 588, "First missing requirement", 22, weight=700)
        for index, (saved, detail, next_work) in enumerate(RECOVERY_ROWS):
            y = 608 + index * 144
            c.panel(24, y, 432, 128)
            c.dot(48, y + 31, 10, "rose", "rose_bg")
            c.text(48, y + 36, str(index + 1), 14, "rose", 700, anchor="middle")
            c.text(72, y + 38, saved, 22, weight=700)
            c.text(48, y + 68, detail, 18, "muted")
            c.text(48, y + 104, "→ " + next_work, 20, "teal", 600)
        c.panel(24, 1200, 432, 84, "blue")
        c.text(48, 1232, "Resource waits → automatic recheck", 20, weight=600)
        c.text(48, 1262, "Power / disk / GPU / pipeline", 18, "muted")
        c.panel(24, 1308, 432, 120, "amber")
        c.text(48, 1342, "Needs attention", 23, weight=700)
        c.lines(48, 1370, ("Prerequisites, changed snapshots,", "exhausted retries or quarantine."), step=24)
        c.lines(24, 1466, ("Sampling restarts. Verified files stay.", "Resume rechecks; quarantine is preserved."), step=24)
    else:
        c.header("02", "OFFLINE RECOVERY", "Recovery follows saved evidence",
                 ("Resume checks saved files and runs the first missing stage.",))
        for x, width, title, detail, color in (
            (36, 336, "Start / Resume", "Save running intent", "rose"),
            (392, 336, "Pause", "Save paused intent, then stop", "amber"),
            (748, 336, "Status", "Read saved state only", "blue"),
        ):
            c.panel(x, 152, width, 100, color)
            c.text(x + 24, 190, title, 24, weight=700)
            c.text(x + 24, 222, detail, 18, "muted")
        c.route(((204, 252), (204, 332)), "rose")
        c.pill(392, 272, 336, "Manual pause survives login", "amber", height=36)
        c.route(((560, 252), (560, 272)), "amber")
        c.text(748, 294, "Read-only; no worker is started.", 18, "muted")
        c.panel(36, 332, 1048, 76, "rose")
        c.icon("file", 58, 346, 44, "rose")
        c.text(124, 365, "Check saved plan, hashes and current QA", 24, weight=700)
        c.text(124, 392, "Keep the same saved selection and settings.", 18, "muted")
        c.route(((560, 408), (560, 438)), "rose")
        c.text(72, 464, "Saved evidence", 18, "muted", 600)
        c.text(590, 464, "Next required work", 18, "muted", 600)
        for index, (saved, detail, next_work) in enumerate(RECOVERY_ROWS):
            y = 480 + index * 80
            c.panel(36, y, 1048, 72)
            c.dot(72, y + 35, 14, "rose", "rose_bg")
            c.text(72, y + 41, str(index + 1), 16, "rose", 700, anchor="middle")
            c.text(108, y + 30, saved, 22, weight=700)
            c.text(108, y + 56, detail, 18, "muted")
            c.route(((530, y + 36), (566, y + 36)), "teal")
            c.text(590, y + 42, next_work, 22, "teal", 600)
        for x, title, lines, color in (
            (36, "Resource waits", ("Power / disk / GPU / pipeline", "Automatically recheck resources"), "blue"),
            (392, "Crash / reboot", ("Restart if intent is running", "After reboot: log in first"), "rose"),
            (748, "Needs attention", ("Prerequisites / snapshots", "Exhausted failures / quarantine"), "amber"),
        ):
            c.panel(x, 824, 336, 104, color)
            c.text(x + 20, 857, title, 22, weight=700)
            c.lines(x + 20, 885, lines, step=24)
        c.text(36, 965, "Interrupted sampling restarts. Verified files are reused; quarantine is preserved.", 18, "muted")
    return c.finish()


def evidence(theme, compact):
    c = Canvas(480 if compact else 1120, 1384 if compact else 844, theme,
               "From files to release membership", DESCRIPTIONS["artifact-evidence"])
    if compact:
        c.header("03", "EVIDENCE + RELEASE", ("From files to", "release membership"),
                 ("Three distinct claims. One audit trail.",), True)
        c.panel(24, 184, 432, 320)
        c.text(48, 217, "01 / COMMITTED FILES", 14, "rose", 700)
        c.text(48, 254, "Reference + geometry bundle", 24, weight=700)
        c.text(48, 298, "<id>.png + image_meta.json", 18, mono=True)
        c.text(48, 340, "<id>.glb + Gaussian container", 18, mono=True)
        c.text(48, 374, "metadata.json", 18, mono=True)
        c.lines(48, 426, ("Image metadata: PNG hash + model.", "Geometry: actual source PNG hash.", "Metadata binds files to their source."), step=26)
        c.route(((240, 504), (240, 536)))
        c.panel(24, 536, 432, 272, "blue")
        c.text(48, 569, "02 / CURRENT QA", 14, "blue", 700)
        c.text(48, 606, "Inspect the actual artifacts", 24, weight=700)
        c.lines(48, 650, ("Image: alpha + safety screening", "Mesh: geometry + views", "Gaussians: attributes + views"), 20, "ink", step=34)
        c.text(48, 776, "Current contract + matching file hashes", 18, "blue", 600)
        c.route(((240, 808), (240, 840)))
        c.panel(24, 840, 432, 272, "teal")
        c.text(48, 873, "03 / POSITIVE MEMBERSHIP", 14, "teal", 700)
        c.text(48, 910, "Release requires every rule", 24, weight=700)
        for index, rule in enumerate(RELEASE_RULES):
            y = 949 + index * 32
            c.dot(54, y - 6, 3, "teal")
            c.text(72, y, rule, 20)
        c.route(((240, 1112), (240, 1144)), "teal")
        c.panel(24, 1144, 432, 104, "teal")
        c.text(48, 1182, "release_manifest.json", 21, "teal", 600, mono=True)
        c.lines(48, 1211, ("Included IDs + file hashes.", "A local accounting record."), step=24)
        c.text(24, 1290, "Every saved target must be included.", 20, weight=600)
        c.lines(24, 1324, ("Visual fidelity, scale, collisions and physics", "still need separate review."), step=25)
    else:
        c.header("03", "EVIDENCE + RELEASE", "From files to release membership",
                 ("Bind what was generated to what was actually checked.",))
        for x, label, title, color in (
            (36, "01 / COMMITTED FILES", "Reference + geometry", None),
            (396, "02 / CURRENT QA", "Inspected artifacts", "blue"),
            (756, "03 / RELEASE AUDIT", "Positive membership", "teal"),
        ):
            c.panel(x, 164, 328, 436, color)
            c.text(x + 24, 198, label, 14, color or "rose", 700)
            c.text(x + 24, 236, title, 23, weight=700)
        c.icon("reference", 60, 270, 40)
        c.text(116, 290, "<id>.png", 20, mono=True)
        c.text(60, 336, "image_meta.json", 18, mono=True)
        c.text(60, 365, "PNG hash + model identity", 18, "muted")
        c.path("M60 390H340", "line", 1)
        c.icon("mesh", 60, 412, 38)
        c.icon("gaussians", 112, 412, 38)
        c.text(60, 483, "GLB + Gaussian container", 20, weight=600)
        c.text(60, 519, "metadata.json", 18, mono=True)
        c.lines(60, 554, ("Geometry binds its actual", "source PNG hash."), step=25)
        for index, (title, detail) in enumerate((("Image", "Alpha + safety screen"),
                                                ("Mesh", "Geometry + views"),
                                                ("Gaussians", "Attributes + views"))):
            y = 272 + index * 84
            c.icon("shield", 420, y, 36)
            c.text(474, y + 18, title, 22, weight=600)
            c.text(474, y + 46, detail, 18, "muted")
        c.lines(420, 554, ("Current QA contract", "+ matching inspected hashes"), step=25, color="blue")
        for index, rule in enumerate(RELEASE_RULES):
            y = 290 + index * 54
            c.dot(784, y - 7, 3, "teal")
            c.text(800, y, rule, 20)
        c.text(780, 579, "All criteria must hold.", 18, "teal", 600)
        c.route(((364, 380), (396, 380)))
        c.route(((724, 380), (756, 380)))
        c.route(((920, 600), (920, 638)), "teal")
        c.panel(36, 638, 1048, 104, "teal")
        c.icon("file", 62, 663, 52, "teal")
        c.text(134, 682, "release_manifest.json", 24, "teal", 600, mono=True)
        c.text(134, 716, "Explicit included IDs + file hashes. Written locally.", 20, "muted")
        c.text(36, 788, "The controller completes only when every saved target is included.", 21, weight=600)
        c.text(36, 820, "Visual fidelity, scale, collision quality and physics still need separate review.", 18, "muted")
    return c.finish()


def render_all():
    result = {}
    for name, render in (("local-pipeline", pipeline), ("offline-recovery", recovery), ("artifact-evidence", evidence)):
        for compact in (False, True):
            for theme in THEMES:
                filename = f"{name}{'-compact' if compact else ''}-{theme}.svg"
                result[filename] = render(theme, compact)
    return result


def validate(source):
    root = ET.fromstring(source)
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.tag == namespace + "svg"
    assert root.get("role") == "img" and root.get("aria-labelledby") == "title description"
    assert root.find(namespace + "title") is not None and root.find(namespace + "desc") is not None
    for element in root.iter():
        tag = element.tag.removeprefix(namespace)
        assert tag not in ("script", "foreignObject", "image", "iframe"), tag
        for key, value in element.attrib.items():
            assert not key.lower().startswith("on"), key
            if key.rsplit("}", 1)[-1] == "href":
                assert value.startswith("#"), "External SVG dependency"
            assert "http://" not in value and "https://" not in value, "External SVG dependency"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the committed variants without writing files")
    args = parser.parse_args(argv)
    changed = []
    for name, source in render_all().items():
        validate(source)
        path = OUTPUT / name
        if not path.is_file() or path.read_text(encoding="utf-8") != source:
            changed.append(name)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
    if args.check and changed:
        print("Stale or missing SVGs: " + ", ".join(changed), file=sys.stderr)
        return 1
    print(f"{'Verified' if args.check else 'Rendered'} 12 self-contained SVGs; {len(changed)} changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
