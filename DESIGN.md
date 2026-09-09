# CADraw Design System

## 0. Research Log

- Embedded refs: shortlisted `claude.md`, `notion.md`, and `linear.app.md` from the curated design index; picked `taste-skill` + `claude.md` because a warm paper-like surface fits a hand-drawing workflow while the UI must remain operational and restrained.
- StyleGallery: the local layout guidance supplied the `fixed-sidenav-shell` and `list-detail` spatial patterns; no external fetch was needed for this desktop-only PoC.
- Skipped lanes: lazyweb and Imagen drafts — this is a native Windows tool, not a web marketing surface, and no visual reference was supplied.

## 1. Atmosphere & Identity

CADraw feels like a calm workshop instrument: the paper sketch stays visible, the cleaned geometry is deliberate, and every uncertain value is surfaced instead of hidden. The signature is a warm drawing-board canvas with a thin terracotta verification rail, making "what the computer saw" and "what a person must confirm" visually distinct.

The first release is intentionally not a general CAD editor. It is a focused conversion workbench for one sketch at a time.

## 2. Color

### Palette

| Role | Token | Value | Usage |
|---|---|---:|---|
| Surface / primary | `surface-paper` | `#F5F4ED` | Window background |
| Surface / panel | `surface-ivory` | `#FAF9F5` | Cards and work areas |
| Surface / raised | `surface-white` | `#FFFFFF` | Inputs and preview frame |
| Text / primary | `text-ink` | `#141413` | Headings and values |
| Text / secondary | `text-warm-gray` | `#5E5D59` | Supporting copy |
| Text / tertiary | `text-stone` | `#87867F` | Metadata and hints |
| Border / default | `border-cream` | `#F0EEE6` | Quiet separators |
| Border / strong | `border-warm` | `#E8E6DC` | Panel outlines |
| Accent / primary | `accent-terracotta` | `#C96442` | Load, save, confirmation |
| Accent / hover | `accent-coral` | `#D97757` | Hover and selected states |
| Status / success | `status-success` | `#3F7D58` | Confirmed values |
| Status / warning | `status-warning` | `#B26A2C` | Missing or uncertain values |
| Status / error | `status-error` | `#B53333` | Actionable failures |
| Focus | `focus-blue` | `#3898EC` | Keyboard focus only |

Rules:

- The warm neutral palette is based on the Claude reference, but CADraw uses it for a compact native workbench rather than a marketing page.
- Accent color is reserved for actions, selected geometry, and verification status.
- No pure black or neon colors are used.

## 3. Typography

| Level | Size | Weight | Usage |
|---|---:|---:|---|
| Window title | 22 px | 600 | Main heading |
| Section heading | 16 px | 600 | Panel titles |
| Body | 14 px | 400 | Instructions and rows |
| Body small | 12 px | 400 | Metadata and warnings |
| Numeric | 14 px | 500 | Dimension values and counts |

Font stacks:

- Primary: `Segoe UI`, `system-ui`, sans-serif.
- Numeric: `Cascadia Mono`, `Consolas`, monospace.

Korean copy stays at body size or above. Labels are short and placed above controls instead of being used as placeholders.

## 4. Spacing & Layout

Base unit: 4 px.

| Token | Value | Usage |
|---|---:|---|
| `space-1` | 4 px | Icon and label gap |
| `space-2` | 8 px | Compact row gap |
| `space-3` | 12 px | Input padding |
| `space-4` | 16 px | Panel padding |
| `space-5` | 20 px | Work-area separation |
| `space-6` | 24 px | Major panel separation |

The application uses a bounded `fixed-sidenav-shell`: a fixed top toolbar and status bar surround a two-pane body. The central body owns vertical scrolling; the image and CAD preview each own their own zoom/pan viewport only when that feature is added. In this PoC, the primary body stays within the window and the verification list scrolls.

Desktop layout:

- Left: source image, 55% minimum width.
- Right: normalized preview and verification list, 45% minimum width.
- Below the previews: compact action row and warning strip.

At narrow widths the two panes stack in source order: source image, preview, verification. The first implementation targets Windows desktop, but this order keeps resizing predictable.

## 5. Components

### Workbench shell

- **Structure:** title bar → toolbar → split work area → status bar.
- **Variants:** empty, analyzed, error.
- **Spacing:** `space-4` internal padding; `space-5` between panes.
- **States:** empty prompt, loading, analyzed, export-ready, error.
- **Accessibility:** tab order follows load → recalculate → save → verification fields; all actions have text labels.
- **Motion:** none beyond native button feedback; correctness is more important than decoration.
- **Layout:** `fixed-sidenav-shell` adapted to a two-pane native desktop shell. The verification list is the only scroll owner in the initial window.

### Drawing viewport

- **Structure:** titled frame → image/preview canvas → status caption.
- **Variants:** source, normalized CAD, empty.
- **Spacing:** `space-3` frame padding.
- **States:** empty, rendered, selected entity.
- **Accessibility:** caption identifies the surface; selected entity is also named in the verification list.
- **Motion:** no automatic movement; future zoom/pan is user-driven.
- **Layout:** `frame` with an intrinsic aspect ratio and no hidden overflow of the drawing.

### Verification row

- **Structure:** status marker → label → editable value → unit → confidence hint.
- **Variants:** confirmed, needs review, unknown, unconnected.
- **Spacing:** `space-2` row gap.
- **States:** default, focus, edited, invalid, disabled.
- **Accessibility:** label is associated with the editable field; status is textual, not color-only.
- **Motion:** native focus/press feedback only.
- **Layout:** `stack` inside the verification list.

### Action button

- **Structure:** text label with optional small status marker.
- **Variants:** primary terracotta, neutral warm-sand, disabled.
- **Spacing:** `space-2` horizontal and `space-3` vertical.
- **States:** default, hover, active, focus, disabled.
- **Accessibility:** minimum 40 px height in the desktop UI and visible focus ring.
- **Motion:** 100–150 ms native color/press feedback.
- **Layout:** `cluster` in the toolbar and footer.

## 6. Motion & Interaction

This is a precision utility, so motion intensity is 1. No automatic entrance animation or decorative looping motion is used. Button and focus feedback are delegated to native Qt styling. Recalculation and export update the status bar immediately after the deterministic operation completes.

## 7. Depth & Surface

Strategy: `mixed`, with warm borders for containment and a whisper shadow only on the preview frame. This keeps the workbench readable in a bright shop environment without turning every region into a floating card.

## 8. Accessibility Constraints & Accepted Debt

### Constraints

- Target: WCAG 2.2 AA principles applied to the native surface.
- Body text uses high-contrast warm ink and is not smaller than 12 px.
- Every interactive control is keyboard reachable and has a visible focus state.
- Warning state is communicated with a text label and color, never color alone.
- No confirmation is implied by a high confidence score when a physical dimension was not supplied.

### Accepted Debt

| Item | Location | Why accepted | Owner / Exit |
|---|---|---|---|
| Native Qt widgets are not a full screen-reader audit yet | `src/cadraw/ui/` | Phase 1 prioritizes the working conversion path; labels and tab order are explicit | Validate with Windows Narrator before installer release |
| Zoom/pan is not in the first viewport | `src/cadraw/ui/preview.py` | A clean fit-to-frame view is enough for the first PoC | Add user-driven zoom/pan in Phase 2 |
| No real handwriting OCR in Phase 1 | `src/cadraw/vision.py` | Manual dimensions are safer than guessed fabrication values | Add an adapter and review states in Phase 3 |
