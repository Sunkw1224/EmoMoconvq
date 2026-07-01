// EmoMoconvq spotlight presentation -- minimalist monochrome design.
// Charcoal + neutral grey + one restrained teal-grey accent. No decorative bars.

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

function imageToDataUri(filepath) {
  const ext = path.extname(filepath).slice(1).toLowerCase();
  const mime = ext === "jpg" ? "jpeg" : ext;
  return `image/${mime};base64,` + fs.readFileSync(filepath).toString("base64");
}

const ASSETS = __dirname;
const OUT = path.join(__dirname, "EmoMoconvq_spotlight.pptx");

// --- Palette ---------------------------------------------------------------
const INK      = "1F2937"; // deep charcoal (titles, body emphasis)
const TEXT     = "374151"; // softer charcoal (body text)
const MUTED    = "6B7280"; // mid grey (captions, secondary)
const LINE     = "D1D5DB"; // light grey (dividers)
const BG       = "FFFFFF"; // white background for content slides
const DARKBG   = "1F2937"; // charcoal background for cover + conclusion
const ACCENT   = "5A8A99"; // restrained teal-grey (numbers, key marks)

// --- Fonts -----------------------------------------------------------------
const HEAD = "Georgia";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";   // 10 x 5.625 in
pres.author = "Kangwei Sun";
pres.title  = "EmoMoconvq -- Spotlight Presentation";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function addPageNum(slide, n, total) {
  slide.addText(`${n} / ${total}`, {
    x: 9.1, y: 5.25, w: 0.8, h: 0.25,
    fontFace: BODY, fontSize: 9, color: MUTED, align: "right",
  });
}

function addProjectMark(slide) {
  slide.addText("EmoMoconvq", {
    x: 0.5, y: 5.25, w: 3, h: 0.25,
    fontFace: BODY, fontSize: 9, color: MUTED, align: "left",
    italic: true, margin: 0,
  });
}

function addSlideTitle(slide, title) {
  slide.addText(title, {
    x: 0.5, y: 0.35, w: 9, h: 0.6,
    fontFace: HEAD, fontSize: 26, bold: true, color: INK,
    align: "left", valign: "top", margin: 0,
  });
  slide.addShape(pres.shapes.LINE, {
    x: 0.5, y: 1.05, w: 1.0, h: 0,
    line: { color: ACCENT, width: 1.5 },
  });
}

const TOTAL = 9;

// ===========================================================================
// SLIDE 1 — Title (dark)
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: DARKBG };

  s.addText("On the Decoder Bottleneck in", {
    x: 0.7, y: 1.4, w: 8.6, h: 0.55,
    fontFace: HEAD, fontSize: 22, color: "9CA3AF",
    italic: true, align: "left", margin: 0,
  });
  s.addText("Physics-Based Emotion-Conditioned", {
    x: 0.7, y: 1.95, w: 8.6, h: 0.7,
    fontFace: HEAD, fontSize: 34, bold: true, color: "FFFFFF",
    align: "left", margin: 0,
  });
  s.addText("Motion Generation", {
    x: 0.7, y: 2.6, w: 8.6, h: 0.7,
    fontFace: HEAD, fontSize: 34, bold: true, color: "FFFFFF",
    align: "left", margin: 0,
  });

  // accent divider
  s.addShape(pres.shapes.LINE, {
    x: 0.7, y: 3.55, w: 0.6, h: 0,
    line: { color: ACCENT, width: 1.5 },
  });

  s.addText("EmoMoconvq  —  Spotlight Presentation", {
    x: 0.7, y: 3.75, w: 8.6, h: 0.35,
    fontFace: BODY, fontSize: 14, color: "9CA3AF",
    italic: true, align: "left", margin: 0,
  });
  s.addText("Kangwei Sun   |   524531910006", {
    x: 0.7, y: 4.6, w: 8.6, h: 0.35,
    fontFace: BODY, fontSize: 13, color: "D1D5DB",
    align: "left", margin: 0,
  });
  s.addText("Intelligent Robot — Final Project", {
    x: 0.7, y: 4.95, w: 8.6, h: 0.3,
    fontFace: BODY, fontSize: 11, color: "9CA3AF",
    align: "left", margin: 0,
  });
}

// ===========================================================================
// SLIDE 2 — Problem & Motivation
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSlideTitle(s, "Problem & Motivation");

  s.addText("Physics-based text-to-motion models (MoConVQ) produce dynamically plausible motion but expose only one control: free-text.", {
    x: 0.5, y: 1.35, w: 9.0, h: 0.7,
    fontFace: BODY, fontSize: 14, color: TEXT, align: "left",
  });

  // Two-column comparison: limitation / question
  const colY = 2.25, colH = 2.6, gap = 0.3;
  const colW = (9.0 - gap) / 2;

  // Left column
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: colY, w: colW, h: colH,
    fill: { color: "F9FAFB" }, line: { color: LINE, width: 0.75 },
  });
  s.addText("Limitation", {
    x: 0.7, y: colY + 0.15, w: colW - 0.4, h: 0.35,
    fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0,
  });
  s.addText([
    { text: "Emotion lives in the prompt wording.",
      options: { breakLine: true } },
    { text: "Brittle (exact words matter), entangled (mixes with content), and unreachable for unseen action × emotion pairs.",
      options: {} },
  ], {
    x: 0.7, y: colY + 0.55, w: colW - 0.4, h: colH - 0.7,
    fontFace: BODY, fontSize: 12, color: TEXT, paraSpaceAfter: 6,
  });

  // Right column
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5 + colW + gap, y: colY, w: colW, h: colH,
    fill: { color: "F9FAFB" }, line: { color: LINE, width: 0.75 },
  });
  s.addText("Our question", {
    x: 0.7 + colW + gap, y: colY + 0.15, w: colW - 0.4, h: 0.35,
    fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0,
  });
  s.addText("Can we attach a small discrete emotion interface to MoConVQ so the same prompt can be rendered in different emotional styles by flipping a flag?", {
    x: 0.7 + colW + gap, y: colY + 0.55, w: colW - 0.4, h: colH - 0.7,
    fontFace: BODY, fontSize: 12, color: TEXT, italic: true,
  });

  addProjectMark(s); addPageNum(s, 2, TOTAL);
}

// ===========================================================================
// SLIDE 3 — Background (related work, very compact)
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSlideTitle(s, "Background");

  const rows = [
    ["MoConVQ (Yao et al., 2023)", "Physics-based text-to-motion. Residual-VQ + GPT + MoE tracker over an ODE simulator. We build on it; backbone stays frozen."],
    ["T2M-GPT, MDM",              "Kinematic-only text-to-motion. No physics; no explicit emotion control."],
    ["Style / emotion transfer",  "ACTOR (action labels); Holden, Aberman (style transfer on mocap). Need style-labelled motion data, which we do not have for MoConVQ's skeleton."],
    ["LoRA / adapters",           "Inspiration for parameter-efficient adaptation. Our emotion module is ~0.8M params on a 194M backbone."],
  ];

  const yStart = 1.4, rowH = 0.8;
  rows.forEach((r, i) => {
    const y = yStart + i * rowH;
    s.addText(r[0], {
      x: 0.5, y, w: 2.7, h: rowH - 0.05,
      fontFace: HEAD, fontSize: 13, bold: true, color: INK, valign: "top", margin: 0,
    });
    s.addText(r[1], {
      x: 3.3, y, w: 6.2, h: rowH - 0.05,
      fontFace: BODY, fontSize: 11, color: TEXT, valign: "top",
    });
    if (i < rows.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: 0.5, y: y + rowH - 0.08, w: 9.0, h: 0,
        line: { color: LINE, width: 0.5 },
      });
    }
  });

  addProjectMark(s); addPageNum(s, 3, TOTAL);
}

// ===========================================================================
// SLIDE 4 — Method: architecture
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSlideTitle(s, "Method  —  Architecture");

  s.addText("EmoMoconvq plugs a tiny trainable emotion branch into MoConVQ's frozen  text → tokens → latent → physics  pipeline.", {
    x: 0.5, y: 1.15, w: 9.0, h: 0.35,
    fontFace: BODY, fontSize: 12, color: MUTED, italic: true, margin: 0,
  });

  // ---- Emotion module (top, above the +) --------------------------------
  const plusCx = 2.80;       // center of the (+) operator
  const emoW   = 2.10;
  const emoH   = 0.70;
  const emoX   = plusCx - emoW / 2;
  const emoY   = 1.65;

  s.addShape(pres.shapes.RECTANGLE, {
    x: emoX, y: emoY, w: emoW, h: emoH,
    fill: { color: "EEF2F5" }, line: { color: ACCENT, width: 1.5 },
  });
  s.addText("Emotion label  c", {
    x: emoX + 0.05, y: emoY + 0.05, w: emoW - 0.10, h: 0.28,
    fontFace: HEAD, fontSize: 11, bold: true, color: INK,
    align: "center", margin: 0,
  });
  s.addText("E_e  (5 × 512)   →   MLP_proj  (→ 1024)", {
    x: emoX + 0.05, y: emoY + 0.35, w: emoW - 0.10, h: 0.30,
    fontFace: BODY, fontSize: 10, color: ACCENT, italic: true,
    align: "center", margin: 0,
  });

  // ---- Pipeline band ----------------------------------------------------
  const dy = 2.80, dh = 0.95;
  const cy = dy + dh / 2;     // vertical center for arrows / +

  const makeBox = (x, w, label, sub, isAdapt) => {
    const fill   = isAdapt ? "EEF2F5" : "F9FAFB";
    const border = isAdapt ? ACCENT   : LINE;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: dy, w, h: dh,
      fill: { color: fill }, line: { color: border, width: isAdapt ? 1.5 : 0.75 },
    });
    s.addText(label, {
      x: x + 0.05, y: dy + 0.08, w: w - 0.10, h: 0.42,
      fontFace: HEAD, fontSize: 12, bold: true, color: INK,
      align: "center", valign: "middle", margin: 0,
    });
    if (sub) {
      s.addText(sub, {
        x: x + 0.05, y: dy + 0.50, w: w - 0.10, h: 0.40,
        fontFace: BODY, fontSize: 9, color: MUTED, italic: true,
        align: "center", valign: "top", margin: 0,
      });
    }
  };

  const arrow = (x1, x2) => {
    s.addShape(pres.shapes.LINE, {
      x: x1, y: cy, w: x2 - x1, h: 0,
      line: { color: TEXT, width: 1.0, endArrowType: "triangle" },
    });
  };

  // "text →" label on the left
  s.addText("text  →", {
    x: 0.20, y: cy - 0.18, w: 0.85, h: 0.36,
    fontFace: BODY, fontSize: 11, color: TEXT,
    align: "right", valign: "middle", margin: 0,
  });

  // Box 1: T5-large (frozen)
  makeBox(1.10, 1.40, "T5-large", "encoder · frozen", false);
  arrow(2.50, 2.60);

  // (+) operator
  s.addShape(pres.shapes.OVAL, {
    x: 2.60, y: cy - 0.20, w: 0.40, h: 0.40,
    fill: { color: "FFFFFF" }, line: { color: ACCENT, width: 1.5 },
  });
  s.addText("+", {
    x: 2.60, y: cy - 0.24, w: 0.40, h: 0.40,
    fontFace: HEAD, fontSize: 18, bold: true, color: ACCENT,
    align: "center", valign: "middle", margin: 0,
  });
  arrow(3.00, 3.15);

  // Box 2: T2M-MoConGPT (emotion injected here; trainable / top-K tuned)
  makeBox(3.15, 2.05, "T2M-MoConGPT", "12-layer autoreg. transformer · top-K tuned · emits RQ tokens", true);
  arrow(5.20, 5.40);

  // Box 3: ConvVQ decoder (frozen)
  makeBox(5.40, 1.65, "ConvVQ decoder", "RQ tokens → motion latent  z", false);
  arrow(7.05, 7.25);

  // Box 4: Physics decoder (frozen)
  makeBox(7.25, 1.90, "Physics decoder", "MoE tracker + ODE simulator", false);

  // "→ BVH" on the right
  s.addText("→  BVH", {
    x: 9.18, y: cy - 0.18, w: 0.80, h: 0.36,
    fontFace: BODY, fontSize: 11, color: TEXT,
    align: "left", valign: "middle", margin: 0,
  });

  // Arrow down from emotion module to (+)
  s.addShape(pres.shapes.LINE, {
    x: plusCx, y: emoY + emoH, w: 0,
    h: (cy - 0.20) - (emoY + emoH),
    line: { color: ACCENT, width: 1.5, endArrowType: "triangle" },
  });

  // ---- MoConVQ pipeline walk-through caption ---------------------------
  s.addText([
    { text: "MoConVQ pipeline (everything outside the teal box is frozen): ", options: { bold: true, color: INK } },
    { text: "T5 encodes the prompt; ", options: {} },
    { text: "T2M-MoConGPT ", options: { bold: true, color: ACCENT } },
    { text: "autoregressively emits a residual-VQ token sequence; ", options: {} },
    { text: "ConvVQ decodes those tokens to a continuous motion latent ", options: {} },
    { text: "z", options: { italic: true } },
    { text: "; the MoE tracking policy then drives an ODE simulator that follows ", options: {} },
    { text: "z", options: { italic: true } },
    { text: " and outputs physically valid BVH motion.", options: {} },
  ], {
    x: 0.5, y: 3.92, w: 9.0, h: 0.62,
    fontFace: BODY, fontSize: 10, color: TEXT, valign: "top", margin: 0,
  });

  // ---- Fusion equation + zero-init note --------------------------------
  s.addText("f_cond  =  f_T5  +  MLP_proj( E_e (c) )", {
    x: 0.5, y: 4.65, w: 9.0, h: 0.35,
    fontFace: "Consolas", fontSize: 14, color: INK,
    align: "center", margin: 0,
  });
  s.addText("0.8 M trainable emotion params  •  zero-init last linear → step 0 = baseline (graceful degradation)", {
    x: 0.5, y: 5.00, w: 9.0, h: 0.25,
    fontFace: BODY, fontSize: 10, color: MUTED, italic: true,
    align: "center", margin: 0,
  });

  addProjectMark(s); addPageNum(s, 4, TOTAL);
}

// ===========================================================================
// SLIDE 5 — Method: progressive experimental design
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSlideTitle(s, "Method  —  Progressive distillation");

  s.addText("Distil emotion knowledge by running the baseline on text with emotion adverbs; train EmoMoconvq from stripped text + label. Tighten the design at each step until the label is the only useful signal.", {
    x: 0.5, y: 1.20, w: 9.0, h: 0.65,
    fontFace: BODY, fontSize: 12, color: TEXT, italic: true,
  });

  const cards = [
    {
      tag: "v1", name: "Uncontrolled",
      detail: "463 emotion-bearing captions  •  one emotion / caption  •  top-4 layers tuned",
      params: "47.3M (24.3%)", risk: "Text can memorise; label may be ignored",
    },
    {
      tag: "v2", name: "5-variant compositional",
      detail: "100 base × 5 emotions = 500  •  base-stratified split  •  top-2 layers",
      params: "24.0M (12.4%)", risk: "Same text forces label use",
    },
    {
      tag: "B",  name: "Emotion-module only",
      detail: "v2 dataset  •  every temporal layer frozen  •  only E_e + MLP_proj",
      params: "0.79M (0.41%)", risk: "Module is the only thing that can learn",
    },
    {
      tag: "D'", name: "3-class focus",
      detail: "{neutral, happy, fearful}  •  300 samples  •  pre-registered fallback",
      params: "3-class  •  chance 33.3%", risk: "Drop classes baseline confuses",
    },
  ];

  const yT = 2.05, h = 0.78, gap = 0.10;
  cards.forEach((c, i) => {
    const y = yT + i * (h + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9.0, h, fill: { color: BG }, line: { color: LINE, width: 0.75 },
    });
    // left tag column
    s.addText(c.tag, {
      x: 0.5, y, w: 0.85, h, margin: 0,
      fontFace: HEAD, fontSize: 18, bold: true, color: ACCENT,
      align: "center", valign: "middle",
    });
    // vertical divider
    s.addShape(pres.shapes.LINE, {
      x: 1.35, y: y + 0.10, w: 0, h: h - 0.20,
      line: { color: LINE, width: 0.5 },
    });
    // content
    s.addText(c.name, {
      x: 1.50, y: y + 0.06, w: 4.5, h: 0.35,
      fontFace: HEAD, fontSize: 13, bold: true, color: INK, margin: 0,
    });
    s.addText(c.detail, {
      x: 1.50, y: y + 0.40, w: 4.7, h: 0.34,
      fontFace: BODY, fontSize: 10.5, color: TEXT, margin: 0,
    });
    s.addText(c.params, {
      x: 6.3, y: y + 0.06, w: 3.1, h: 0.35,
      fontFace: BODY, fontSize: 11, bold: true, color: INK,
      align: "right", margin: 0,
    });
    s.addText(c.risk, {
      x: 6.3, y: y + 0.40, w: 3.1, h: 0.34,
      fontFace: BODY, fontSize: 10, color: MUTED, italic: true,
      align: "right", margin: 0,
    });
  });

  addProjectMark(s); addPageNum(s, 5, TOTAL);
}

// ===========================================================================
// SLIDE 6 — Result A: Token-level evaluation
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSlideTitle(s, "Result A  —  Token-level evaluation");

  // Left: explanation
  s.addText("On 50 held-out neutral captions, EmoMoconvq's outputs differ across emotion labels by 1.36× baseline stochasticity, and a 3-class classifier picks up the conditioned emotion above chance.", {
    x: 0.5, y: 1.25, w: 4.2, h: 1.30,
    fontFace: BODY, fontSize: 12, color: TEXT,
  });

  // Stat blocks
  const sx = 0.5, sy = 2.7, sw = 4.2, gv = 0.2;
  const statH = 1.20;
  // Stat 1
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx, y: sy, w: sw, h: statH,
    fill: { color: "F9FAFB" }, line: { color: LINE, width: 0.75 },
  });
  s.addText("1.36×", {
    x: sx + 0.15, y: sy + 0.10, w: 1.8, h: 0.7,
    fontFace: HEAD, fontSize: 40, bold: true, color: ACCENT,
    valign: "middle", margin: 0,
  });
  s.addText("Token disagreement vs.\nbaseline noise (Path B)", {
    x: sx + 1.95, y: sy + 0.15, w: sw - 2.0, h: 0.9,
    fontFace: BODY, fontSize: 11, color: TEXT, valign: "middle", margin: 0,
  });

  // Stat 2
  const sy2 = sy + statH + gv;
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx, y: sy2, w: sw, h: statH,
    fill: { color: "F9FAFB" }, line: { color: LINE, width: 0.75 },
  });
  s.addText("39.3 %", {
    x: sx + 0.15, y: sy2 + 0.10, w: 1.8, h: 0.7,
    fontFace: HEAD, fontSize: 36, bold: true, color: ACCENT,
    valign: "middle", margin: 0,
  });
  s.addText("3-class classifier acc.\nchance 33.3% — happy 50%", {
    x: sx + 1.95, y: sy2 + 0.15, w: sw - 2.0, h: 0.9,
    fontFace: BODY, fontSize: 11, color: TEXT, valign: "middle", margin: 0,
  });

  // Right: progression chart
  s.addImage({
    path: path.join(ASSETS, "fig_progression.png"),
    x: 5.0, y: 1.25, w: 4.7, h: 2.65,
  });
  s.addText("v1 → v2 → B → D'  :  tightening the design steadily increases the emotion-induced variation.", {
    x: 5.0, y: 3.95, w: 4.7, h: 0.5,
    fontFace: BODY, fontSize: 10, color: MUTED, italic: true, align: "left",
  });

  addProjectMark(s); addPageNum(s, 6, TOTAL);
}

// ===========================================================================
// SLIDE 7 — Result B: kinematic null result + decoder bottleneck
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSlideTitle(s, "Result B  —  Motion-level: null result");

  // Top finding
  s.addText("Decoded through MoConVQ's physics simulator, 6 affective kinematic features show no significant per-emotion effect.", {
    x: 0.5, y: 1.20, w: 9.0, h: 0.40,
    fontFace: BODY, fontSize: 13, color: INK, bold: true,
  });
  s.addText("One-way ANOVA across 20 captions × 3 emotions  —  all  p > 0.7,  all  η² ≤ 0.011.", {
    x: 0.5, y: 1.58, w: 9.0, h: 0.30,
    fontFace: BODY, fontSize: 11, color: MUTED, italic: true,
  });

  // Left: ANOVA box plots
  s.addImage({
    path: path.join(ASSETS, "fig_kinematic_anova.png"),
    x: 0.5, y: 2.05, w: 5.6, h: 3.0,
  });

  // Right: diagnosis card
  const cx = 6.3, cy = 2.05, cw = 3.4, ch = 3.0;
  s.addShape(pres.shapes.RECTANGLE, {
    x: cx, y: cy, w: cw, h: ch,
    fill: { color: "1F2937" }, line: { color: "1F2937", width: 0 },
  });
  s.addText("Diagnosis", {
    x: cx + 0.20, y: cy + 0.15, w: cw - 0.4, h: 0.35,
    fontFace: HEAD, fontSize: 14, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("Decoder bottleneck.", {
    x: cx + 0.20, y: cy + 0.55, w: cw - 0.4, h: 0.32,
    fontFace: HEAD, fontSize: 13, italic: true, color: ACCENT, margin: 0,
  });
  s.addText("MoConVQ's tracking policy was trained to follow latents within a fixed physical manifold. The emotion-conditioned signal lives in the latent but is projected back to the tracker's prior — emotion is normalised out of the physical motion.", {
    x: cx + 0.20, y: cy + 0.95, w: cw - 0.4, h: ch - 1.1,
    fontFace: BODY, fontSize: 10.5, color: "E5E7EB", valign: "top",
  });

  addProjectMark(s); addPageNum(s, 7, TOTAL);
}

// ===========================================================================
// SLIDE 8 — Qualitative demo & interactive system
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSlideTitle(s, "Qualitative output & live demo");

  // Left: embedded comparison video (click to play in PowerPoint)
  s.addMedia({
    type: "video",
    path: path.join(ASSETS, "cap008_compare.mp4"),
    cover: imageToDataUri(path.join(ASSETS, "frame_cap008.png")),
    x: 0.5, y: 1.30, w: 5.5, h: 3.10,
  });
  s.addText("3-emotion comparison on \"a person runs forward\"  —  left: neutral · middle: happy · right: fearful   (click to play)", {
    x: 0.5, y: 4.45, w: 5.5, h: 0.45,
    fontFace: BODY, fontSize: 10, color: MUTED, italic: true,
  });

  // Right: text + demo card
  const rx = 6.3, ry = 1.30, rw = 3.4;
  s.addText([
    { text: "Released artefacts", options: { breakLine: true, bold: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "5 × 3 side-by-side comparison videos", options: { breakLine: true } },
    { text: "3 trained EmoMoconvq checkpoints", options: { breakLine: true } },
    { text: "v1 + v2 distillation datasets", options: { breakLine: true } },
    { text: "Gradio interactive web demo", options: {} },
  ], {
    x: rx, y: ry, w: rw, h: 1.85,
    fontFace: BODY, fontSize: 11.5, color: TEXT, valign: "top",
  });

  // Demo callout box
  s.addShape(pres.shapes.RECTANGLE, {
    x: rx, y: ry + 2.0, w: rw, h: 1.4,
    fill: { color: "F9FAFB" }, line: { color: ACCENT, width: 1.0 },
  });
  s.addText("LIVE DEMO", {
    x: rx + 0.15, y: ry + 2.05, w: rw - 0.3, h: 0.28,
    fontFace: HEAD, fontSize: 10, bold: true, color: ACCENT,
    charSpacing: 3, margin: 0,
  });
  s.addText("Prompt + emotion → physics-simulated BVH in ~10s.", {
    x: rx + 0.15, y: ry + 2.35, w: rw - 0.3, h: 0.55,
    fontFace: BODY, fontSize: 11, color: INK, margin: 0,
  });
  s.addText("Script/gradio_demo.py", {
    x: rx + 0.15, y: ry + 2.9, w: rw - 0.3, h: 0.30,
    fontFace: "Consolas", fontSize: 10, color: MUTED, margin: 0,
  });

  s.addText("github.com/Sunkw1224/EmoMoconvq", {
    x: rx, y: 4.55, w: rw, h: 0.30,
    fontFace: "Consolas", fontSize: 10, color: MUTED, italic: true, margin: 0,
  });

  addProjectMark(s); addPageNum(s, 8, TOTAL);
}

// ===========================================================================
// SLIDE 9 — Conclusion + future work (dark)
// ===========================================================================
{
  const s = pres.addSlide();
  s.background = { color: DARKBG };

  s.addText("Takeaway", {
    x: 0.7, y: 0.55, w: 9.0, h: 0.5,
    fontFace: HEAD, fontSize: 28, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addShape(pres.shapes.LINE, {
    x: 0.7, y: 1.15, w: 0.6, h: 0,
    line: { color: ACCENT, width: 1.5 },
  });

  // Headline
  s.addText("The emotion module learns. The frozen tracker erases what it learnt.", {
    x: 0.7, y: 1.40, w: 8.8, h: 0.7,
    fontFace: HEAD, fontSize: 18, italic: true, color: ACCENT, margin: 0,
  });

  // Two columns: What we found  |  Where it goes next
  const lx = 0.7, ly = 2.30, colW = 4.2, colH = 2.5, lgap = 0.3;

  s.addText("What we found", {
    x: lx, y: ly, w: colW, h: 0.30,
    fontFace: HEAD, fontSize: 14, bold: true, color: "FFFFFF",
    charSpacing: 1, margin: 0,
  });
  s.addText([
    { text: "Token-level emotion control is real (1.36×; happy 50%).",
      options: { bullet: true, breakLine: true, color: "E5E7EB" } },
    { text: "Motion-level effect is null (ANOVA p > 0.7).",
      options: { bullet: true, breakLine: true, color: "E5E7EB" } },
    { text: "Bottleneck localised to the frozen physics tracker.",
      options: { bullet: true, color: "E5E7EB" } },
  ], {
    x: lx, y: ly + 0.40, w: colW, h: colH - 0.4,
    fontFace: BODY, fontSize: 12, color: "E5E7EB", paraSpaceAfter: 6,
  });

  const rx = lx + colW + lgap;
  s.addText("Future directions", {
    x: rx, y: ly, w: colW, h: 0.30,
    fontFace: HEAD, fontSize: 14, bold: true, color: "FFFFFF",
    charSpacing: 1, margin: 0,
  });
  s.addText([
    { text: "Joint fine-tuning of the tracker (differentiable physics).",
      options: { bullet: true, breakLine: true, color: "E5E7EB" } },
    { text: "Emotion-preservation loss in the tracking objective.",
      options: { bullet: true, breakLine: true, color: "E5E7EB" } },
    { text: "End-to-end training on real emotion-labelled motion (needs retargeter).",
      options: { bullet: true, color: "E5E7EB" } },
  ], {
    x: rx, y: ly + 0.40, w: colW, h: colH - 0.4,
    fontFace: BODY, fontSize: 12, color: "E5E7EB", paraSpaceAfter: 6,
  });

  // Footer
  s.addText("Thank you.", {
    x: 0.7, y: 5.05, w: 4, h: 0.35,
    fontFace: HEAD, fontSize: 14, italic: true, color: "FFFFFF", margin: 0,
  });
  s.addText("github.com/Sunkw1224/EmoMoconvq", {
    x: 5, y: 5.05, w: 4.5, h: 0.35,
    fontFace: "Consolas", fontSize: 10, color: "9CA3AF",
    align: "right", margin: 0,
  });
}

// ---------------------------------------------------------------------------
pres.writeFile({ fileName: OUT }).then(p => {
  console.log("[ok] wrote", p);
});
