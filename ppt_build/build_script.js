// EmoMoconvq spotlight 演讲稿 — 中文 Word 文档
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
  HeadingLevel, LevelFormat, BorderStyle, ShadingType,
  Table, TableRow, TableCell, WidthType, PageBreak,
  Header, Footer, PageNumber, TabStopType, TabStopPosition,
} = require("docx");

// ===== style constants ====================================================
const FONT  = "Microsoft YaHei";           // Chinese-safe sans
const FONT_EN = "Calibri";                  // English/numbers
const INK   = "1F2937";
const MUTED = "6B7280";
const ACCENT = "5A8A99";
const LIGHTBG = "F3F4F6";

// run helpers --------------------------------------------------------------
const run = (text, opts = {}) => new TextRun({
  text, font: FONT, size: opts.size ?? 22, color: opts.color ?? INK,
  bold: opts.bold ?? false, italics: opts.italics ?? false,
  highlight: opts.highlight, ...opts,
});
const runEn = (text, opts = {}) => new TextRun({
  text, font: FONT_EN, size: opts.size ?? 22, color: opts.color ?? INK,
  bold: opts.bold ?? false, italics: opts.italics ?? false,
  ...opts,
});

// section divider (page break + heading) -----------------------------------
function slideHeader(num, title, seconds) {
  // colored thin top rule
  const ruler = new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 1 } },
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "" })],
  });
  // title row: "Slide N · Title" + right-aligned time budget
  const titlePara = new Paragraph({
    spacing: { before: 160, after: 100 },
    tabStops: [{ type: TabStopType.RIGHT, position: 9360 }],
    children: [
      run(`第 ${num} 页 · ${title}`, { bold: true, size: 28, color: INK }),
      new TextRun({ text: "\t" }),
      runEn(`⏱ ${seconds} 秒`, { color: ACCENT, bold: true, size: 22 }),
    ],
  });
  return [ruler, titlePara];
}

// speech paragraph (main body) ---------------------------------------------
const speech = (parts) => new Paragraph({
  spacing: { before: 60, after: 60, line: 360 },
  indent: { left: 240 },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 12 } },
  children: parts,
});

// delivery note ("舞台提示") -----------------------------------------------
const stageNote = (text) => new Paragraph({
  spacing: { before: 100, after: 80 },
  shading: { type: ShadingType.CLEAR, fill: LIGHTBG },
  children: [
    run("【舞台提示】 ", { bold: true, size: 18, color: ACCENT }),
    run(text, { size: 18, color: MUTED, italics: true }),
  ],
});

// blank line ---------------------------------------------------------------
const blank = () => new Paragraph({ spacing: { before: 80, after: 80 }, children: [new TextRun("")] });

// ===========================================================================
// Build document content
// ===========================================================================
const children = [];

// --- COVER -----------------------------------------------------------------
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 1200, after: 200 },
  children: [run("EmoMoconvq", { bold: true, size: 56, color: INK })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 600 },
  children: [run("Spotlight 演讲稿（中文，5 分钟）", { italics: true, size: 28, color: MUTED })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 80 },
  children: [run("配套 PPT：EmoMoconvq_spotlight.pptx", { size: 22, color: MUTED })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 600 },
  children: [run("孙康玮 · 524531910006", { size: 22, color: INK })],
}));

// box with usage instructions
children.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: LIGHTBG },
  spacing: { before: 200, after: 100, line: 340 },
  children: [run("使用说明", { bold: true, size: 24, color: INK })],
}));
children.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: LIGHTBG },
  spacing: { after: 100, line: 340 },
  children: [
    run("• 每一段对应一张幻灯片，括号里是时间预算。", { size: 20, color: INK }),
  ],
}));
children.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: LIGHTBG },
  spacing: { after: 100, line: 340 },
  children: [
    run("• 蓝色竖条段落是【正式台词】，可直接照念。", { size: 20, color: INK }),
  ],
}));
children.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: LIGHTBG },
  spacing: { after: 100, line: 340 },
  children: [
    run("• 加粗 + 青灰色字 = 重音 / 关键数字，念慢一拍。", { size: 20, color: INK }),
  ],
}));
children.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: LIGHTBG },
  spacing: { after: 100, line: 340 },
  children: [
    run("• 灰底【舞台提示】是动作指令（指屏幕、停顿、切片等），不要念出口。", { size: 20, color: INK }),
  ],
}));
children.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: LIGHTBG },
  spacing: { after: 200, line: 340 },
  children: [
    run("• 总时长约 5 分 50 秒；现场比稿子快 5–10%，落到约 5 分 15 秒，刚好 Spotlight 时长。", { size: 20, color: INK }),
  ],
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// --- SLIDE 1 ---------------------------------------------------------------
children.push(...slideHeader(1, "开场", 10));
children.push(speech([
  run("各位老师、同学好。我是孙康玮。我汇报的题目是 ", { size: 22 }),
  run("《物理动作生成中情绪条件化的解码瓶颈》", { bold: true, color: ACCENT, size: 22 }),
  run("。", { size: 22 }),
]));
children.push(stageNote("语气平稳，停顿 1 秒进入下一页。第一句话决定全场印象，慢、清晰。"));

// --- SLIDE 2 ---------------------------------------------------------------
children.push(...slideHeader(2, "问题与动机", 30));
children.push(speech([
  run("这里我们做的是物理仿真驱动的文本到动作生成。当前最强的方法 MoConVQ，可以从文本生成物理上合理的动作。", { size: 22 }),
]));
children.push(speech([
  run("但它有一个控制上的缺陷：", { size: 22 }),
  run("唯一的输入是自由文本。", { bold: true, color: ACCENT, size: 22 }),
  run("要让动画角色“带情绪”，只能改 prompt 的措辞——这既脆弱、又把情绪和动作语义混在一起，更没办法对训练里没见过的“动作 × 情绪”组合下指令。", { size: 22 }),
]));
children.push(speech([
  run("我们的问题很简单：", { size: 22 }),
  run("能不能给它加一个离散的情绪开关，让同一个 prompt 通过切换标签产生不同情绪风格的动作？", { bold: true, color: INK, size: 22 }),
]));
children.push(stageNote("讲到“离散的情绪开关”时，可右手做一个“拨开关”手势。"));

// --- SLIDE 3 ---------------------------------------------------------------
children.push(...slideHeader(3, "相关工作", 25));
children.push(speech([
  run("三句话扫一下相关工作。", { size: 22 }),
]));
children.push(speech([
  run("我们以 ", { size: 22 }),
  runEn("MoConVQ", { bold: true, size: 22 }),
  run(" 为骨干，整个骨干都冻结。", { size: 22 }),
]));
children.push(speech([
  runEn("T2M-GPT", { size: 22 }),
  run(" 和 ", { size: 22 }),
  runEn("MDM", { size: 22 }),
  run(" 不是物理驱动的，也没有显式情绪控制。", { size: 22 }),
]));
children.push(speech([
  run("风格迁移类工作需要风格标注的真实 mocap 数据——而 MoConVQ 是定制骨架，没有现成数据集，所以这条路被堵死了。", { size: 22 }),
]));
children.push(speech([
  run("参数高效微调方面我们借鉴了 ", { size: 22 }),
  runEn("LoRA / adapter", { bold: true, size: 22 }),
  run(" 的思路——我们的情绪模块只有 0.8M 参数，相比 194M 主干微不足道。", { size: 22 }),
]));
children.push(stageNote("节奏要快。这一页讲完不停顿，直接切下一页。"));

// --- SLIDE 4 ---------------------------------------------------------------
children.push(...slideHeader(4, "方法：模型架构", 75));
children.push(speech([
  run("我们先把图上从左到右这条管线讲清楚——它就是 MoConVQ 完整的生成流程，所有灰色块都是", { size: 22 }),
  run("冻结的", { bold: true, color: ACCENT, size: 22 }),
  run("，唯一会被训练的是上面 teal 描边的两个块。", { size: 22 }),
]));

// ----- Pipeline walk-through (4 stages) -----
children.push(speech([
  run("第一格 ", { size: 22 }),
  runEn("T5-large", { bold: true, size: 22 }),
  run("：把 prompt 编码成一段 1024 维的语义向量序列——把自然语言翻译成机器能用的语义。这一层是冻结的。", { size: 22 }),
]));
children.push(speech([
  run("第二格 ", { size: 22 }),
  runEn("T2M-MoConGPT", { bold: true, color: ACCENT, size: 22 }),
  run("：一个 12 层的自回归 transformer，用", { size: 22 }),
  run("交叉注意力", { bold: true, size: 22 }),
  run("读 T5 的语义特征，", { size: 22 }),
  run("逐时间步吐出离散的 RQ token", { bold: true, size: 22 }),
  run("。注意是 ", { size: 22 }),
  runEn("residual VQ", { italics: true, size: 22 }),
  run("——每个时间步同时生成若干层残差码本索引，比单层 VQ 表达细很多。", { size: 22 }),
]));
children.push(speech([
  run("这里也正是我们", { size: 22 }),
  run("注入情绪的位置", { bold: true, color: ACCENT, size: 22 }),
  run("——在 T5 输出进入 MoConGPT 之前，把情绪向量加上去。", { size: 22 }),
]));
children.push(speech([
  run("第三格 ", { size: 22 }),
  runEn("ConvVQ decoder", { bold: true, size: 22 }),
  run("：拿这些离散 token 去码本里查向量，残差层相加后用 1D 卷积上采样，得到一个", { size: 22 }),
  run("连续的动作潜变量 z", { bold: true, size: 22 }),
  run("。注意 z 还不是关节角，它是给物理控制器使用的中间表征。", { size: 22 }),
]));
children.push(speech([
  run("第四格 ", { size: 22 }),
  runEn("Physics decoder", { bold: true, size: 22 }),
  run(" 是整套系统的灵魂：内部是一个", { size: 22 }),
  run("MoE 跟踪策略", { bold: true, size: 22 }),
  run("，吃当前角色的物理状态和目标 z，输出每个关节的", { size: 22 }),
  run("力矩", { bold: true, color: ACCENT, size: 22 }),
  run("；力矩送进 ODE 物理仿真器按牛顿力学积分一步——脚不会穿地，摔倒会真摔。一帧一帧跑下去就得到标准的 BVH 动作。", { size: 22 }),
]));

// ----- Our emotion module -----
children.push(speech([
  run("回到我们加的小模块：", { size: 22 }),
  run("一张 5 × 512 的情绪嵌入表", { bold: true, size: 22 }),
  run(" 加一个两层 MLP，把情绪向量投到 1024 维直接", { size: 22 }),
  run("加到文本特征上", { bold: true, color: ACCENT, size: 22 }),
  run("。公式就一行：", { size: 22 }),
  runEn(" f_cond = f_T5 + MLP_proj(E_e(c)) ", { bold: true, font: "Consolas", size: 22 }),
  run("。", { size: 22 }),
]));
children.push(speech([
  run("一个", { size: 22 }),
  run("关键设计", { bold: true, color: ACCENT, size: 22 }),
  run("：MLP 最后一层", { size: 22 }),
  run("零初始化", { bold: true, size: 22 }),
  run("——训练第 0 步加性项为 0，模型完全等价于预训练基线，", { size: 22 }),
  run("优雅退化保护。", { bold: true, size: 22 }),
]));
children.push(stageNote("依次指四个盒子（T5 → MoConGPT → ConvVQ → Physics），节奏要稳；讲到“注入情绪的位置”指 + 号。若总时长压力大，可把第三、第四格各压成一句。指公式时停半拍。"));

// --- SLIDE 5 ---------------------------------------------------------------
children.push(...slideHeader(5, "渐进式蒸馏设计", 40));
children.push(speech([
  run("没有现成的“情绪 + 物理动作”数据集，我们走", { size: 22 }),
  run("蒸馏", { bold: true, color: ACCENT, size: 22 }),
  run("：让基线模型对带情绪副词的文本（比如 ", { size: 22 }),
  runEn("walks happily", { italics: true, size: 22 }),
  run("）生成 token，然后训练 EmoMoconvq——", { size: 22 }),
  run("用剥离情绪词的文本 + 离散标签", { bold: true, size: 22 }),
  run("——去复现这些 token。", { size: 22 }),
]));
children.push(speech([
  run("我们做了", { size: 22 }),
  run("四轮逐步收紧", { bold: true, color: ACCENT, size: 22 }),
  run("的实验。", { size: 22 }),
]));
children.push(speech([
  runEn("v1", { bold: true, size: 22 }),
  run(" 是宽松版，模型可能根本不用情绪标签；", { size: 22 }),
]));
children.push(speech([
  runEn("v2", { bold: true, size: 22 }),
  run(" 把同一 base 配 5 种情绪，", { size: 22 }),
  run("逼模型必须用标签", { bold: true, size: 22 }),
  run("；", { size: 22 }),
]));
children.push(speech([
  runEn("Path B", { bold: true, size: 22 }),
  run(" 把所有 temporal 层也冻结，", { size: 22 }),
  run("只剩情绪模块自己", { bold: true, size: 22 }),
  run("学；", { size: 22 }),
]));
children.push(speech([
  run("最后我们做了 ", { size: 22 }),
  runEn("3 类聚焦 D'", { bold: true, size: 22 }),
  run("，把最难学的两类拿掉。", { size: 22 }),
]));
children.push(stageNote("讲 v1 → v2 → B → D' 时可右手依次比四个手指，建立“渐进收紧”的视觉印象。"));

// --- SLIDE 6 ---------------------------------------------------------------
children.push(...slideHeader(6, "结果 A · Token 层级评估", 50));
children.push(speech([
  run("Token 层级的结果是", { size: 22 }),
  run("正向的", { bold: true, color: ACCENT, size: 22 }),
  run("。", { size: 22 }),
]));
children.push(speech([
  run("我们在 50 条", { size: 22 }),
  run("未见过", { bold: true, size: 22 }),
  run("的、不含任何情绪词的测试 caption 上做组合泛化评估。", { size: 22 }),
]));
children.push(speech([
  run("第一个数字：", { size: 22 }),
  runEn("1.36 倍", { bold: true, color: ACCENT, size: 26 }),
  run("——情绪标签产生的输出变化，是基线随机噪声的 1.36 倍。", { size: 22 }),
  run("情绪标签真的在驱动生成。", { bold: true, size: 22 }),
]));
children.push(speech([
  run("第二个数字：在 3 类设置下分类器准确率 ", { size: 22 }),
  runEn("39.3%", { bold: true, color: ACCENT, size: 26 }),
  run("，随机线只有 33.3%；其中 ", { size: 22 }),
  runEn("happy 类达到 50%", { bold: true, size: 22 }),
  run("，被识别得最清楚。", { size: 22 }),
]));
children.push(speech([
  run("右边这张图你能看到趋势——v1 还在噪声水平，越收紧实验设计，情绪信号越强。", { size: 22 }),
  run("实验设计能放大信号。", { bold: true, color: ACCENT, size: 22 }),
]));
children.push(stageNote("两个大数字念慢、念清楚。指着柱状图“从左到右涨上去”做手势。这页是“胜利时刻”。"));

// --- SLIDE 7 ---------------------------------------------------------------
children.push(...slideHeader(7, "结果 B · Motion 层级 — Decoder Bottleneck ⭐", 60));
children.push(speech([
  run("但是——", { bold: true, color: ACCENT, size: 26 }),
  run("故事到这里转折。", { bold: true, size: 22 }),
]));
children.push(speech([
  run("把同样的输出通过 MoConVQ 的物理解码器跑出来，我们计算了 6 个情感运动学文献支持的特征：垂直跳跃能量、步频、躯干倾角、手臂摆幅、运动速度、头部高度。", { size: 22 }),
]));
children.push(speech([
  run("做单因素方差分析。", { size: 22 }),
  run("全军覆没", { bold: true, color: ACCENT, size: 26 }),
  run("——所有 6 个特征的 p 值都大于 0.7，效应量 η² 都不到 0.011。", { size: 22 }),
  run("物理动作上情绪是完全测不出来的。", { bold: true, size: 22 }),
]));
children.push(speech([
  run("我们的诊断是 ", { size: 22 }),
  runEn("decoder bottleneck", { bold: true, color: ACCENT, size: 24 }),
  run("——解码瓶颈。", { bold: true, size: 22 }),
]));
children.push(speech([
  run("MoConVQ 的物理跟踪策略是为了", { size: 22 }),
  run("精确跟踪", { bold: true, size: 22 }),
  run("训练分布而训出来的。它本能地把输入潜变量", { size: 22 }),
  run("投影回它见过的物理流形", { bold: true, color: ACCENT, size: 22 }),
  run("。我们的情绪信号活在 token 和潜变量里，", { size: 22 }),
  run("但被冻结的跟踪器抹平在了物理动作输出中。", { bold: true, size: 22 }),
]));
children.push(speech([
  run("这是这类“冻结骨干 + 加新接口”方法的", { size: 22 }),
  run("本质局限", { bold: true, color: ACCENT, size: 22 }),
  run("，不是 EmoMoconvq 的方法缺陷。", { size: 22 }),
]));
children.push(stageNote("这一页最重要。讲到“但是”时停顿 1 秒制造反转感。讲到“decoder bottleneck”时指右侧深色诊断卡。"));

// --- SLIDE 8 ---------------------------------------------------------------
children.push(...slideHeader(8, "定性结果与现场演示", 35));
children.push(speech([
  run("左边这个视频是模型对", { size: 22 }),
  run("同一个 prompt——", { bold: true, size: 22 }),
  runEn("“a person runs forward”", { bold: true, italics: true, size: 22 }),
  run("——在三种情绪下生成的物理仿真渲染。", { size: 22 }),
]));
children.push(speech([
  run("左边 ", { size: 22 }),
  runEn("neutral", { bold: true, size: 22 }),
  run(" 是普通跑姿；中间 ", { size: 22 }),
  runEn("happy", { bold: true, size: 22 }),
  run(" 的步频更快、有点跳跃感；右边 ", { size: 22 }),
  runEn("fearful", { bold: true, size: 22 }),
  run(" 步幅明显收缩、有犹豫感。", { size: 22 }),
]));
children.push(speech([
  run("视觉上能看出明显差异——", { size: 22 }),
  run("但运动学统计量抓不到", { bold: true, color: ACCENT, size: 22 }),
  run("，这是一个有意思的开放问题。", { size: 22 }),
]));
children.push(speech([
  run("我们还发布了一个 ", { size: 22 }),
  runEn("Gradio 交互演示", { bold: true, size: 22 }),
  run("：输入文本、选择情绪、按生成，大约 10 秒就能拿到物理仿真的 BVH 文件。", { size: 22 }),
]));
children.push(speech([
  run("完整的代码、三个 checkpoint、两个数据集、15 段对比视频和 demo 都在 GitHub 仓库里。", { size: 22 }),
]));
children.push(stageNote("单击视频区域开始播放；3-5 秒后切回讲下一句即可。"));

// --- SLIDE 9 ---------------------------------------------------------------
children.push(...slideHeader(9, "结论与展望", 25));
children.push(speech([
  run("一句话总结：", { size: 22 }),
]));
children.push(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: 120, after: 120, line: 380 },
  indent: { left: 240 },
  border: { left: { style: BorderStyle.SINGLE, size: 24, color: ACCENT, space: 12 } },
  children: [
    run("情绪模块学到了。被冻结的跟踪器把它学到的东西又抹掉了。", {
      bold: true, italics: true, color: INK, size: 28,
    }),
  ],
}));
children.push(speech([
  run("未来三个方向：", { size: 22 }),
  run("联合可微物理微调跟踪器", { bold: true, color: ACCENT, size: 22 }),
  run("；", { size: 22 }),
  run("在跟踪损失里加情绪保持项", { bold: true, color: ACCENT, size: 22 }),
  run("；以及最根本的——", { size: 22 }),
  run("拿到真情绪标注的物理动作数据，端到端训练", { bold: true, color: ACCENT, size: 22 }),
  run("。", { size: 22 }),
]));
children.push(speech([
  run("这是我们识别出来的", { size: 22 }),
  run("真问题", { bold: true, size: 22 }),
  run("。", { size: 22 }),
  run("谢谢大家。", { bold: true, color: ACCENT, size: 26 }),
]));
children.push(stageNote("“情绪模块学到了。被冻结的跟踪器把它学到的东西又抹掉了”是 punchline，一字一顿、念慢。"));
children.push(blank());

// --- APPENDIX --------------------------------------------------------------
children.push(new Paragraph({ children: [new PageBreak()] }));

children.push(new Paragraph({
  spacing: { before: 0, after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 1 } },
  children: [run("演练 Checklist", { bold: true, size: 32, color: INK })],
}));

const checks = [
  "PowerPoint 第一次打开后，跳到第 8 页单击视频确认能播——并勾选“循环播放，直到停止”。",
  "Spotlight 没有 Q&A，全程读稿没压力，但语速注意——5 分钟 = 约 1100 中文字。",
  "重音和关键数字（1.36 倍 / 39.3% / 全军覆没 / decoder bottleneck）念慢一拍，让评委记住。",
  "现场不要跑 Gradio——加载要 1–2 分钟，风险太高。视频已经够了。",
  "如果时间富裕：在最后一句“谢谢大家”前停顿 2 秒，让 punchline 落地。",
  "如果时间紧：先压缩第 3 页（相关工作）和第 8 页（视频）；第 4 页可把 ConvVQ / Physics 两格合成一句过；核心 5/6/7/9 一定不能砍。",
];
for (const c of checks) {
  children.push(new Paragraph({
    spacing: { before: 60, after: 60, line: 340 },
    indent: { left: 360, hanging: 240 },
    children: [
      run("◆ ", { color: ACCENT, size: 20 }),
      run(c, { size: 22, color: INK }),
    ],
  }));
}

// === Build & write ========================================================
const doc = new Document({
  creator: "Kangwei Sun",
  title: "EmoMoconvq Spotlight 演讲稿",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },          // US Letter
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },  // 0.75"
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            run("EmoMoconvq · Spotlight 演讲稿  —  第 ", { size: 18, color: MUTED }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: MUTED }),
            run(" 页", { size: 18, color: MUTED }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("演讲稿.docx", buf);
  console.log("[ok] wrote 演讲稿.docx");
});
