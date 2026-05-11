const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  PageBreak, AlignmentType, TableOfContents
} = require("docx");

const content = fs.readFileSync(
  "C:\\Users\\xubin\\Desktop\\audio_denoise_graduation\\毕业论文_正文.md",
  "utf-8"
);

const lines = content.split("\n");
const children = [];
let inCodeBlock = false;

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];

  // Skip empty lines
  if (line.trim() === "") {
    if (!inCodeBlock) continue;
  }

  // Page break at "---" (between major sections)
  if (line.trim() === "---" && !inCodeBlock) {
    children.push(new Paragraph({ children: [new PageBreak()] }));
    continue;
  }

  // Code blocks
  if (line.trim().startsWith("```")) {
    inCodeBlock = !inCodeBlock;
    continue;
  }
  if (inCodeBlock) continue;

  // Heading 1 (# )
  if (line.startsWith("# ")) {
    const text = line.slice(2).trim();
    children.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text, font: "SimHei", size: 32, bold: true })],
        spacing: { before: 360, after: 200 },
      })
    );
    continue;
  }

  // Heading 2 (## )
  if (line.startsWith("## ")) {
    const text = line.slice(3).trim();
    children.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text, font: "SimHei", size: 28, bold: true })],
        spacing: { before: 240, after: 160 },
      })
    );
    continue;
  }

  // Heading 3 (### )
  if (line.startsWith("### ")) {
    const text = line.slice(4).trim();
    children.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun({ text, font: "SimHei", size: 26, bold: true })],
        spacing: { before: 200, after: 120 },
      })
    );
    continue;
  }

  // Tables (|...| format)
  if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
    // Skip separator lines like |---|---|
    if (/^\|[\s\-:]*\|[\s\-:]*\|[\s\-:]*\|\s*$/.test(line.trim())) continue;
    // Simple render: parse | cells | and create paragraph
    const cells = line
      .split("|")
      .filter((c) => c.trim())
      .map((c) => c.trim());
    const runs = cells.flatMap((cell, idx) => {
      const segments = parseInlineFormatting(cell);
      if (idx > 0)
        segments.unshift({ text: "  |  ", bold: false, italic: false });
      return segments.map((s) => new TextRun(s));
    });
    children.push(
      new Paragraph({
        spacing: { before: 40, after: 40 },
        indent: { left: 360 },
        children: runs,
      })
    );
    continue;
  }

  // Regular paragraph with inline formatting
  const segments = parseInlineFormatting(line);
  children.push(
    new Paragraph({
      spacing: { before: 60, after: 60, line: 360 },
      children: segments.map(
        (s) =>
          new TextRun({
            text: s.text,
            font: s.bold ? "SimHei" : "SimSun",
            size: 24,
            bold: s.bold,
            italics: s.italic,
          })
      ),
    })
  );
}

function parseInlineFormatting(text) {
  const segments = [];
  let remaining = text;
  while (remaining.length > 0) {
    const boldMatch = remaining.match(/^(.*?)\*\*(.+?)\*\*/);
    const italicMatch = remaining.match(/^(.*?)\*(.+?)\*/);
    const inlineCode = remaining.match(/^(.*?)`(.+?)`/);

    let matchType = null;
    let matchIdx = Infinity;

    if (boldMatch && boldMatch.index < matchIdx) {
      matchType = "bold";
      matchIdx = boldMatch.index;
    }
    if (italicMatch && italicMatch.index < matchIdx) {
      matchType = "italic";
      matchIdx = italicMatch.index;
    }
    if (inlineCode && inlineCode.index < matchIdx) {
      matchType = "code";
      matchIdx = inlineCode.index;
    }

    if (matchType === null) {
      segments.push({ text: remaining, bold: false, italic: false });
      break;
    }

    let match;
    if (matchType === "bold") match = boldMatch;
    else if (matchType === "italic") match = italicMatch;
    else match = inlineCode;

    if (match[1].length > 0)
      segments.push({ text: match[1], bold: false, italic: false });

    if (matchType === "bold")
      segments.push({ text: match[2], bold: true, italic: false });
    else if (matchType === "italic")
      segments.push({ text: match[2], bold: false, italic: true });
    else segments.push({ text: match[2], bold: false, italic: false });

    remaining = remaining.slice(match[0].length);
  }
  return segments.length === 0
    ? [{ text: "", bold: false, italic: false }]
    : segments;
}

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "SimSun", size: 24 },
        paragraph: { spacing: { line: 360 } },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1 Heading 1 char char",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 32, bold: true, font: "SimHei" },
        paragraph: {
          spacing: { before: 360, after: 240 },
          outlineLevel: 0,
        },
      },
      {
        id: "Heading2",
        name: "Heading 2 Heading 2 char char char",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "SimHei" },
        paragraph: {
          spacing: { before: 240, after: 160 },
          outlineLevel: 1,
        },
      },
      {
        id: "Heading3",
        name: "Heading 3 Heading 3 char char char",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 26, bold: true, font: "SimHei" },
        paragraph: {
          spacing: { before: 200, after: 120 },
          outlineLevel: 2,
        },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(
    "C:\\Users\\xubin\\Desktop\\audio_denoise_graduation\\毕业论文_完整版.docx",
    buf
  );
  console.log("Generated 毕业论文_完整版.docx (" + (buf.length / 1024).toFixed(0) + " KB)");
});
