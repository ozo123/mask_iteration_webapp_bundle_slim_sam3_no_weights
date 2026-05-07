import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const defaultWorkbook =
  "C:/Users/admin/Documents/xwechat_files/wxid_vads14tpynk312_d3fb/msg/file/2026-04/2D室内目标检测明细.xlsx";
const defaultAnnotationRoot = "C:/Users/admin/Desktop/标注";

const workbookPath = process.argv[2] ?? defaultWorkbook;
const annotationRoot = process.argv[3] ?? defaultAnnotationRoot;

async function listJsonFiles(root) {
  const result = [];
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      result.push(...(await listJsonFiles(fullPath)));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".json")) {
      result.push(fullPath);
    }
  }
  return result;
}

function hasValidBox(annotation) {
  return Array.isArray(annotation?.bbox) && annotation.bbox.length >= 4;
}

async function countJsonAnnotations(root) {
  const counts = new Map();
  const knownNames = new Set();
  const files = await listJsonFiles(root);

  for (const file of files) {
    const data = JSON.parse(await fs.readFile(file, "utf8"));
    const idToName = new Map();
    for (const category of data.categories ?? []) {
      if (category?.id != null && category?.name) {
        idToName.set(category.id, String(category.name));
        knownNames.add(String(category.name));
      }
    }
    for (const annotation of data.annotations ?? []) {
      if (!hasValidBox(annotation)) continue;
      const name = idToName.get(annotation.category_id) ?? `unknown:${annotation.category_id}`;
      counts.set(name, (counts.get(name) ?? 0) + 1);
      knownNames.add(name);
    }
  }

  return { counts, knownNames, fileCount: files.length };
}

async function readOutsourceWorkbook(file) {
  const input = await FileBlob.load(file);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const inspect = await workbook.inspect({
    kind: "table",
    range: "Sheet1!A1:B80",
    maxChars: 20000,
    tableMaxRows: 80,
    tableMaxCols: 2,
    tableMaxCellChars: 120,
  });
  const tableLine = inspect.ndjson
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .find((record) => record.kind === "table" && Array.isArray(record.values));

  if (!tableLine) {
    throw new Error("No readable table found in workbook.");
  }

  const counts = new Map();
  let total = null;
  for (const row of tableLine.values.slice(1)) {
    const name = String(row[0] ?? "").trim();
    const value = Number(row[1] ?? 0);
    if (!name) continue;
    if (name === "合计") {
      total = value;
    } else {
      counts.set(name, value);
    }
  }
  return { counts, total };
}

function sumCounts(counts) {
  let total = 0;
  for (const value of counts.values()) total += value;
  return total;
}

const ours = await countJsonAnnotations(annotationRoot);
const outsource = await readOutsourceWorkbook(workbookPath);

const allNames = new Set([...ours.knownNames, ...ours.counts.keys(), ...outsource.counts.keys()]);
const diffs = [];
for (const name of allNames) {
  const oursValue = ours.counts.get(name) ?? 0;
  const outsourceValue = outsource.counts.get(name) ?? 0;
  if (oursValue !== outsourceValue) {
    diffs.push({
      name,
      ours: oursValue,
      outsource: outsourceValue,
      diff: outsourceValue - oursValue,
      inJsonCategories: ours.knownNames.has(name),
      inOutsource: outsource.counts.has(name),
    });
  }
}

diffs.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff) || a.name.localeCompare(b.name));

console.log(`Annotation files: ${ours.fileCount}`);
console.log(`JSON total boxes: ${sumCounts(ours.counts)}`);
console.log(`Outsource total boxes: ${sumCounts(outsource.counts)}`);
console.log(`Outsource workbook total row: ${outsource.total ?? "N/A"}`);
console.log(`Total difference outsource - JSON: ${sumCounts(outsource.counts) - sumCounts(ours.counts)}`);
console.log("");
console.log("Category\tJSON\tOutsource\tDiff(outsource-JSON)\tNote");
for (const row of diffs) {
  let note = "";
  if (!row.inOutsource) note = "missing in outsource workbook";
  if (!row.inJsonCategories) note = "missing in JSON categories";
  console.log(`${row.name}\t${row.ours}\t${row.outsource}\t${row.diff}\t${note}`);
}
