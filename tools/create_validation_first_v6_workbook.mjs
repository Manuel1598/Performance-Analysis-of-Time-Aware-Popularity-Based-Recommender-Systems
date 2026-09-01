import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function argumentsFrom(argv) {
  const result = {};
  for (let index = 2; index < argv.length; index += 2) {
    if (!argv[index]?.startsWith("--") || argv[index + 1] === undefined) {
      throw new Error("Use --input <json> --output <xlsx> --preview-dir <dir>");
    }
    result[argv[index].slice(2)] = argv[index + 1];
  }
  for (const key of ["input", "output", "preview-dir"]) {
    if (!result[key]) throw new Error(`Missing --${key}`);
  }
  return result;
}


function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    result = String.fromCharCode(65 + ((value - 1) % 26)) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}


function headersFor(records) {
  const headers = [];
  for (const record of records) {
    for (const key of Object.keys(record)) {
      if (!headers.includes(key)) headers.push(key);
    }
  }
  return headers;
}


function typedValue(header, value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (
    /(^seed$|count|expected|successful|failed|missing|runtime|score|mrr|hit|ndcg|coverage|popularity|gini|catalogue)/i.test(header)
    && String(value).trim() !== ""
    && Number.isFinite(Number(value))
  ) return Number(value);
  return String(value);
}


function tableName(sheetName, index) {
  return `${sheetName.replace(/[^A-Za-z0-9]/g, "") || "Sheet"}Table${index + 1}`;
}


function addStatusRules(sheet, headers, rowCount) {
  const statusIndex = headers.findIndex((header) => header === "status" || header === "value");
  if (statusIndex < 0 || rowCount < 1) return;
  const column = columnName(statusIndex);
  const range = sheet.getRange(`${column}2:${column}${rowCount + 1}`);
  for (const [text, fill, color] of [
    ["success", "#DCFCE7", "#166534"],
    ["complete", "#DCFCE7", "#166534"],
    ["failed", "#FEE2E2", "#991B1B"],
    ["incomplete", "#FEF3C7", "#92400E"],
  ]) {
    range.conditionalFormats.add("containsText", {
      text,
      format: { fill, font: { color, bold: true } },
    });
  }
}


function addSheet(workbook, name, records, index) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const rows = records.length ? records : [{ status: "No rows available yet" }];
  const headers = headersFor(rows);
  const matrix = [headers, ...rows.map((row) => headers.map((header) => typedValue(header, row[header])))];
  const lastColumn = columnName(headers.length - 1);
  const lastRow = matrix.length;
  const used = sheet.getRange(`A1:${lastColumn}${lastRow}`);
  used.values = matrix;
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "medium", color: "#17365D" },
    rowHeight: 30,
  };
  if (lastRow > 1) {
    sheet.getRange(`A2:${lastColumn}${lastRow}`).format = {
      verticalAlignment: "top",
      borders: { preset: "inside", style: "thin", color: "#D9E2F3" },
      rowHeight: 20,
    };
  }
  headers.forEach((header, columnIndex) => {
    const column = columnName(columnIndex);
    let width = 16;
    if (/id$|config|path|error|message|source/i.test(header)) width = 34;
    else if (/dataset|model|scenario|decision|status|item/i.test(header)) width = 22;
    sheet.getRange(`${column}1:${column}${lastRow}`).format.columnWidth = width;
  });
  const table = sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, tableName(name, index));
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  addStatusRules(sheet, headers, rows.length);
  return { sheet, headers, rowCount: rows.length };
}


function rowNumber(records, item) {
  return records.findIndex((record) => record.item === item) + 2;
}


function addReadmeFormulas(readme, readmeRecords, sheetInfo) {
  const valueColumn = columnName(readme.headers.indexOf("value"));
  for (const [item, targetSheet] of [
    ["Validation successful", "Validation Merged"],
    ["Final tests successful", "Final Raw"],
  ]) {
    const info = sheetInfo.get(targetSheet);
    const statusIndex = info.headers.indexOf("status");
    if (statusIndex < 0) continue;
    const statusColumn = columnName(statusIndex);
    readme.sheet.getRange(`${valueColumn}${rowNumber(readmeRecords, item)}`).formulas = [[
      `=COUNTIF('${targetSheet}'!$${statusColumn}$2:$${statusColumn}$${info.rowCount + 1},"success")`,
    ]];
  }
  const validationExpected = rowNumber(readmeRecords, "Validation expected");
  const validationSuccessful = rowNumber(readmeRecords, "Validation successful");
  const validationComplete = rowNumber(readmeRecords, "Validation complete");
  const finalExpected = rowNumber(readmeRecords, "Final tests expected");
  const finalSuccessful = rowNumber(readmeRecords, "Final tests successful");
  const finalComplete = rowNumber(readmeRecords, "Final tests complete");
  const overall = rowNumber(readmeRecords, "Overall state");
  readme.sheet.getRange(`${valueColumn}${validationComplete}`).formulas = [[
    `=IF(${valueColumn}${validationSuccessful}=${valueColumn}${validationExpected},"TRUE","FALSE")`,
  ]];
  readme.sheet.getRange(`${valueColumn}${finalComplete}`).formulas = [[
    `=IF(${valueColumn}${finalSuccessful}=${valueColumn}${finalExpected},"TRUE","FALSE")`,
  ]];
  readme.sheet.getRange(`${valueColumn}${overall}`).formulas = [[
    `=IF(AND(${valueColumn}${validationComplete}="TRUE",${valueColumn}${finalComplete}="TRUE"),"complete","incomplete")`,
  ]];
}


const args = argumentsFrom(process.argv);
const payload = JSON.parse(await fs.readFile(args.input, "utf8"));
const workbook = Workbook.create();
const info = new Map();
let index = 0;
for (const [name, records] of Object.entries(payload.sheets)) {
  info.set(name, addSheet(workbook, name, records, index));
  index += 1;
}
addReadmeFormulas(info.get("README"), payload.sheets.README ?? [], info);

console.log((await workbook.inspect({
  kind: "table",
  range: `README!A1:B${(payload.sheets.README ?? []).length + 1}`,
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 6,
  maxChars: 5000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 5000,
})).ndjson);

await fs.mkdir(path.dirname(args.output), { recursive: true });
await fs.mkdir(args["preview-dir"], { recursive: true });
for (const name of Object.keys(payload.sheets)) {
  const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
  const fileName = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  await fs.writeFile(path.join(args["preview-dir"], `${fileName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(args.output);
console.log(JSON.stringify({ output: args.output, previewDir: args["preview-dir"] }));
