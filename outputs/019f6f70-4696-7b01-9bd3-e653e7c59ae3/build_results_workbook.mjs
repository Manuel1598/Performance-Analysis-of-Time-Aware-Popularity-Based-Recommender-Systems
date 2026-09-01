import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "F:/TimeAware Popularity Models and Fair Evaluation";
const outDir = `${root}/outputs/019f6f70-4696-7b01-9bd3-e653e7c59ae3`;
const outputPath = `${outDir}/Results_evaluierung_aktuell_17_07_2026.xlsx`;

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ""; }
    else if (ch === '\n') { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows;
}

function records(rows) {
  const headers = rows[0];
  return rows.slice(1).filter(r => r.some(Boolean)).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

function numeric(value) {
  if (value === "" || value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

const overviewData = records(parseCsv(await fs.readFile(`${root}/recbole_results/summary/best_models_by_dataset.csv`, "utf8")));
const winnerData = records(parseCsv(await fs.readFile(`${root}/recbole_results/summary/best_overall_model_per_dataset.csv`, "utf8")));
const winnerKeys = new Set(winnerData.map(r => `${r.scenario}|${r.dataset}|${r.model}`));

const wb = Workbook.create();
const overview = wb.worksheets.add("Best_Config_Per_Model");
const winners = wb.worksheets.add("Datensatzsieger");
const compare = wb.worksheets.add("VSKNN_Legacy_Vergleich");
const quality = wb.worksheets.add("Quality_Runtime");

const blue = "#7FC3DF", darkBlue = "#1F4E78", green = "#4EA72E", paleGreen = "#E2F0D9", yellow = "#FFD966", light = "#F3F6F8", border = "#7F8C8D";
const headerFormat = { fill: blue, font: { bold: true, color: "#000000" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: border } };
const titleFormat = { fill: darkBlue, font: { bold: true, color: "#FFFFFF", size: 15 }, verticalAlignment: "center" };
const bodyBorder = { preset: "all", style: "thin", color: "#B7B7B7" };

function setupSheet(sheet) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(3);
}

// Detailed best configuration per model/dataset.
setupSheet(overview);
overview.getRange("A1:Q1").merge();
overview.getRange("A1").values = [["Aktuelle RecBole-Ergebnisse – beste Konfiguration je Modell und Datensatz"]];
overview.getRange("A1:Q1").format = titleFormat;
overview.getRange("A2:Q2").merge();
overview.getRange("A2").values = [["Auswahlmetrik: MRR@10; bei Gleichstand niedrigere Laufzeit. Audited VSKNN ersetzt die Legacy-VS-KNN-Zeilen."]];
overview.getRange("A2:Q2").format = { fill: "#D9EAF2", font: { italic: true, color: "#404040" }, wrapText: true };
const overviewHeaders = ["Szenario","Datensatz","Modell","Implementierung","Gerät","Hit@5","Hit@10","NDCG@5","NDCG@10","MRR@5","MRR@10","Laufzeit (s)","Laufzeit (min)","Beste Konfiguration (JSON)","Ergebnisquelle","Run-ID","Datensatzsieger"];
overview.getRange("A3:Q3").values = [overviewHeaders];
overview.getRange("A3:Q3").format = headerFormat;
const overviewRows = overviewData.map(r => [r.scenario,r.dataset,r.model,r.implementation,r.device,numeric(r["hit@5"]),numeric(r["hit@10"]),numeric(r["ndcg@5"]),numeric(r["ndcg@10"]),numeric(r["mrr@5"]),numeric(r["mrr@10"]),numeric(r.runtime_seconds),null,r.config_json,r.result_source,r.run_id,winnerKeys.has(`${r.scenario}|${r.dataset}|${r.model}`) ? "Ja" : ""]);
overview.getRange(`A4:Q${overviewRows.length + 3}`).values = overviewRows;
overview.getRange("M4").formulas = [["=IFERROR(L4/60,\"\")"]];
overview.getRange(`M4:M${overviewRows.length + 3}`).fillDown();
overview.getRange(`A4:Q${overviewRows.length + 3}`).format.borders = bodyBorder;
overview.getRange(`F4:M${overviewRows.length + 3}`).format.numberFormat = "0.0000";
overview.getRange(`L4:M${overviewRows.length + 3}`).format.numberFormat = "0.00";
overview.getRange(`N4:N${overviewRows.length + 3}`).format.wrapText = true;
overview.getRange(`O4:P${overviewRows.length + 3}`).format.wrapText = true;
overview.getRange(`Q4:Q${overviewRows.length + 3}`).conditionalFormats.add("containsText", { text: "Ja", format: { fill: green, font: { bold: true, color: "#FFFFFF" } } });
overview.tables.add(`A3:Q${overviewRows.length + 3}`, true, "BestConfigTable").style = "TableStyleMedium2";
overview.getRange("A:A").format.columnWidth = 15; overview.getRange("B:B").format.columnWidth = 27; overview.getRange("C:E").format.columnWidth = 15;
overview.getRange("F:M").format.columnWidth = 13; overview.getRange("N:N").format.columnWidth = 58; overview.getRange("O:P").format.columnWidth = 35; overview.getRange("Q:Q").format.columnWidth = 16;
overview.getRange("1:1").format.rowHeight = 28; overview.getRange("2:2").format.rowHeight = 28; overview.getRange("3:3").format.rowHeight = 34;
overview.getRange(`4:${overviewRows.length + 3}`).format.rowHeight = 38;

// Dataset winners.
setupSheet(winners);
winners.getRange("A1:J1").merge(); winners.getRange("A1").values = [["Bestes Modell je Datensatz nach MRR@10"]]; winners.getRange("A1:J1").format = titleFormat;
winners.getRange("A2:J2").merge(); winners.getRange("A2").values = [["Top-N- und Session-Szenarien werden getrennt interpretiert. Laufzeiten sind bei unterschiedlicher Hardware nur beschreibend."]]; winners.getRange("A2:J2").format = { fill: "#D9EAF2", font: { italic: true }, wrapText: true };
winners.getRange("A3:J3").values = [["Szenario","Datensatz","Bestes Modell","Gerät","Hit@10","NDCG@10","MRR@10","Laufzeit (s)","Konfiguration","Quelle"]]; winners.getRange("A3:J3").format = headerFormat;
const winnerRows = winnerData.map(r => [r.scenario,r.dataset,r.model,r.device,numeric(r["hit@10"]),numeric(r["ndcg@10"]),numeric(r["mrr@10"]),numeric(r.runtime_seconds),r.config_json,r.result_source]);
winners.getRange(`A4:J${winnerRows.length + 3}`).values = winnerRows;
winners.getRange(`A4:J${winnerRows.length + 3}`).format = { fill: paleGreen, borders: bodyBorder };
winners.getRange(`C4:C${winnerRows.length + 3}`).format.font = { bold: true, color: "#215E21" };
winners.getRange(`E4:H${winnerRows.length + 3}`).format.numberFormat = "0.0000"; winners.getRange(`H4:H${winnerRows.length + 3}`).format.numberFormat = "0.00";
winners.getRange(`I4:J${winnerRows.length + 3}`).format.wrapText = true;
winners.tables.add(`A3:J${winnerRows.length + 3}`, true, "DatasetWinnersTable").style = "TableStyleMedium4";
winners.getRange("A:A").format.columnWidth = 16; winners.getRange("B:B").format.columnWidth = 28; winners.getRange("C:D").format.columnWidth = 16; winners.getRange("E:H").format.columnWidth = 14; winners.getRange("I:I").format.columnWidth = 62; winners.getRange("J:J").format.columnWidth = 35;
winners.getRange("1:1").format.rowHeight = 28; winners.getRange("2:2").format.rowHeight = 30; winners.getRange("3:3").format.rowHeight = 32;
winners.getRange(`4:${winnerRows.length + 3}`).format.rowHeight = 45;

// Legacy versus audited VSKNN; deltas remain auditable formulas.
setupSheet(compare);
compare.getRange("A1:N1").merge(); compare.getRange("A1").values = [["VSKNN: Legacy-Lauf versus korrigierte und kompakt getunte Implementierung"]]; compare.getRange("A1:N1").format = titleFormat;
compare.getRange("A2:N2").merge(); compare.getRange("A2").values = [["Die Differenzen enthalten sowohl Korrekturen als auch Tuningeffekte und isolieren daher keine einzelne Codeänderung."]]; compare.getRange("A2:N2").format = { fill: "#FFF2CC", font: { italic: true }, wrapText: true };
compare.getRange("A3:N3").values = [["Datensatz","Legacy Hit@10","Aktuell Hit@10","Δ Hit@10","Legacy NDCG@10","Aktuell NDCG@10","Δ NDCG@10","Legacy MRR@10","Aktuell MRR@10","Δ MRR@10","Rel. Δ MRR","Neighbor Size","Sample Size","Gewichtungen"]]; compare.getRange("A3:N3").format = headerFormat;
const legacy = [
  ["yoochoose_recbole_sample",0.4947,0.5377,null,0.3177,0.3478,null,0.2624,0.2880,null,null,100,500,"vec / div / quadratic"],
  ["globo_recbole_sample",0.3373,0.3298,null,0.1326,0.1341,null,0.0713,0.0751,null,null,100,1000,"vec / div / div"],
  ["adressa_recbole_sample",0.3428,0.4313,null,0.1735,0.2312,null,0.1225,0.1703,null,null,200,500,"vec / div / div"],
];
compare.getRange("A4:N6").values = legacy;
compare.getRange("D4").formulas = [["=C4-B4"]]; compare.getRange("D4:D6").fillDown();
compare.getRange("G4").formulas = [["=F4-E4"]]; compare.getRange("G4:G6").fillDown();
compare.getRange("J4").formulas = [["=I4-H4"]]; compare.getRange("J4:J6").fillDown();
compare.getRange("K4").formulas = [["=IFERROR(J4/H4,\"\")"]]; compare.getRange("K4:K6").fillDown();
compare.getRange("A4:N6").format.borders = bodyBorder; compare.getRange("B4:J6").format.numberFormat = "0.0000"; compare.getRange("K4:K6").format.numberFormat = "0.0%";
compare.getRange("D4:K6").conditionalFormats.add("cellIs", { operator: "greaterThan", formula: 0, format: { fill: paleGreen, font: { color: "#215E21", bold: true } } });
compare.getRange("D4:K6").conditionalFormats.add("cellIs", { operator: "lessThan", formula: 0, format: { fill: "#F4CCCC", font: { color: "#9C0006", bold: true } } });
compare.tables.add("A3:N6", true, "VsknnLegacyTable").style = "TableStyleMedium2";
compare.getRange("A:A").format.columnWidth = 28; compare.getRange("B:K").format.columnWidth = 17; compare.getRange("L:M").format.columnWidth = 15; compare.getRange("N:N").format.columnWidth = 25;
compare.getRange("1:1").format.rowHeight = 28; compare.getRange("2:2").format.rowHeight = 30; compare.getRange("3:3").format.rowHeight = 42;

// Quality/runtime analysis, linked to the detailed source sheet.
setupSheet(quality);
quality.getRange("A1:M1").merge(); quality.getRange("A1").values = [["Qualitäts- und Laufzeitanalyse aller aktuellen Modell-Datensatz-Kombinationen"]]; quality.getRange("A1:M1").format = titleFormat;
quality.getRange("A2:M2").merge(); quality.getRange("A2").values = [["MRR@10 ist die primäre Qualitätsmetrik. Laufzeitvergleiche zwischen CPU und CUDA sind nicht als Hardwarebenchmark zu interpretieren."]]; quality.getRange("A2:M2").format = { fill: "#D9EAF2", font: { italic: true }, wrapText: true };
quality.getRange("A3:M3").values = [["Szenario","Datensatz","Modell","Hit@10","NDCG@10","MRR@10","Laufzeit (s)","Laufzeit (min)","MRR@10/min","Rang im Datensatz","Relativ zum Besten","Abstand zum Besten","Pareto-effizient"]]; quality.getRange("A3:M3").format = headerFormat;
const n = overviewRows.length, end = n + 3;
for (let i = 0; i < n; i++) {
  const row = i + 4;
  quality.getRange(`A${row}:G${row}`).formulas = [[`='Best_Config_Per_Model'!A${row}`,`='Best_Config_Per_Model'!B${row}`,`='Best_Config_Per_Model'!C${row}`,`='Best_Config_Per_Model'!G${row}`,`='Best_Config_Per_Model'!I${row}`,`='Best_Config_Per_Model'!K${row}`,`='Best_Config_Per_Model'!L${row}`]];
  quality.getRange(`H${row}:M${row}`).formulas = [[`=IFERROR(G${row}/60,\"\")`,`=IFERROR(F${row}/H${row},\"\")`,`=1+COUNTIFS($B$4:$B$${end},B${row},$F$4:$F$${end},\">\"&F${row})`,`=IFERROR(F${row}/MAXIFS($F$4:$F$${end},$B$4:$B$${end},B${row}),\"\")`,`=MAXIFS($F$4:$F$${end},$B$4:$B$${end},B${row})-F${row}`,`=IF(COUNTIFS($B$4:$B$${end},B${row},$F$4:$F$${end},\">=\"&F${row},$G$4:$G$${end},\"<=\"&G${row})>1,\"Nein\",\"Ja\")`]];
}
quality.getRange(`A4:M${end}`).format.borders = bodyBorder; quality.getRange(`D4:I${end}`).format.numberFormat = "0.0000"; quality.getRange(`G4:H${end}`).format.numberFormat = "0.00"; quality.getRange(`K4:K${end}`).format.numberFormat = "0.0%";
quality.getRange(`M4:M${end}`).conditionalFormats.add("containsText", { text: "Ja", format: { fill: green, font: { bold: true, color: "#FFFFFF" } } });
quality.getRange(`J4:J${end}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 1, format: { fill: yellow, font: { bold: true } } });
quality.tables.add(`A3:M${end}`, true, "QualityRuntimeTable").style = "TableStyleMedium2";
quality.getRange("A:A").format.columnWidth = 16; quality.getRange("B:B").format.columnWidth = 28; quality.getRange("C:C").format.columnWidth = 16; quality.getRange("D:M").format.columnWidth = 17;
quality.getRange("1:1").format.rowHeight = 28; quality.getRange("2:2").format.rowHeight = 30; quality.getRange("3:3").format.rowHeight = 42;

await fs.mkdir(`${outDir}/final_previews`, { recursive: true });
for (const sheet of wb.worksheets.items) {
  const preview = await wb.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outDir}/final_previews/${sheet.name}.png`, new Uint8Array(await preview.arrayBuffer()));
}

console.log((await wb.inspect({ kind: "sheet,table", include: "id,name", maxChars: 5000 })).ndjson);
console.log((await wb.inspect({ kind: "table", sheetId: "Datensatzsieger", range: "A1:J8", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 12, maxChars: 6000 })).ndjson);
console.log((await wb.inspect({ kind: "table", sheetId: "VSKNN_Legacy_Vergleich", range: "A1:N6", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 16, maxChars: 6000 })).ndjson);
console.log((await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" })).ndjson);

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(`OUTPUT ${outputPath}`);
