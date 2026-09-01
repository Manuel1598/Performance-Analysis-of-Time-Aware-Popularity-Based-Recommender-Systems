import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const source = "C:/Users/manue/Desktop/Uni Unterlagen/MSc/Semester 3/Msc_Arbeit/Wöchentliche_Treffen/Recbole_Resulta_24_06/Results_evaluierung_17_06.xlsx";
const outDir = "F:/TimeAware Popularity Models and Fair Evaluation/outputs/019f6f70-4696-7b01-9bd3-e653e7c59ae3/template_previews";
await fs.mkdir(outDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
console.log((await workbook.inspect({
  kind: "workbook,sheet,table,drawing",
  include: "id,name,values,formulas",
  maxChars: 12000,
  tableMaxRows: 12,
  tableMaxCols: 18,
})).ndjson);

for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange();
  console.log(`USED ${sheet.name}: ${used?.address ?? "none"}`);
  if (used) {
    console.log((await workbook.inspect({
      kind: "region",
      sheetId: sheet.name,
      range: used.address,
      include: "values,formulas",
      maxChars: 14000,
      tableMaxRows: 40,
      tableMaxCols: 24,
    })).ndjson);
    const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
    const safe = sheet.name.replace(/[^a-zA-Z0-9_-]+/g, "_");
    await fs.writeFile(`${outDir}/${safe}.png`, new Uint8Array(await preview.arrayBuffer()));
  }
}
