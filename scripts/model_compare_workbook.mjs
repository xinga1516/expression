import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function parseArgs(argv) {
  let payload = null;
  let output = null;
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--payload") payload = argv[++index];
    else if (token === "--output") output = argv[++index];
    else if (token === "--help" || token === "-h") {
      console.log("Usage: node scripts/model_compare_workbook.mjs --payload report.json --output model_compare_summary.xlsx");
      process.exit(0);
    } else throw new Error(`Unknown argument: ${token}`);
  }
  if (!payload || !output) throw new Error("--payload and --output are required");
  return { payload, output };
}

function valuesFor(rows, headers) {
  return rows.map((row) => headers.map((header) => row[header] ?? null));
}

function headersFor(rows, leading = []) {
  const available = new Set(rows.flatMap((row) => Object.keys(row)));
  return [...leading.filter((key) => available.has(key)), ...[...available].filter((key) => !leading.includes(key)).sort()];
}

function columnLabel(index) {
  let number = index + 1;
  let label = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    label = String.fromCharCode(65 + remainder) + label;
    number = Math.floor((number - 1) / 26);
  }
  return label;
}

function formatTable(sheet, title, headers, rows, startRow = 0) {
  const titleRow = startRow;
  const headerRow = startRow + 2;
  const dataRow = startRow + 3;
  sheet.getRangeByIndexes(titleRow, 0, 1, Math.max(headers.length, 1)).merge();
  sheet.getCell(titleRow, 0).values = [[title]];
  sheet.getCell(titleRow, 0).format = { fill: "#163A5F", font: { bold: true, color: "#FFFFFF", size: 14 } };
  if (!headers.length) return dataRow + 1;
  sheet.getRangeByIndexes(headerRow, 0, 1, headers.length).values = [headers];
  sheet.getRangeByIndexes(headerRow, 0, 1, headers.length).format = {
    fill: "#2F75B5",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    horizontalAlignment: "center",
  };
  if (rows.length) {
    sheet.getRangeByIndexes(dataRow, 0, rows.length, headers.length).values = valuesFor(rows, headers);
    if (rows.length <= 10000) {
      sheet.getRangeByIndexes(dataRow, 0, rows.length, headers.length).format.borders = {
        preset: "inside",
        style: "thin",
        color: "#E7E6E6",
      };
    }
    sheet.tables.add(`${columnLabel(0)}${headerRow + 1}:${columnLabel(headers.length - 1)}${dataRow + rows.length}`, true, `Table_${sheet.name.replace(/\W/g, "_")}_${titleRow + 1}`);
  }
  return dataRow + Math.max(rows.length, 1) + 3;
}

function applyColumnSizing(sheet) {
  const range = sheet.getUsedRange();
  range.format.autofitColumns();
}

async function addImage(sheet, figure, row) {
  const bytes = await fs.readFile(figure.path);
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
  sheet.images.add({
    dataUrl,
    anchor: { from: { row, col: 0 }, extent: { widthPx: 760, heightPx: 510 } },
  });
  sheet.getCell(row - 1, 0).values = [[figure.title]];
  sheet.getCell(row - 1, 0).format = { font: { bold: true, color: "#163A5F" } };
  return row + 31;
}

async function main() {
  const { payload: payloadPath, output } = parseArgs(process.argv.slice(2));
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const workbook = Workbook.create();

  const readme = workbook.worksheets.add("README");
  readme.showGridLines = false;
  const readmeRows = [
    { field: "Manifest version", value: payload.manifest_version },
    { field: "Stages", value: payload.stages.map((stage) => stage.id).join(", ") },
    { field: "Run audit", value: "Completed runs with config.json and test/test_metrics.json" },
    { field: "Seed summary", value: "Only configurations identical except seed are grouped" },
    { field: "Paired summary", value: "Consumes registered paired statistics; raw paired detail is intentionally excluded" },
    { field: "Stage figures", value: "Embeds canonical stage PNG outputs without recomputation" },
  ];
  formatTable(readme, "Model Comparison Report", ["field", "value"], readmeRows);
  readme.getRange("B:B").format.columnWidth = 92;

  const inventory = workbook.worksheets.add("Input Inventory");
  inventory.showGridLines = false;
  const inventoryHeaders = headersFor(payload.inventory, ["stage", "kind", "id", "path", "status"]);
  formatTable(inventory, "Validated manifest inputs", inventoryHeaders, payload.inventory);
  inventory.freezePanes.freezeRows(3);

  const audit = workbook.worksheets.add("Run Audit");
  audit.showGridLines = false;
  const auditHeaders = payload.run_audit_columns;
  formatTable(audit, "Run checkpoint, validation, and test audit", auditHeaders, payload.run_audit_rows);
  audit.freezePanes.freezeRows(3);

  const seeds = workbook.worksheets.add("Seed Summary");
  seeds.showGridLines = false;
  const seedHeaders = headersFor(payload.seed_rows, ["stage", "config_group", "n_runs", "seeds", "model", "loss", "sequence_column", "contrastive_weight"]);
  formatTable(seeds, "Configurations identical except seed", seedHeaders, payload.seed_rows);
  seeds.freezePanes.freezeRows(3);

  const paired = workbook.worksheets.add("Paired Summary");
  paired.showGridLines = false;
  const pairedHeaders = headersFor(payload.paired_rows, ["stage", "artifact", "comparison", "level", "seed", "n_pairs", "mean_delta", "mean_ci_low", "mean_ci_high", "win_fraction"]);
  formatTable(paired, "Registered paired comparison statistics", pairedHeaders, payload.paired_rows);
  paired.freezePanes.freezeRows(3);

  for (const stage of payload.stages) {
    const raw = workbook.worksheets.add(`${stage.id === "stage1" ? "Stage 1" : "Stage 2"} Raw Summary`);
    raw.showGridLines = false;
    formatTable(raw, `${stage.title}: all available run variables`, stage.raw_columns, stage.raw_rows);
    raw.freezePanes.freezeRows(3);

    const overview = workbook.worksheets.add(stage.id === "stage1" ? "Stage 1" : "Stage 2");
    overview.showGridLines = false;
    let row = 0;
    for (const table of stage.tables) {
      const headers = headersFor(table.rows);
      row = formatTable(overview, table.title, headers, table.rows, row);
    }
    for (const figure of stage.figures) row = await addImage(overview, figure, row + 1);
    overview.freezePanes.freezeRows(3);
  }

  for (const sheet of workbook.worksheets.items) applyColumnSizing(sheet);
  const inspection = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 3000 });
  if (!inspection.ndjson.includes("README") || !inspection.ndjson.includes("Paired Summary")) {
    throw new Error("Workbook sheet inspection failed.");
  }
  await fs.mkdir(path.dirname(output), { recursive: true });
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(output);
  console.log(JSON.stringify({ output, sheets: workbook.worksheets.items.length }));
}

await main();
