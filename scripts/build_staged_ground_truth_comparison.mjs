import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..");
const sourceDir = path.join(
  projectRoot,
  "outputs",
  "activation_gt_comparison_20260813",
);
const frameComparisonPath = path.join(sourceDir, "frame_comparison.csv");
const originalSummaryPath = path.join(sourceDir, "summary_metrics.csv");
const outputSpecs = [
  {
    id: "three_stage",
    outputDir: path.join(
      projectRoot,
      "outputs",
      "ground_truth_3stage_comparison_20260820",
    ),
    workbookName: "ground_truth_3stage_comparison.xlsx",
    title: "Ground Truth 3단계 분류 비교",
    subtitle: "적다=1명 · 중간=2명 · 많다=3명 이상 · 예측 0명=MISS",
    classKeys: ["few", "medium", "many"],
    classLabels: {
      few: "적다 (1명)",
      medium: "중간 (2명)",
      many: "많다 (3명 이상)",
      miss: "MISS (예측 0명)",
    },
    mapGroundTruth(count) {
      if (count === 1) return "few";
      if (count === 2) return "medium";
      if (count >= 3) return "many";
      throw new Error(`3단계 정답 범위 밖 인원수: ${count}`);
    },
    mapPrediction(count) {
      if (count === 0) return "miss";
      if (count === 1) return "few";
      if (count === 2) return "medium";
      return "many";
    },
  },
  {
    id: "two_stage",
    outputDir: path.join(
      projectRoot,
      "outputs",
      "ground_truth_2stage_comparison_20260820",
    ),
    workbookName: "ground_truth_2stage_comparison.xlsx",
    title: "Ground Truth 2단계 분류 비교",
    subtitle: "적다=1~2명 · 많다=3명 이상 · 예측 0명=MISS",
    classKeys: ["few", "many"],
    classLabels: {
      few: "적다 (1~2명)",
      many: "많다 (3명 이상)",
      miss: "MISS (예측 0명)",
    },
    mapGroundTruth(count) {
      if (count >= 1 && count <= 2) return "few";
      if (count >= 3) return "many";
      throw new Error(`2단계 정답 범위 밖 인원수: ${count}`);
    },
    mapPrediction(count) {
      if (count === 0) return "miss";
      if (count <= 2) return "few";
      return "many";
    },
  },
];

const variants = [
  { key: "activation_0_20", threshold: 0.2 },
  { key: "activation_0_30", threshold: 0.3 },
  { key: "activation_0_40", threshold: 0.4 },
  { key: "activation_0_50", threshold: 0.5 },
];
const fps = 25;
const eventToleranceSec = 3;

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trimEnd().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""]),
    );
  });
}

function toCsv(rows, headers) {
  const escape = (value) => {
    if (value === null || value === undefined) return "";
    const text = String(value);
    return /[",\r\n]/.test(text)
      ? `"${text.replaceAll('"', '""')}"`
      : text;
  };
  return `\uFEFF${headers.join(",")}\r\n${rows
    .map((row) => headers.map((header) => escape(row[header])).join(","))
    .join("\r\n")}\r\n`;
}

function round(value, digits = 6) {
  return Number(Number(value).toFixed(digits));
}

function hms(seconds) {
  const totalMs = Math.round(seconds * 1000);
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const secs = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

function transitions(labels, frameIndexes, timestamps) {
  const result = [];
  for (let index = 1; index < labels.length; index += 1) {
    if (labels[index] === labels[index - 1]) continue;
    result.push({
      frame: frameIndexes[index],
      sec: timestamps[index],
      from: labels[index - 1],
      to: labels[index],
    });
  }
  return result;
}

function matchEvents(gtEvents, predEvents, toleranceFrames) {
  const usedPredictions = new Set();
  const rows = [];
  for (let index = 0; index < gtEvents.length; index += 1) {
    const gt = gtEvents[index];
    let bestIndex = -1;
    let bestDistance = Infinity;
    for (let predIndex = 0; predIndex < predEvents.length; predIndex += 1) {
      if (usedPredictions.has(predIndex)) continue;
      const pred = predEvents[predIndex];
      if (pred.from !== gt.from || pred.to !== gt.to) continue;
      const distance = Math.abs(pred.frame - gt.frame);
      if (distance <= toleranceFrames && distance < bestDistance) {
        bestIndex = predIndex;
        bestDistance = distance;
      }
    }
    if (bestIndex >= 0) usedPredictions.add(bestIndex);
    const pred = bestIndex >= 0 ? predEvents[bestIndex] : null;
    rows.push({
      event_id: index + 1,
      gt_frame: gt.frame,
      gt_sec: gt.sec,
      gt_hms: hms(gt.sec),
      from_label: gt.from,
      to_label: gt.to,
      pred_frame: pred?.frame ?? null,
      pred_sec: pred?.sec ?? null,
      pred_hms: pred ? hms(pred.sec) : "",
      timing_error_sec: pred ? pred.sec - gt.sec : null,
      matched: pred ? 1 : 0,
    });
  }
  return { rows, matchedCount: usedPredictions.size };
}

function excelColumn(index) {
  let value = index;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

const sourceFrameRows = parseCsv(await fs.readFile(frameComparisonPath, "utf8"));
const originalSummaryRows = parseCsv(
  await fs.readFile(originalSummaryPath, "utf8"),
);
const originalSummaryByVariant = new Map(
  originalSummaryRows.map((row) => [row.variant, row]),
);

const frameIndexes = sourceFrameRows.map((row) => Number(row.frame_index));
const timestamps = sourceFrameRows.map((row) => Number(row.timestamp_sec));
const groundTruthCounts = sourceFrameRows.map((row) =>
  Number(row.ground_truth_count),
);
const predictions = Object.fromEntries(
  variants.map(({ key }) => [key, sourceFrameRows.map((row) => Number(row[key]))]),
);

if (sourceFrameRows.length !== 53964) {
  throw new Error(
    `공통 프레임 수가 예상과 다릅니다: ${sourceFrameRows.length} (예상 53964)`,
  );
}
for (const { key } of variants) {
  if (predictions[key].some((value) => !Number.isInteger(value) || value < 0)) {
    throw new Error(`${key}에 유효하지 않은 예측 인원수가 있습니다.`);
  }
}

function analyze(spec) {
  const predictedKeys = ["miss", ...spec.classKeys];
  const gtLabels = groundTruthCounts.map(spec.mapGroundTruth);
  const gtEvents = transitions(gtLabels, frameIndexes, timestamps);
  const distribution = Object.fromEntries(
    spec.classKeys.map((key) => [
      key,
      gtLabels.filter((label) => label === key).length,
    ]),
  );
  const majorityBaseline = Math.max(...Object.values(distribution)) / gtLabels.length;
  const summary = [];
  const classMetrics = [];
  const confusionRows = [];
  const eventRows = [];
  const errorSegments = [];
  const frameRows = sourceFrameRows.map((row, index) => ({
    frame_index: frameIndexes[index],
    timestamp_sec: timestamps[index],
    timestamp_hms: row.timestamp_hms,
    ground_truth_count: groundTruthCounts[index],
    ground_truth_stage: spec.classLabels[gtLabels[index]],
  }));

  for (const { key: variant, threshold } of variants) {
    const predLabels = predictions[variant].map(spec.mapPrediction);
    const confusion = Object.fromEntries(
      spec.classKeys.map((actual) => [
        actual,
        Object.fromEntries(predictedKeys.map((predicted) => [predicted, 0])),
      ]),
    );
    let correctCount = 0;
    let missCount = 0;
    let absoluteStageErrorSum = 0;
    let squaredStageErrorSum = 0;
    const level = Object.fromEntries(
      spec.classKeys.map((classKey, index) => [classKey, index + 1]),
    );
    level.miss = 0;

    for (let index = 0; index < gtLabels.length; index += 1) {
      const actual = gtLabels[index];
      const predicted = predLabels[index];
      confusion[actual][predicted] += 1;
      if (actual === predicted) correctCount += 1;
      if (predicted === "miss") missCount += 1;
      const stageError = level[predicted] - level[actual];
      absoluteStageErrorSum += Math.abs(stageError);
      squaredStageErrorSum += stageError * stageError;
      frameRows[index][`${variant}_count`] = predictions[variant][index];
      frameRows[index][`${variant}_stage`] = spec.classLabels[predicted];
      frameRows[index][`${variant}_correct`] = Number(actual === predicted);
    }

    const perClass = spec.classKeys.map((classKey) => {
      const support = predictedKeys.reduce(
        (sum, predicted) => sum + confusion[classKey][predicted],
        0,
      );
      const tp = confusion[classKey][classKey];
      const fp = spec.classKeys
        .filter((actual) => actual !== classKey)
        .reduce((sum, actual) => sum + confusion[actual][classKey], 0);
      const fn = support - tp;
      const precision = tp + fp ? tp / (tp + fp) : 0;
      const recall = tp + fn ? tp / (tp + fn) : 0;
      const f1 = precision + recall
        ? (2 * precision * recall) / (precision + recall)
        : 0;
      const row = {
        variant,
        threshold,
        class_key: classKey,
        class_label: spec.classLabels[classKey],
        support,
        tp,
        fp,
        fn,
        precision,
        recall,
        f1,
      };
      classMetrics.push(row);
      return row;
    });

    const predEvents = transitions(predLabels, frameIndexes, timestamps);
    const matched = matchEvents(
      gtEvents,
      predEvents,
      eventToleranceSec * fps,
    );
    const matchedTimingErrors = matched.rows
      .filter((row) => row.matched)
      .map((row) => Number(row.timing_error_sec));
    for (const row of matched.rows) {
      eventRows.push({
        variant,
        threshold,
        ...row,
        from_label: spec.classLabels[row.from_label],
        to_label: spec.classLabels[row.to_label],
      });
    }

    let segmentStart = null;
    let segmentActual = null;
    let segmentPredicted = null;
    const closeSegment = (endIndex) => {
      errorSegments.push({
        variant,
        threshold,
        start_frame: frameIndexes[segmentStart],
        end_frame: frameIndexes[endIndex],
        start_sec: timestamps[segmentStart],
        end_sec: timestamps[endIndex],
        start_hms: hms(timestamps[segmentStart]),
        end_hms: hms(timestamps[endIndex]),
        duration_sec:
          timestamps[endIndex] - timestamps[segmentStart] + 1 / fps,
        actual_stage: spec.classLabels[segmentActual],
        predicted_stage: spec.classLabels[segmentPredicted],
        error_type: segmentPredicted === "miss" ? "미검출" : "단계 오분류",
      });
    };
    for (let index = 0; index < gtLabels.length; index += 1) {
      const actual = gtLabels[index];
      const predicted = predLabels[index];
      const gap =
        index > 0 && frameIndexes[index] !== frameIndexes[index - 1] + 1;
      const changed =
        segmentStart !== null &&
        (gap || actual !== segmentActual || predicted !== segmentPredicted);
      if (changed) {
        closeSegment(index - 1);
        segmentStart = null;
      }
      if (actual !== predicted && segmentStart === null) {
        segmentStart = index;
        segmentActual = actual;
        segmentPredicted = predicted;
      }
    }
    if (segmentStart !== null) closeSegment(gtLabels.length - 1);

    const macroPrecision =
      perClass.reduce((sum, row) => sum + row.precision, 0) / perClass.length;
    const balancedAccuracy =
      perClass.reduce((sum, row) => sum + row.recall, 0) / perClass.length;
    const macroF1 =
      perClass.reduce((sum, row) => sum + row.f1, 0) / perClass.length;
    const weightedF1 =
      perClass.reduce((sum, row) => sum + row.f1 * row.support, 0) /
      gtLabels.length;
    const original = originalSummaryByVariant.get(variant);
    if (!original) throw new Error(`${variant}의 기존 요약 결과를 찾지 못했습니다.`);
    const timingAbsErrorSum = matchedTimingErrors.reduce(
      (sum, value) => sum + Math.abs(value),
      0,
    );
    const row = {
      variant,
      threshold,
      frames: gtLabels.length,
      original_count_mae: Number(original.mae),
      original_exact_count_accuracy: Number(original.exactAccuracy),
      correct_count: correctCount,
      miss_count: missCount,
      stage_absolute_error_sum: absoluteStageErrorSum,
      stage_squared_error_sum: squaredStageErrorSum,
      stage_accuracy: correctCount / gtLabels.length,
      balanced_accuracy: balancedAccuracy,
      macro_precision: macroPrecision,
      macro_recall: balancedAccuracy,
      macro_f1: macroF1,
      weighted_f1: weightedF1,
      stage_mae: absoluteStageErrorSum / gtLabels.length,
      stage_rmse: Math.sqrt(squaredStageErrorSum / gtLabels.length),
      miss_ratio: missCount / gtLabels.length,
      gt_events: gtEvents.length,
      pred_events: predEvents.length,
      matched_events: matched.matchedCount,
      timing_abs_error_sum: timingAbsErrorSum,
      event_recall: gtEvents.length ? matched.matchedCount / gtEvents.length : 0,
      event_precision: predEvents.length
        ? matched.matchedCount / predEvents.length
        : 0,
      event_timing_mae_sec: matchedTimingErrors.length
        ? timingAbsErrorSum / matchedTimingErrors.length
        : null,
    };
    summary.push(row);

    for (const actual of spec.classKeys) {
      const total = predictedKeys.reduce(
        (sum, predicted) => sum + confusion[actual][predicted],
        0,
      );
      confusionRows.push({
        variant,
        threshold,
        actual_key: actual,
        actual_label: spec.classLabels[actual],
        ...Object.fromEntries(
          predictedKeys.map((predicted) => [
            `pred_${predicted}`,
            confusion[actual][predicted],
          ]),
        ),
        total,
        recall: total ? confusion[actual][actual] / total : 0,
      });
    }
  }

  summary.sort(
    (left, right) =>
      right.balanced_accuracy - left.balanced_accuracy ||
      right.macro_f1 - left.macro_f1 ||
      right.stage_accuracy - left.stage_accuracy,
  );
  errorSegments.sort(
    (left, right) =>
      right.duration_sec - left.duration_sec || left.threshold - right.threshold,
  );
  const bestBalanced = summary[0];
  const bestAccuracy = [...summary].sort(
    (left, right) => right.stage_accuracy - left.stage_accuracy,
  )[0];
  const bestMacroF1 = [...summary].sort(
    (left, right) => right.macro_f1 - left.macro_f1,
  )[0];
  const perClassAccuracy = summary.flatMap((summaryRow) =>
    classMetrics
      .filter((row) => row.variant === summaryRow.variant)
      .map((row) => ({
        variant: row.variant,
        threshold: row.threshold,
        actual_class_key: row.class_key,
        actual_class_label: row.class_label,
        correctly_detected_frames: row.tp,
        total_actual_frames: row.support,
        incorrect_frames: row.fn,
        class_accuracy: row.recall,
      })),
  );

  return {
    spec,
    predictedKeys,
    distribution,
    majorityBaseline,
    summary,
    classMetrics,
    perClassAccuracy,
    confusionRows,
    eventRows,
    errorSegments,
    frameRows,
    bestBalanced,
    bestAccuracy,
    bestMacroF1,
  };
}

function analysisLines(result) {
  const {
    spec,
    distribution,
    majorityBaseline,
    summary,
    bestBalanced,
    bestAccuracy,
    bestMacroF1,
    perClassAccuracy,
  } = result;
  const distributionText = spec.classKeys
    .map(
      (key) =>
        `${spec.classLabels[key]} ${distribution[key].toLocaleString()}프레임 (${(
          (distribution[key] / sourceFrameRows.length) *
          100
        ).toFixed(2)}%)`,
    )
    .join(", ");
  const imbalanceNote =
    majorityBaseline >= 0.8
      ? `- 최다 클래스만 항상 예측해도 ${(majorityBaseline * 100).toFixed(2)}% 정확도가 나오므로 전체 정확도만으로 모델을 선택하면 안 된다.`
      : `- 최다 클래스 기준선은 ${(majorityBaseline * 100).toFixed(2)}%이며, 클래스 불균형을 고려해 Balanced Accuracy와 Macro F1을 우선한다.`;
  return [
    `# ${spec.title} 결과 해석`,
    "",
    `정답 단계 정의: ${spec.subtitle}`,
    `기존 activation 0.20, 0.30, 0.40, 0.50의 저장 예측을 재사용했으며 모델은 다시 실행하지 않았다. 공통 ${sourceFrameRows.length.toLocaleString()}개 프레임만 평가했다.`,
    "",
    "## 정답 분포",
    "",
    `- ${distributionText}`,
    imbalanceNote,
    "- 예측 0명은 낮은 단계에 합치지 않고 MISS로 처리하여 미검출이 정답 처리되지 않도록 했다.",
    "",
    "## 핵심 결과",
    "",
    "| 순위 | Activation | 기존 정확 인원 일치율 | 단계 정확도 | Balanced Accuracy | Macro F1 | Stage MAE | MISS 비율 | 이벤트 Recall | 이벤트 Precision |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...summary.map(
      (row, index) =>
        `| ${index + 1} | ${row.threshold.toFixed(2)} | ${(row.original_exact_count_accuracy * 100).toFixed(2)}% | ${(row.stage_accuracy * 100).toFixed(2)}% | ${(row.balanced_accuracy * 100).toFixed(2)}% | ${(row.macro_f1 * 100).toFixed(2)}% | ${row.stage_mae.toFixed(4)} | ${(row.miss_ratio * 100).toFixed(2)}% | ${(row.event_recall * 100).toFixed(2)}% | ${(row.event_precision * 100).toFixed(2)}% |`,
    ),
    "",
    "## 정답 단계별 정확 탐지율",
    "",
    "정확 탐지율은 정답이 해당 단계인 프레임 중 예측도 같은 단계로 나온 비율이다. 계산식은 `정확 탐지 프레임 / 해당 정답 전체 프레임`이며 분류 지표의 Recall과 같다.",
    "",
    `| Activation | ${spec.classKeys.map((key) => spec.classLabels[key]).join(" | ")} |`,
    `|---:|${spec.classKeys.map(() => "---:").join("|")}|`,
    ...summary.map((summaryRow) => {
      const values = spec.classKeys.map((classKey) => {
        const row = perClassAccuracy.find(
          (item) =>
            item.variant === summaryRow.variant &&
            item.actual_class_key === classKey,
        );
        return `${(row.class_accuracy * 100).toFixed(2)}% (${row.correctly_detected_frames.toLocaleString()}/${row.total_actual_frames.toLocaleString()})`;
      });
      return `| ${summaryRow.threshold.toFixed(2)} | ${values.join(" | ")} |`;
    }),
    "",
    "## 해석",
    "",
    `- 주 지표인 Balanced Accuracy 기준 최적 activation은 ${bestBalanced.threshold.toFixed(2)}이며 ${(bestBalanced.balanced_accuracy * 100).toFixed(2)}%이다.`,
    `- 전체 단계 정확도 기준 최적 activation은 ${bestAccuracy.threshold.toFixed(2)}이며 ${(bestAccuracy.stage_accuracy * 100).toFixed(2)}%이다.`,
    `- Macro F1 기준 최적 activation은 ${bestMacroF1.threshold.toFixed(2)}이며 ${(bestMacroF1.macro_f1 * 100).toFixed(2)}%이다.`,
    bestBalanced.threshold === bestAccuracy.threshold
      ? "- 전체 정확도와 클래스 균형 지표가 같은 activation을 선택했다."
      : "- 전체 정확도와 Balanced Accuracy의 최적 activation이 다르다. 다수 클래스에 치우치지 않는 설정 선택에는 Balanced Accuracy와 Macro F1이 더 적합하다.",
    `- 권장 설정 ${bestBalanced.threshold.toFixed(2)}의 MISS 비율은 ${(bestBalanced.miss_ratio * 100).toFixed(2)}%, 단계 MAE는 ${bestBalanced.stage_mae.toFixed(4)}단계이다.`,
    "- 단계 수를 줄이면 문제가 쉬워져 정확도가 올라갈 수 있으므로, 기존 정확 인원 일치율과 단계 정확도의 절대값을 직접적인 성능 향상으로 해석하면 안 된다.",
    "",
    "## 제한사항",
    "",
    "- Ground Truth는 전체 화면 기준이고 기존 모델은 다각형 ROI의 bottom_center 기준이므로 오차에는 ROI 정의 차이가 포함된다.",
    "- Ground Truth는 1차 수동 검수 상태이므로 논문·보고서 최종 수치 사용 전 독립 검수자의 2차 검수가 필요하다.",
    "- 동일한 한 개 영상에서 설정을 비교한 결과이므로 일반화 성능은 별도 영상이나 분리된 평가 구간에서 확인해야 한다.",
    "- 연속 프레임은 서로 독립 표본이 아니므로 통계적 신뢰구간이 필요하면 시간 블록 단위 부트스트랩을 사용해야 한다.",
    "",
  ];
}

async function buildWorkbook(result) {
  const { spec, predictedKeys, summary, classMetrics, confusionRows } = result;
  const workbook = Workbook.create();
  const summarySheet = workbook.worksheets.add("Summary");
  const classAccuracySheet = workbook.worksheets.add("ClassAccuracy");
  const confusionSheet = workbook.worksheets.add("ConfusionMatrix");
  const classSheet = workbook.worksheets.add("ClassMetrics");
  const calculationSheet = workbook.worksheets.add("Calculations");
  const errorSheet = workbook.worksheets.add("TopErrors");
  const protocolSheet = workbook.worksheets.add("Protocol");

  const navy = "#183153";
  const blue = "#2563EB";
  const teal = "#0F766E";
  const green = "#15803D";
  const paleBlue = "#EAF2FF";
  const paleGreen = "#E8F5EE";
  const paleRed = "#FDECEC";
  const paleAmber = "#FFF4D6";
  const gray = "#667085";
  const line = "#D0D5DD";

  for (const sheet of [
    summarySheet,
    classAccuracySheet,
    confusionSheet,
    classSheet,
    calculationSheet,
    errorSheet,
    protocolSheet,
  ]) {
    sheet.showGridLines = false;
  }

  classSheet.getRange("A1:J1").merge();
  classSheet.getRange("A1").values = [[`${spec.title} · 클래스별 성능`]];
  classSheet.getRange("A1:J1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    verticalAlignment: "center",
  };
  classSheet.getRange("A1:J1").format.rowHeight = 32;
  const classHeaders = [
    "Activation",
    "Threshold",
    "Class",
    "Support",
    "TP",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
  ];
  classSheet.getRangeByIndexes(2, 0, 1, classHeaders.length).values = [
    classHeaders,
  ];
  classSheet.getRangeByIndexes(2, 0, 1, classHeaders.length).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    wrapText: true,
  };
  const classRowRanges = new Map();
  const classMetricRowByKey = new Map();
  let classOutputRow = 4;
  for (const summaryRow of summary) {
    const rows = classMetrics.filter(
      (row) => row.variant === summaryRow.variant,
    );
    const startRow = classOutputRow;
    for (const row of rows) {
      classSheet.getRange(`A${classOutputRow}:G${classOutputRow}`).values = [[
        row.variant,
        row.threshold,
        row.class_label,
        row.support,
        row.tp,
        row.fp,
        row.fn,
      ]];
      classSheet.getRange(`H${classOutputRow}`).formulas = [[
        `=IFERROR(E${classOutputRow}/(E${classOutputRow}+F${classOutputRow}),0)`,
      ]];
      classSheet.getRange(`I${classOutputRow}`).formulas = [[
        `=IFERROR(E${classOutputRow}/(E${classOutputRow}+G${classOutputRow}),0)`,
      ]];
      classSheet.getRange(`J${classOutputRow}`).formulas = [[
        `=IFERROR(2*H${classOutputRow}*I${classOutputRow}/(H${classOutputRow}+I${classOutputRow}),0)`,
      ]];
      classMetricRowByKey.set(
        `${row.variant}:${row.class_key}`,
        classOutputRow,
      );
      classOutputRow += 1;
    }
    classRowRanges.set(summaryRow.variant, {
      start: startRow,
      end: classOutputRow - 1,
    });
  }
  classSheet.getRange(`B4:B${classOutputRow - 1}`).format.numberFormat = "0.00";
  classSheet.getRange(`D4:G${classOutputRow - 1}`).format.numberFormat = "#,##0";
  classSheet.getRange(`H4:J${classOutputRow - 1}`).format.numberFormat = "0.00%";
  classSheet.getRange(`A3:J${classOutputRow - 1}`).format.borders = {
    preset: "inside",
    style: "thin",
    color: line,
  };
  classSheet
    .getRange(`H4:J${classOutputRow - 1}`)
    .conditionalFormats.add("colorScale", {
      colors: ["#F8696B", "#FFEB84", "#63BE7B"],
      thresholds: ["min", "50%", "max"],
    });
  classSheet.freezePanes.freezeRows(3);

  const classAccuracyWidth = 1 + spec.classKeys.length * 3;
  const classAccuracyEndColumn = excelColumn(classAccuracyWidth);
  classAccuracySheet.getRange(`A1:${classAccuracyEndColumn}1`).merge();
  classAccuracySheet.getRange("A1").values = [[
    `${spec.title} · 정답 단계별 정확 탐지율`,
  ]];
  classAccuracySheet.getRange(`A1:${classAccuracyEndColumn}1`).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    verticalAlignment: "center",
  };
  classAccuracySheet.getRange(`A2:${classAccuracyEndColumn}2`).merge();
  classAccuracySheet.getRange("A2").values = [[
    "정답이 해당 단계인 프레임 중 예측도 같은 단계인 비율 = 정확 탐지 프레임 / 해당 정답 전체 프레임",
  ]];
  classAccuracySheet.getRange(`A2:${classAccuracyEndColumn}2`).format = {
    fill: "#F2F4F7",
    font: { color: gray, italic: true },
  };
  const classAccuracyHeaders = ["Activation"];
  for (const classKey of spec.classKeys) {
    classAccuracyHeaders.push(
      `${spec.classLabels[classKey]} 정확 탐지`,
      `${spec.classLabels[classKey]} 정답 전체`,
      `${spec.classLabels[classKey]} 정확 탐지율`,
    );
  }
  classAccuracySheet.getRangeByIndexes(
    3,
    0,
    1,
    classAccuracyHeaders.length,
  ).values = [classAccuracyHeaders];
  classAccuracySheet.getRangeByIndexes(
    3,
    0,
    1,
    classAccuracyHeaders.length,
  ).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    wrapText: true,
  };
  for (let summaryIndex = 0; summaryIndex < summary.length; summaryIndex += 1) {
    const summaryRow = summary[summaryIndex];
    const excelRow = 5 + summaryIndex;
    classAccuracySheet.getRange(`A${excelRow}`).values = [[
      summaryRow.threshold,
    ]];
    for (let classIndex = 0; classIndex < spec.classKeys.length; classIndex += 1) {
      const classKey = spec.classKeys[classIndex];
      const sourceRow = classMetricRowByKey.get(
        `${summaryRow.variant}:${classKey}`,
      );
      const correctColumn = excelColumn(2 + classIndex * 3);
      const totalColumn = excelColumn(3 + classIndex * 3);
      const accuracyColumn = excelColumn(4 + classIndex * 3);
      classAccuracySheet.getRange(`${correctColumn}${excelRow}`).formulas = [[
        `='ClassMetrics'!E${sourceRow}`,
      ]];
      classAccuracySheet.getRange(`${totalColumn}${excelRow}`).formulas = [[
        `='ClassMetrics'!D${sourceRow}`,
      ]];
      classAccuracySheet.getRange(`${accuracyColumn}${excelRow}`).formulas = [[
        `=IFERROR(${correctColumn}${excelRow}/${totalColumn}${excelRow},0)`,
      ]];
    }
  }
  const classAccuracyLastRow = 4 + summary.length;
  classAccuracySheet.getRange(`A5:A${classAccuracyLastRow}`).format.numberFormat =
    "0.00";
  for (let classIndex = 0; classIndex < spec.classKeys.length; classIndex += 1) {
    const correctColumn = excelColumn(2 + classIndex * 3);
    const totalColumn = excelColumn(3 + classIndex * 3);
    const accuracyColumn = excelColumn(4 + classIndex * 3);
    classAccuracySheet.getRange(
      `${correctColumn}5:${totalColumn}${classAccuracyLastRow}`,
    ).format.numberFormat = "#,##0";
    classAccuracySheet.getRange(
      `${accuracyColumn}5:${accuracyColumn}${classAccuracyLastRow}`,
    ).format.numberFormat = "0.00%";
    classAccuracySheet
      .getRange(`${accuracyColumn}5:${accuracyColumn}${classAccuracyLastRow}`)
      .conditionalFormats.add("colorScale", {
        colors: ["#F8696B", "#FFEB84", "#63BE7B"],
        thresholds: ["min", "50%", "max"],
      });
  }
  classAccuracySheet
    .getRange(`A4:${classAccuracyEndColumn}${classAccuracyLastRow}`)
    .format.borders = {
    preset: "inside",
    style: "thin",
    color: line,
  };
  const classAccuracyChart = classAccuracySheet.charts.add("bar", {
    chartType: "bar",
    title: "Activation별 정답 단계 정확 탐지율",
    hasLegend: true,
  });
  const chartColors = [blue, teal, green];
  for (let classIndex = 0; classIndex < spec.classKeys.length; classIndex += 1) {
    const accuracyColumn = excelColumn(4 + classIndex * 3);
    const series = classAccuracyChart.series.add(
      spec.classLabels[spec.classKeys[classIndex]],
    );
    series.categoryFormula = `'ClassAccuracy'!$A$5:$A$${classAccuracyLastRow}`;
    series.formula = `'ClassAccuracy'!$${accuracyColumn}$5:$${accuracyColumn}$${classAccuracyLastRow}`;
    series.fill = chartColors[classIndex];
  }
  classAccuracyChart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };
  const chartStartColumn = excelColumn(classAccuracyWidth + 2);
  const chartEndColumn = excelColumn(classAccuracyWidth + 9);
  classAccuracyChart.setPosition(
    `${chartStartColumn}3`,
    `${chartEndColumn}18`,
  );
  classAccuracySheet.freezePanes.freezeRows(4);

  calculationSheet.getRange("A1:W1").merge();
  calculationSheet.getRange("A1").values = [[
    `${spec.title} · 재현 가능한 집계와 수식`,
  ]];
  calculationSheet.getRange("A1:W1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
  };
  const calculationHeaders = [
    "Variant",
    "Threshold",
    "Frames",
    "Original Exact",
    "Original MAE",
    "Correct",
    "MISS",
    "Stage Abs Error Sum",
    "Stage Sq Error Sum",
    "GT Events",
    "Pred Events",
    "Matched Events",
    "Timing Abs Error Sum",
    "Stage Accuracy",
    "Balanced Accuracy",
    "Macro F1",
    "Weighted F1",
    "Stage MAE",
    "Stage RMSE",
    "MISS Ratio",
    "Event Recall",
    "Event Precision",
    "Event Timing MAE",
  ];
  calculationSheet.getRangeByIndexes(2, 0, 1, calculationHeaders.length).values = [
    calculationHeaders,
  ];
  calculationSheet.getRange("A3:W3").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    wrapText: true,
  };
  const calculationRowByVariant = new Map();
  for (let index = 0; index < summary.length; index += 1) {
    const row = summary[index];
    const excelRow = 4 + index;
    calculationRowByVariant.set(row.variant, excelRow);
    calculationSheet.getRange(`A${excelRow}:M${excelRow}`).values = [[
      row.variant,
      row.threshold,
      row.frames,
      row.original_exact_count_accuracy,
      row.original_count_mae,
      row.correct_count,
      row.miss_count,
      row.stage_absolute_error_sum,
      row.stage_squared_error_sum,
      row.gt_events,
      row.pred_events,
      row.matched_events,
      row.timing_abs_error_sum,
    ]];
    const classRange = classRowRanges.get(row.variant);
    calculationSheet.getRange(`N${excelRow}:W${excelRow}`).formulas = [[
      `=F${excelRow}/C${excelRow}`,
      `=AVERAGE('ClassMetrics'!I${classRange.start}:I${classRange.end})`,
      `=AVERAGE('ClassMetrics'!J${classRange.start}:J${classRange.end})`,
      `=SUMPRODUCT('ClassMetrics'!J${classRange.start}:J${classRange.end},'ClassMetrics'!D${classRange.start}:D${classRange.end})/C${excelRow}`,
      `=H${excelRow}/C${excelRow}`,
      `=SQRT(I${excelRow}/C${excelRow})`,
      `=G${excelRow}/C${excelRow}`,
      `=IFERROR(L${excelRow}/J${excelRow},0)`,
      `=IFERROR(L${excelRow}/K${excelRow},0)`,
      `=IFERROR(M${excelRow}/L${excelRow},0)`,
    ]];
  }
  calculationSheet.getRange("B4:B7").format.numberFormat = "0.00";
  calculationSheet.getRange("C4:C7").format.numberFormat = "#,##0";
  calculationSheet.getRange("D4:D7").format.numberFormat = "0.00%";
  calculationSheet.getRange("E4:E7").format.numberFormat = "0.0000";
  calculationSheet.getRange("F4:M7").format.numberFormat = "#,##0.00";
  calculationSheet.getRange("N4:Q7").format.numberFormat = "0.00%";
  calculationSheet.getRange("R4:S7").format.numberFormat = "0.0000";
  calculationSheet.getRange("T4:V7").format.numberFormat = "0.00%";
  calculationSheet.getRange("W4:W7").format.numberFormat = "0.00";
  calculationSheet.getRange("A3:W7").format.borders = {
    preset: "inside",
    style: "thin",
    color: line,
  };
  calculationSheet.freezePanes.freezeRows(3);

  summarySheet.getRange("A1:K1").merge();
  summarySheet.getRange("A1").values = [[spec.title]];
  summarySheet.getRange("A1:K1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 18 },
    verticalAlignment: "center",
  };
  summarySheet.getRange("A1:K1").format.rowHeight = 34;
  summarySheet.getRange("A2:K2").merge();
  summarySheet.getRange("A2").values = [[
    `${spec.subtitle} · 기존 예측 재사용 · 공통 ${sourceFrameRows.length.toLocaleString()}프레임`,
  ]];
  summarySheet.getRange("A2:K2").format = {
    fill: "#F2F4F7",
    font: { color: gray, italic: true },
  };

  const bestCalculationRow = calculationRowByVariant.get(
    result.bestBalanced.variant,
  );
  summarySheet.getRange("A4:B4").values = [["권장 Activation", null]];
  summarySheet.getRange("B4").formulas = [[`='Calculations'!B${bestCalculationRow}`]];
  summarySheet.getRange("D4:E4").values = [["Best Balanced Acc.", null]];
  summarySheet.getRange("E4").formulas = [[`='Calculations'!O${bestCalculationRow}`]];
  summarySheet.getRange("G4:H4").values = [["Best Macro F1", null]];
  summarySheet.getRange("H4").formulas = [[`='Calculations'!P${bestCalculationRow}`]];
  summarySheet.getRange("J4:K4").values = [[
    "Majority 기준선",
    result.majorityBaseline,
  ]];
  for (const address of ["A4:B4", "D4:E4", "G4:H4", "J4:K4"]) {
    summarySheet.getRange(address).format = {
      fill: paleGreen,
      borders: { preset: "outside", style: "thin", color: teal },
    };
  }
  summarySheet.getRange("A4,D4,G4,J4").format = {
    font: { bold: true, color: teal },
  };
  summarySheet.getRange("B4").format.numberFormat = "0.00";
  for (const address of ["E4", "H4", "K4"]) {
    summarySheet.getRange(address).format.numberFormat = "0.00%";
  }

  const summaryHeaders = [
    "순위",
    "Activation",
    "기존 정확 인원 일치율",
    "단계 정확도",
    "Balanced Accuracy",
    "Macro F1",
    "Weighted F1",
    "Stage MAE",
    "MISS 비율",
    "이벤트 Recall",
    "이벤트 Precision",
  ];
  summarySheet.getRangeByIndexes(6, 0, 1, summaryHeaders.length).values = [
    summaryHeaders,
  ];
  summarySheet.getRange("A7:K7").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    wrapText: true,
  };
  for (let index = 0; index < summary.length; index += 1) {
    const row = summary[index];
    const excelRow = 8 + index;
    const calcRow = calculationRowByVariant.get(row.variant);
    summarySheet.getRange(`A${excelRow}`).values = [[index + 1]];
    summarySheet.getRange(`B${excelRow}:K${excelRow}`).formulas = [[
      `='Calculations'!B${calcRow}`,
      `='Calculations'!D${calcRow}`,
      `='Calculations'!N${calcRow}`,
      `='Calculations'!O${calcRow}`,
      `='Calculations'!P${calcRow}`,
      `='Calculations'!Q${calcRow}`,
      `='Calculations'!R${calcRow}`,
      `='Calculations'!T${calcRow}`,
      `='Calculations'!U${calcRow}`,
      `='Calculations'!V${calcRow}`,
    ]];
  }
  summarySheet.getRange("B8:B11").format.numberFormat = "0.00";
  summarySheet.getRange("C8:G11").format.numberFormat = "0.00%";
  summarySheet.getRange("H8:H11").format.numberFormat = "0.0000";
  summarySheet.getRange("I8:K11").format.numberFormat = "0.00%";
  summarySheet.getRange("A7:K11").format.borders = {
    preset: "inside",
    style: "thin",
    color: line,
  };
  summarySheet.getRange("B8:K8").format.fill = paleGreen;
  summarySheet.getRange("D8:F11").conditionalFormats.add("colorScale", {
    colors: ["#F8696B", "#FFEB84", "#63BE7B"],
    thresholds: ["min", "50%", "max"],
  });

  summarySheet.getRange("A14:K14").merge();
  summarySheet.getRange("A14").values = [["결과 해석"]];
  summarySheet.getRange("A14:K14").format = {
    fill: paleBlue,
    font: { bold: true, color: navy },
  };
  const interpretation = analysisLines(result)
    .slice(analysisLines(result).indexOf("## 해석") + 2)
    .filter((line) => line.startsWith("- "))
    .slice(0, 6)
    .map((line, index) => [index + 1, line.slice(2)]);
  summarySheet
    .getRangeByIndexes(14, 0, interpretation.length, 2)
    .values = interpretation;
  summarySheet.getRange(`A15:A${14 + interpretation.length}`).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
  };
  summarySheet.getRange(`B15:K${14 + interpretation.length}`).merge(true);
  summarySheet.getRange(`B15:K${14 + interpretation.length}`).format = {
    wrapText: true,
    verticalAlignment: "top",
  };
  summarySheet.getRange(`A15:K${14 + interpretation.length}`).format.borders = {
    preset: "inside",
    style: "thin",
    color: line,
  };
  summarySheet.getRange(`A15:K${14 + interpretation.length}`).format.rowHeight = 32;

  const chart = summarySheet.charts.add("bar", {
    chartType: "bar",
    title: "Activation별 단계 분류 성능",
    hasLegend: true,
  });
  const stageAccuracySeries = chart.series.add("단계 정확도");
  stageAccuracySeries.categoryFormula = "'Summary'!$B$8:$B$11";
  stageAccuracySeries.formula = "'Summary'!$D$8:$D$11";
  stageAccuracySeries.fill = blue;
  const balancedSeries = chart.series.add("Balanced Accuracy");
  balancedSeries.categoryFormula = "'Summary'!$B$8:$B$11";
  balancedSeries.formula = "'Summary'!$E$8:$E$11";
  balancedSeries.fill = teal;
  const macroF1Series = chart.series.add("Macro F1");
  macroF1Series.categoryFormula = "'Summary'!$B$8:$B$11";
  macroF1Series.formula = "'Summary'!$F$8:$F$11";
  macroF1Series.fill = green;
  chart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };
  chart.setPosition("M3", "T18");
  summarySheet.freezePanes.freezeRows(7);

  const confusionWidth = predictedKeys.length + 3;
  const confusionEndColumn = excelColumn(confusionWidth);
  confusionSheet.getRange(`A1:${confusionEndColumn}1`).merge();
  confusionSheet.getRange("A1").values = [[
    `${spec.title} · 혼동행렬 (프레임 수)`,
  ]];
  confusionSheet.getRange(`A1:${confusionEndColumn}1`).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
  };
  let confusionStart = 3;
  for (const summaryRow of summary) {
    const rows = confusionRows.filter(
      (row) => row.variant === summaryRow.variant,
    );
    confusionSheet
      .getRangeByIndexes(confusionStart - 1, 0, 1, confusionWidth)
      .merge();
    confusionSheet.getRangeByIndexes(
      confusionStart - 1,
      0,
      1,
      confusionWidth,
    ).values = [[`Activation ${summaryRow.threshold.toFixed(2)}`]];
    confusionSheet.getRangeByIndexes(
      confusionStart - 1,
      0,
      1,
      confusionWidth,
    ).format = { fill: paleBlue, font: { bold: true, color: navy } };
    confusionSheet.getRangeByIndexes(
      confusionStart,
      0,
      1,
      confusionWidth,
    ).values = [[
      "Actual \\ Pred",
      ...predictedKeys.map((key) => spec.classLabels[key]),
      "Total",
      "Recall",
    ]];
    confusionSheet.getRangeByIndexes(
      confusionStart,
      0,
      1,
      confusionWidth,
    ).format = {
      fill: navy,
      font: { bold: true, color: "#FFFFFF" },
      horizontalAlignment: "center",
      wrapText: true,
    };
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
      const item = rows[rowIndex];
      const excelRow = confusionStart + 2 + rowIndex;
      confusionSheet.getRangeByIndexes(
        confusionStart + 1 + rowIndex,
        0,
        1,
        predictedKeys.length + 1,
      ).values = [[
        item.actual_label,
        ...predictedKeys.map((key) => item[`pred_${key}`]),
      ]];
      const totalColumn = excelColumn(predictedKeys.length + 2);
      const recallColumn = excelColumn(predictedKeys.length + 3);
      confusionSheet.getRange(`${totalColumn}${excelRow}`).formulas = [[
        `=SUM(B${excelRow}:${excelColumn(predictedKeys.length + 1)}${excelRow})`,
      ]];
      const diagonalColumn = excelColumn(
        3 + spec.classKeys.indexOf(item.actual_key),
      );
      confusionSheet.getRange(`${recallColumn}${excelRow}`).formulas = [[
        `=IFERROR(${diagonalColumn}${excelRow}/${totalColumn}${excelRow},0)`,
      ]];
    }
    const dataStartRow = confusionStart + 2;
    const dataEndRow = dataStartRow + rows.length - 1;
    confusionSheet
      .getRange(
        `B${dataStartRow}:${excelColumn(predictedKeys.length + 1)}${dataEndRow}`,
      )
      .conditionalFormats.add("colorScale", {
        colors: ["#FFFFFF", "#BDD7EE", "#4472C4"],
        thresholds: ["min", "50%", "max"],
      });
    confusionSheet
      .getRange(
        `${excelColumn(confusionWidth)}${dataStartRow}:${excelColumn(confusionWidth)}${dataEndRow}`,
      )
      .format.numberFormat = "0.00%";
    confusionStart += rows.length + 4;
  }
  confusionSheet.freezePanes.freezeRows(1);

  errorSheet.getRange("A1:H1").merge();
  errorSheet.getRange("A1").values = [[
    `${spec.title} · 최장 연속 오분류 구간 상위 100개`,
  ]];
  errorSheet.getRange("A1:H1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
  };
  const errorHeaders = [
    "Activation",
    "시작",
    "종료",
    "지속(초)",
    "시작 frame",
    "종료 frame",
    "정답 단계",
    "예측 단계",
  ];
  errorSheet.getRange("A3:H3").values = [errorHeaders];
  errorSheet.getRange("A3:H3").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    wrapText: true,
  };
  const topErrors = result.errorSegments.slice(0, 100);
  errorSheet.getRangeByIndexes(3, 0, topErrors.length, errorHeaders.length).values =
    topErrors.map((row) => [
      row.threshold,
      row.start_hms,
      row.end_hms,
      row.duration_sec,
      row.start_frame,
      row.end_frame,
      row.actual_stage,
      row.predicted_stage,
    ]);
  errorSheet.getRange(`A4:A${topErrors.length + 3}`).format.numberFormat = "0.00";
  errorSheet.getRange(`D4:D${topErrors.length + 3}`).format.numberFormat = "0.00";
  errorSheet
    .getRange(`H4:H${topErrors.length + 3}`)
    .conditionalFormats.add("containsText", {
      text: "MISS",
      format: { fill: paleRed, font: { color: "#B42318" } },
    });
  errorSheet.freezePanes.freezeRows(3);

  protocolSheet.getRange("A1:D1").merge();
  protocolSheet.getRange("A1").values = [["비교 기준과 재현 정보"]];
  protocolSheet.getRange("A1:D1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
  };
  const protocolRows = [
    ["항목", "값", "설명", "주의"],
    ["단계 정의", spec.subtitle, "정답·예측에 동일 적용", "예측 0은 MISS"],
    ["입력 프레임", frameComparisonPath, "기존 정렬 결과 재사용", "원본 파일 변경 없음"],
    ["공통 프레임", sourceFrameRows.length, "frame_index와 timestamp 정렬 완료", "기존 예측은 activation별 8프레임 누락"],
    ["주 선택 지표", "Balanced Accuracy", "클래스별 Recall의 단순 평균", "다수 클래스 편향 방지"],
    ["보조 지표", "Macro F1", "클래스별 F1의 단순 평균", "클래스별 성능 확인"],
    ["단계별 정확 탐지율", "정확 탐지 프레임 / 해당 정답 전체", "ClassAccuracy 시트와 per_class_accuracy.csv", "분류 지표 Recall과 동일"],
    ["전체 정확도", "Stage Accuracy", "정답 단계와 예측 단계 일치 비율", `Majority 기준선 ${(result.majorityBaseline * 100).toFixed(2)}%`],
    ["이벤트", `동일 단계 전이 ±${eventToleranceSec}초`, "단계화한 시계열에서 재계산", "기존 count 이벤트 점수 재사용 안 함"],
    ["Ground Truth", "1차 수동 검수", "전체 화면에서 보이는 인원", "최종 보고 전 독립 2차 검수 필요"],
    ["ROI 차이", "GT=전체 화면 / 예측=다각형 ROI bottom_center", "오차에 ROI 영향 포함", "activation의 순수 효과가 아님"],
  ];
  for (const key of spec.classKeys) {
    protocolRows.push([
      `정답 분포: ${spec.classLabels[key]}`,
      result.distribution[key],
      `${((result.distribution[key] / sourceFrameRows.length) * 100).toFixed(2)}%`,
      "",
    ]);
  }
  protocolSheet
    .getRangeByIndexes(2, 0, protocolRows.length, 4)
    .values = protocolRows;
  protocolSheet.getRange("A3:D3").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
  };
  protocolSheet.getRange(`A4:A${protocolRows.length + 2}`).format = {
    font: { bold: true, color: navy },
  };
  protocolSheet.getRange(`A3:D${protocolRows.length + 2}`).format.borders = {
    preset: "inside",
    style: "thin",
    color: line,
  };
  protocolSheet.getRange(`A3:D${protocolRows.length + 2}`).format.wrapText = true;

  const widths = {
    Summary: [7, 11, 18, 14, 17, 12, 12, 11, 11, 13, 13],
    ClassAccuracy: [
      12,
      ...spec.classKeys.flatMap(() => [18, 18, 18]),
    ],
    ConfusionMatrix: Array(confusionWidth).fill(17),
    ClassMetrics: [18, 11, 18, 11, 10, 10, 10, 12, 12, 12],
    Calculations: [18, 11, 11, 13, 13, 11, 11, 15, 15, 11, 11, 12, 16, 13, 16, 12, 12, 11, 12, 11, 12, 13, 15],
    TopErrors: [11, 16, 16, 11, 13, 13, 20, 20],
    Protocol: [22, 58, 38, 38],
  };
  for (const sheet of [
    summarySheet,
    classAccuracySheet,
    confusionSheet,
    classSheet,
    calculationSheet,
    errorSheet,
    protocolSheet,
  ]) {
    widths[sheet.name].forEach((width, index) => {
      sheet.getRangeByIndexes(0, index, 1, 1).format.columnWidth = width;
    });
  }
  confusionSheet.getRange(`A1:A${confusionStart}`).format.columnWidth = 22;
  classSheet.getRange(`A4:J${classOutputRow - 1}`).format.rowHeight = 21;
  protocolSheet.getRange(`A3:D${protocolRows.length + 2}`).format.rowHeight = 30;

  const summaryCheck = await workbook.inspect({
    kind: "table",
    range: "Summary!A1:K20",
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 11,
  });
  const classAccuracyCheck = await workbook.inspect({
    kind: "table",
    range: `ClassAccuracy!A1:${classAccuracyEndColumn}${classAccuracyLastRow}`,
    include: "values,formulas",
    tableMaxRows: 12,
    tableMaxCols: classAccuracyWidth,
  });
  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });
  await fs.writeFile(
    path.join(spec.outputDir, "verification_summary.ndjson"),
    summaryCheck.ndjson,
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "verification_class_accuracy.ndjson"),
    classAccuracyCheck.ndjson,
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "verification_errors.ndjson"),
    formulaErrors.ndjson,
    "utf8",
  );

  for (const sheetName of [
    "Summary",
    "ClassAccuracy",
    "ConfusionMatrix",
    "ClassMetrics",
    "Calculations",
    "TopErrors",
    "Protocol",
  ]) {
    const preview = await workbook.render({
      sheetName,
      autoCrop: "all",
      scale: 1,
      format: "png",
    });
    await fs.writeFile(
      path.join(spec.outputDir, `preview_${sheetName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(path.join(spec.outputDir, spec.workbookName));
}

async function writeAnalysisOutputs(result) {
  const { spec, predictedKeys } = result;
  await fs.mkdir(spec.outputDir, { recursive: true });
  const summaryHeaders = [
    "variant",
    "threshold",
    "frames",
    "original_count_mae",
    "original_exact_count_accuracy",
    "stage_accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "stage_mae",
    "stage_rmse",
    "miss_ratio",
    "gt_events",
    "pred_events",
    "matched_events",
    "event_recall",
    "event_precision",
    "event_timing_mae_sec",
  ];
  await fs.writeFile(
    path.join(spec.outputDir, "summary_metrics.csv"),
    toCsv(result.summary, summaryHeaders),
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "class_metrics.csv"),
    toCsv(result.classMetrics, [
      "variant",
      "threshold",
      "class_key",
      "class_label",
      "support",
      "tp",
      "fp",
      "fn",
      "precision",
      "recall",
      "f1",
    ]),
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "per_class_accuracy.csv"),
    toCsv(result.perClassAccuracy, [
      "variant",
      "threshold",
      "actual_class_key",
      "actual_class_label",
      "correctly_detected_frames",
      "total_actual_frames",
      "incorrect_frames",
      "class_accuracy",
    ]),
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "confusion_matrix.csv"),
    toCsv(result.confusionRows, [
      "variant",
      "threshold",
      "actual_key",
      "actual_label",
      ...predictedKeys.map((key) => `pred_${key}`),
      "total",
      "recall",
    ]),
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "event_comparison.csv"),
    toCsv(result.eventRows, [
      "variant",
      "threshold",
      "event_id",
      "gt_frame",
      "gt_sec",
      "gt_hms",
      "from_label",
      "to_label",
      "pred_frame",
      "pred_sec",
      "pred_hms",
      "timing_error_sec",
      "matched",
    ]),
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "error_segments.csv"),
    toCsv(result.errorSegments, [
      "variant",
      "threshold",
      "start_frame",
      "end_frame",
      "start_sec",
      "end_sec",
      "start_hms",
      "end_hms",
      "duration_sec",
      "actual_stage",
      "predicted_stage",
      "error_type",
    ]),
    "utf8",
  );
  const frameHeaders = [
    "frame_index",
    "timestamp_sec",
    "timestamp_hms",
    "ground_truth_count",
    "ground_truth_stage",
    ...variants.flatMap(({ key }) => [
      `${key}_count`,
      `${key}_stage`,
      `${key}_correct`,
    ]),
  ];
  await fs.writeFile(
    path.join(spec.outputDir, "frame_stage_comparison.csv"),
    toCsv(result.frameRows, frameHeaders),
    "utf8",
  );
  await fs.writeFile(
    path.join(spec.outputDir, "analysis_ko.md"),
    analysisLines(result).join("\n"),
    "utf8",
  );
  await buildWorkbook(result);
}

const results = outputSpecs.map(analyze);
for (const result of results) {
  await writeAnalysisOutputs(result);
}

console.log(
  JSON.stringify(
    {
      commonFrames: sourceFrameRows.length,
      outputs: results.map((result) => ({
        analysis: result.spec.id,
        outputDir: result.spec.outputDir,
        recommendedActivation: result.bestBalanced.threshold,
        balancedAccuracy: round(result.bestBalanced.balanced_accuracy),
        macroF1: round(result.bestBalanced.macro_f1),
        stageAccuracy: round(result.bestBalanced.stage_accuracy),
        majorityBaseline: round(result.majorityBaseline),
      })),
    },
    null,
    2,
  ),
);
