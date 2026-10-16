"""Evaluate an Icentia patient range with all three checkpoints and resumable records."""
import argparse
import contextlib
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .label_config import FS, WINDOW_SAMPLES, LABEL_RADIUS_MS
from .ecg_data import load_record_for_inference
from .record_inference_metrics import predict_record
from .cnn_transformer import MultiScaleCNNTransformer
from .beat_postprocess import decode_v5
from . import evaluate_records as evaluation

ROOT = Path(__file__).resolve().parents[1]
MODEL_CHECKPOINTS = {s: ROOT / 'checkpoints' / f'ecg_cnn_transformer_{s}s.pt' for s in (16, 32, 64)}
BATCHES = {16: 16, 32: 2, 64: 1}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



SCORING_POLICY = 'icentia_q_position_only_v2'
SCORED_CLASSES = ('N', 'S', 'V')


def count_icentia_events(reference, predicted, tolerance_samples):
    """Score all reference positions, excluding reference Q only from class scoring."""
    matches, missed, extras = evaluation.match_by_position(reference, predicted, tolerance_samples)
    counts = evaluation._empty_counts()
    counts['R峰'].update(TP=len(matches), FN=len(missed), FP=len(extras))
    matrix = np.zeros((5, 5), dtype=np.int64)
    q_matches = 0
    for (_, ref_class), (_, pred_class) in matches:
        if ref_class == 5:
            q_matches += 1
            continue
        if 1 <= ref_class <= 5 and 1 <= pred_class <= 5:
            matrix[ref_class - 1, pred_class - 1] += 1
        for class_id, name in enumerate(('N', 'S', 'V', 'F'), start=1):
            if ref_class == class_id and pred_class == class_id:
                counts[name]['TP'] += 1
            elif ref_class == class_id:
                counts[name]['FN'] += 1
            elif pred_class == class_id:
                counts[name]['FP'] += 1
    for _, class_id in missed:
        if 1 <= class_id <= 4:
            counts[('N', 'S', 'V', 'F')[class_id - 1]]['FN'] += 1
    for _, class_id in extras:
        if 1 <= class_id <= 4:
            counts[('N', 'S', 'V', 'F')[class_id - 1]]['FP'] += 1
    result = counts, matrix, len(reference), len(predicted), len(matches)
    return result, {'reference_q': sum(c == 5 for _, c in reference),
                    'matched_reference_q_excluded_from_classification': q_matches}


def classification_metrics(matrix):
    # Keep prediction F/Q columns: predicting these for a known beat remains wrong.
    return {c: v for c, v in evaluation.classification_metrics(matrix).items()
            if c in SCORED_CLASSES}


def write_reports(folder, count_rows, metric_rows, matrices, tolerance):
    fields = ('记录',) + tuple(f'{c}{k}' for c in ('R', 'N', 'S', 'V') for k in ('TP', 'FP', 'FN'))
    with (folder / '心搏事件计数.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(count_rows)
    headers = ('记录', '真实类别', '预测N', '预测S', '预测V', '预测F', '预测Q')
    with (folder / '分类混淆矩阵.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for name, matrix in matrices.items():
            for i, c in enumerate(SCORED_CLASSES):
                w.writerow([name, c, *map(int, matrix[i])])
    evaluation._write_metrics_csv(folder / '分类指标.csv', metric_rows)
    lines = ['# Icentia 完整记录评估（参考 Q 仅参与位置评分）', '',
             f'位置匹配容差：±{tolerance:g} ms。',
             '全部参考心搏（包括 Q）一起按位置一对一匹配，计算 R 峰 TP/FP/FN。参考 Q 的匹配对不进入分类矩阵，不计任何类别的 TP/FP/FN/TN；漏检的参考 Q 仅计 RFN。未匹配预测仍计 RFP，不因邻近 Q 而忽略。',
             '本库参与类型评分的参考类别为 N/S/V，报告仅计算这三类指标。混淆矩阵保留预测 F/Q 列；确定类别被预测为 F/Q 仍计错分。',
             '分类指标仅基于位置匹配心搏；未匹配漏检、多检另见 R 峰指标和事件计数。R 峰没有定义 TN，不计算特异性。', '',
             '## 心搏事件计数', '',
             *evaluation._markdown_table(fields, [[r[k] for k in fields] for r in count_rows]), '',
             '## 分类混淆矩阵（行：参考；列：预测）', '',
             *evaluation._markdown_table(('参考/预测', 'N', 'S', 'V', 'F', 'Q'),
                 [[c, *map(int, matrices['Total'][i])] for i, c in enumerate(SCORED_CLASSES)]), '',
             '## 检测与分类指标', '']
    metric_fields = ('记录', '类别', 'TP', 'FP', 'FN', 'TN', '阳性预测值', '敏感度', '特异性', 'F1分数')
    rates = metric_fields[6:]
    lines += evaluation._markdown_table(metric_fields,
        [[evaluation._format_rate(r.get(k)) if k in rates else r.get(k, '') for k in metric_fields]
         for r in metric_rows], left_columns=2)
    (folder / '事件评估报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def aggregate(folder, names, tolerance, expected):
    counts_rows, metrics_rows, matrices, infos = [], [], {}, []
    total = evaluation._empty_counts()
    matrix = np.zeros((5, 5), dtype=np.int64)
    for name in names:
        p = folder / 'records' / (name + '.json')
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding='utf-8'))
        if r.get('scoring_policy') != SCORING_POLICY:
            raise ValueError('结果评分口径不同，请使用新的输出目录重新评估。')
        counts = r['counts']
        cm = np.array(r['confusion'], dtype=np.int64)
        counts_rows.append(evaluation._count_row(name, counts))
        matrices[name] = cm
        matrix += cm
        infos.append(r['info'])
        for c in total:
            for k in total[c]:
                total[c][k] += counts[c][k]
        metrics_rows.append({'记录': name, '类别': 'R峰', **evaluation.detection_metrics(counts)})
        metrics_rows.extend({'记录': name, '类别': c, **v} for c, v in classification_metrics(cm).items())
    completed = len(infos)
    counts_rows.append(evaluation._count_row('Total', total))
    matrices['Total'] = matrix
    metrics_rows.append({'记录': 'Total', '类别': 'R峰', **evaluation.detection_metrics(total)})
    classes = classification_metrics(matrix)
    metrics_rows.extend({'记录': 'Total', '类别': c, **v} for c, v in classes.items())
    write_reports(folder, counts_rows, metrics_rows, matrices, tolerance)
    evaluation._write_record_info_csv(folder / '记录信息.csv', infos)
    report = folder / '事件评估报告.md'
    body = report.read_text(encoding='utf-8').replace('详见各记录long_regions.json', '详见records目录各记录JSON的cleanup.long_regions')
    report.write_text(f'# Icentia11k患者子集：已完成 {completed}/{expected} 个片段\n\n'
        + ('状态：完整。\n\n' if completed == expected else '状态：部分结果，不能当作完整子集评估。\n\n')
        + '缺失ATR的片段列于上级excluded_records.json，不参与任何模型计分。\n\n'
        + '每个原始约70分钟片段独立预处理并完整滑窗推理，不拼接片段边界。源250Hz重采样至360Hz；参考R位置沿用原流程在±50ms内校正。参考 Q 仅参与位置评分、不参与类型评分，非心搏节律标注忽略；本库没有 F 参考类别。仅报告 N/S/V 分类指标。\n\n'
        + body, encoding='utf-8')
    summary = {'completed_segments': completed, 'expected_segments': expected,
               'detection': evaluation.detection_metrics(total), 'classification': classes, 'scoring_policy': SCORING_POLICY}
    save_json(folder / 'summary.json', summary)
    return summary


def write_comparison(output, summaries):
    lines = ['# Icentia11k 三模型对比', '', '仅比较相同患者范围；各行完成片段数不同时不能直接比较。', '',
             '| 窗口 | 完成片段 | R峰精确率 | R峰召回率 | R峰F1 | N F1 | S F1 | V F1 |',
             '|---|---|---|---|---|---|---|---|']
    fmt = evaluation._format_rate
    for seconds, s in summaries.items():
        det = s['detection']
        vals = [f'{seconds}秒', f"{s['completed_segments']}/{s['expected_segments']}",
                *[fmt(det[k]) for k in ('阳性预测值', '敏感度', 'F1分数')],
                *[fmt(s['classification'][c]['F1分数']) for c in SCORED_CLASSES]]
        lines.append('| ' + ' | '.join(vals) + ' |')
    (output / '三个模型对比.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-dir', type=Path, default=ROOT / 'icentia11k-continuous-ecg-1.0')
    parser.add_argument('--patient-start', type=int, default=0)
    parser.add_argument('--patient-end', type=int, default=99)
    parser.add_argument('--tolerance-ms', type=float, default=150)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/icentia_p00000_p00099_v5_150ms_Qposition')
    args = parser.parse_args()
    if not 0 <= args.patient_start <= args.patient_end <= 10999:
        raise ValueError('Invalid inclusive patient range')
    if not np.isfinite(args.tolerance_ms) or args.tolerance_ms < 0:
        raise ValueError('Invalid tolerance')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, fingerprints, skipped = [], [], []
    for patient in range(args.patient_start, args.patient_end + 1):
        patient_dir = args.database_dir / f'p{patient // 1000:02d}' / f'p{patient:05d}'
        headers = sorted(patient_dir.glob(f'p{patient:05d}_s*.hea'))
        if not headers:
            raise FileNotFoundError(f'Patient folder empty or missing: {patient_dir}')
        for header in headers:
            if not header.with_suffix('.atr').is_file():
                skipped.append({'record':header.stem,'reason':'Missing ATR annotation; not scored as zero-reference signal'})
                continue
            for suffix in ('.hea', '.dat', '.atr'):
                path = header.with_suffix(suffix)
                stat = path.stat()
                if stat.st_size == 0:
                    raise ValueError(f'Empty input: {path}')
                fingerprints.append([path.relative_to(args.database_dir).as_posix(), stat.st_size, stat.st_mtime_ns])
            records.append((header.stem, patient_dir))
    code_files = ['evaluate_icentia.py','evaluate_records.py','ecg_data.py','label_config.py','cnn_transformer.py','record_inference_metrics.py','label_cleanup.py','beat_postprocess.py']
    signature = {'database': str(args.database_dir.resolve()), 'patients': [args.patient_start,args.patient_end],
                 'tolerance_ms': args.tolerance_ms, 'scoring_policy': SCORING_POLICY, 'skipped': skipped, 'records': [r[0] for r in records],
                 'input_metadata_sha256': hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest(),
                 'checkpoints': {str(s): sha256(path) for s, path in MODEL_CHECKPOINTS.items()},
                 'code_sha256': {f:sha256(Path(__file__).parent / f) for f in code_files}, 'batches': BATCHES}
    # Normalize integer JSON object keys before comparing a resumed run.
    signature = json.loads(json.dumps(signature))
    manifest = output / 'evaluation_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != signature:
        raise ValueError('Inputs/configuration changed. Use a new --output-dir to avoid mixing results.')
    save_json(manifest, signature)
    save_json(output / 'excluded_records.json', skipped)
    for item in skipped:
        print(f"Excluded: {item['record']} ({item['reason']})", flush=True)
    lock = output / 'RUNNING.lock'
    try:
        lock_handle = lock.open('x')
    except FileExistsError:
        raise RuntimeError(f'Run lock exists: {lock}. If a previous process was terminated, confirm it is stopped before removing this lock.')
    try:
        import os
        lock_handle.write(str(os.getpid()));lock_handle.close()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}; patients: {args.patient_end-args.patient_start+1}; segments: {len(records)}; tolerance: {args.tolerance_ms} ms', flush=True)
        models, folders = {}, {}
        for seconds, checkpoint_path in MODEL_CHECKPOINTS.items():
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            samples = int(ckpt.get('input_samples', WINDOW_SAMPLES))
            if samples != seconds * FS or int(ckpt.get('label_radius_ms',LABEL_RADIUS_MS)) != LABEL_RADIUS_MS:
                raise ValueError('Checkpoint window/label configuration mismatch')
            model = MultiScaleCNNTransformer(input_samples=samples).to(device)
            model.load_state_dict(ckpt['model_state']);model.eval();models[seconds] = model
            folders[seconds] = output / f'{seconds}s'
            (folders[seconds] / 'records').mkdir(parents=True, exist_ok=True)
        started = time.time()
        for index, (name, directory) in enumerate(records, 1):
            pending = [s for s in models if not (folders[s] / 'records' / f'{name}.json').exists()]
            if pending:
                print(f'[{index}/{len(records)}] {name}', flush=True)
                record, source_fs = load_record_for_inference(directory, name)
                for seconds in pending:
                    tick = time.time()
                    logits = predict_record(models[seconds], record, device, batch_windows=BATCHES[seconds])
                    labels = np.argmax(logits,axis=0).astype(np.uint8)
                    del logits
                    corrected, regions, beats, cleanup = decode_v5(labels,record.signal,sample_rate=FS)
                    scored, ignored_q = count_icentia_events(record.events, beats, round(args.tolerance_ms*FS/1000))
                    counts, confusion, refs, preds, matched = scored
                    result = {'record':name,'patient':directory.name,'model_seconds':seconds,
                              'counts':counts,'confusion':confusion.tolist(),'cleanup':cleanup,
                              'scoring_policy':SCORING_POLICY, 'ignored_q':ignored_q,
                              'reference_events':[[int(t),int(c)] for t,c in record.events],
                              'predicted_events':[[int(t),int(c)] for t,c in beats],
                              'info':{'记录':name,'源采样率（Hz）':source_fs,'模型采样率（Hz）':FS,
                                      '时长（秒）':round(len(record.signal)/FS,3),'参考心搏数':refs}}
                    save_json(folders[seconds] / 'records' / f'{name}.json',result)
                    print(f'  {seconds}s: TP={matched}, FP={preds-matched}, FN={refs-matched}; elapsed={time.time()-tick:.1f}s',flush=True)
                    del labels,corrected,regions,beats,cleanup,result
                del record
            if index == 1 or index % 10 == 0 or index == len(records):
                summaries = {s:aggregate(folder,signature['records'],args.tolerance_ms,len(records)) for s,folder in folders.items()}
                write_comparison(output,summaries)
                save_json(output / 'progress.json',{'last_record':name,'scan_index':index,'expected':len(records),'elapsed_seconds':time.time()-started,'models':{str(s):v['completed_segments'] for s,v in summaries.items()}})
        print(f'Completed. Reports: {output}',flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
