"""Index saved ECG experiments and reports without loading models or datasets."""
import csv
import json
import re
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}

def link(path):
    return '../' + path.relative_to(ROOT).as_posix()

def main():
    OUT.mkdir(exist_ok=True)
    runs=ROOT/'runs'
    lines=['# ECG 训练与评估结果索引','',
      '汇总来自本地已有 JSON、CSV 和 Markdown 文件，不重新训练或推理。', '',
      '## 当前 16 / 32 / 64 秒模型','',
      '| 输入 | 目录 | 最佳轮 | 已训练轮数 | batch size | 内部测试逐点宏 F1 |',
      '|---|---|---:|---:|---:|---:|']
    training=[]
    for sec,name in [(16,'mitbih_rhythm_n'),(32,'mitbih_32s'),(64,'mitbih_64s')]:
        folder=runs/name
        cfg=read_json(folder/'training_config.json')
        m=read_json(folder/'pointwise_classification_metrics.json')
        hist=read_json(folder/'training_history.json')
        score=m.get('pointwise_classification',{}).get('macro_f1_N_S_V_F_Q')
        score_text=f'{score:.2%}' if score is not None else '未保存'
        lines.append(f'| {sec}秒 | [{name}]({link(folder/"training_config.json")}) | {m.get("best_epoch","未保存")} | {len(hist)} | {cfg.get("batch_size","未保存")} | {score_text} |')
        manifest=read_json(folder/'split_manifest.json')
        training.append({'run':name,'seconds':sec,'best_epoch':m.get('best_epoch'),
           'epochs_completed':len(hist),'batch_size':cfg.get('batch_size'),
           'pointwise_macro_f1':score,'split_counts':{k:len(manifest.get(k,[])) for k in ('train','validation','test')}})
    lines += ['', '三种长度按窗口随机拆分，同记录和重叠片段可跨集合；MIT-BIH全库含训练记录。ST-T为这三个模型的外部评估数据库。',
       '16秒训练batch size为16，32/64秒为8。长窗口的测试评价点集合也不同，结果不能仅归因于窗口长度。',
       '新版分类指标仅基于位置匹配心搏；漏检和多检由R峰指标体现。以下六份报告采用同一V5后处理及±100ms容差。', '']
    for database,prefix in [('MIT-BIH','mitbih'),('ST-T','stt')]:
        lines += [f'## {database} 全库 R 峰检测', '',
           '| 输入 | TP | FP | FN | 阳性预测值 | 敏感度 | F1 | 报告 |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
        totals={}
        for sec in (16,32,64):
            folder=runs/f'{prefix}_{sec}s_eval_v5_matrix'
            f=folder/'分类指标.csv'
            if not f.exists(): continue
            with f.open(encoding='utf-8-sig',newline='') as stream:
                rows={r['类别']:r for r in csv.DictReader(stream) if r['记录']=='Total'}
            totals[sec]=rows
            r=rows['R峰']; tp,fp,fn=[int(r[k]) for k in ('TP','FP','FN')]
            lines.append(f'| {sec}秒 | {tp} | {fp} | {fn} | {tp/(tp+fp):.2%} | {tp/(tp+fn):.2%} | {2*tp/(2*tp+fp+fn):.2%} | [查看]({link(folder/"事件评估报告.md")}) |')
        lines += ['',f'## {database} 五分类 F1（位置匹配心搏）','',
            '| 输入 | N | S | V | F | Q | 混淆矩阵 |','|---|---:|---:|---:|---:|---:|---|']
        for sec,rows in totals.items():
            folder=runs/f'{prefix}_{sec}s_eval_v5_matrix'
            vals=[rows[c]['F1分数'] for c in ('N','S','V','F','Q')]
            lines.append('| '+str(sec)+'秒 | '+' | '.join(vals)+f' | [查看]({link(folder/"五分类混淆矩阵.csv")}) |')
        lines.append('')
    lines += ['## 结果阅读说明','',
        'ST-T：S类F1随16/32/64秒上升，但64秒V类F1低于32秒；延长窗口并非对所有类别都有改善。低样本类别还应同时查看支持数与混淆矩阵。',
        '本表用于结果对照，不基于外部测试分数调整训练参数或阈值。', '',
        '## 当前评估报告','',
        '| 报告 | 指标口径 | 来源数据库 | 模型路径（报告原文） |','|---|---|---|---|']
    report_index=[]
    for report in sorted(runs.rglob('事件评估报告.md')):
        body=report.read_text(encoding='utf-8-sig')
        db=re.search(r'数据库目录：([^\r\n]+)',body)
        model=re.search(r'模型文件：([^\r\n]+)',body)
        matrix=report.parent/'五分类混淆矩阵.csv'
        protocol='位置匹配后五分类矩阵' if matrix.exists() else '历史报告：按原报告解释'
        item={'report':report.relative_to(ROOT).as_posix(),'database':db.group(1).strip() if db else '报告未记录',
              'checkpoint':model.group(1).strip() if model else '报告未记录','classification_protocol':protocol}
        report_index.append(item)
        lines.append(f'| [{report.parent.relative_to(runs).as_posix()}]({link(report)}) | {protocol} | {item["database"]} | {item["checkpoint"]} |')
    lines += ['', '## runs 目录总览','', '| 目录 | 权重数 | 评估报告数 | 文件数 | 大小 MiB |', '|---|---:|---:|---:|---:|']
    inventory={}
    for d in sorted(runs.iterdir()):
        if not d.is_dir(): continue
        files=[p for p in d.rglob('*') if p.is_file()]
        info={'files':len(files),'bytes':sum(p.stat().st_size for p in files),'weights':sum(p.suffix=='.pt' for p in files),'reports':sum(p.name=='事件评估报告.md' for p in files)}
        inventory[d.name]=info
        lines.append(f'| {d.name} | {info["weights"]} | {info["reports"]} | {info["files"]} | {info["bytes"]/1048576:.2f} |')
    (OUT/'实验结果汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'experiment_index.json').write_text(json.dumps({'training':training,'evaluation_reports':report_index,'run_inventory':inventory},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Indexed {len(training)} current models, {len(report_index)} evaluation reports, {len(inventory)} run directories.')

if __name__=='__main__':
    main()
