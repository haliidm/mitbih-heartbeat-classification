"""Analyze per-record ST-T errors from saved V5 reports; no model inference."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/stt_error_analysis'
CLASSES='NSVFQ'

def load(folder,name):
    with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]

def main():
    OUT.mkdir(exist_ok=True)
    data={};summary=[];all_csv=[]
    for sec in (16,32,64):
        folder=ROOT/'runs'/f'stt_{sec}s_eval_v5_matrix'
        counts={r['记录']:r for r in load(folder,'心搏事件计数.csv') if r['记录']!='Total'}
        matrices={k:{} for k in counts}
        for row in load(folder,'五分类混淆矩阵.csv'):
            if row['记录']!='Total':matrices[row['记录']][row['真实类别']]={c:int(row['预测'+c]) for c in CLASSES}
        rows=[]
        for name,c in counts.items():
            m=matrices[name];matched=sum(sum(r.values()) for r in m.values())
            assert matched==int(c['RTP'])
            ref=int(c['RTP'])+int(c['RFN']);fp=int(c['RFP']);fn=int(c['RFN'])
            pairs={f'{a}→{b}':m[a][b] for a in CLASSES for b in CLASSES if a!=b and m[a][b]}
            wrong=sum(pairs.values())
            row={'窗口秒':sec,'记录':name,'参考心搏':ref,'位置匹配':matched,'多检':fp,'漏检':fn,
                 '定位错误':fp+fn,'分类错误':wrong,'总错误计数':fp+fn+wrong,
                 '每千参考心搏定位错误':round(1000*(fp+fn)/ref,3),
                 '匹配心搏分类错误率':wrong/matched if matched else None,
                 'S支持数_已匹配':sum(m['S'].values()),'STP':m['S']['S'],
                 'SFN_已匹配':sum(m['S'][x] for x in CLASSES if x!='S'),
                 'SFP_已匹配':sum(m[x]['S'] for x in CLASSES if x!='S'),
                 '主要错分':'; '.join(f'{k}:{n}' for k,n in sorted(pairs.items(),key=lambda x:-x[1])[:3])}
            rows.append(row)
        data[str(sec)]=rows;all_csv+=rows
        for key in ('定位错误','分类错误','总错误计数'):
            ranked=sorted(rows,key=lambda r:-r[key]);total=sum(r[key] for r in rows)
            summary.append({'窗口秒':sec,'错误类型':key,'总数':total,
             '前1条占比':ranked[0][key]/total,'前5条占比':sum(r[key] for r in ranked[:5])/total,
             '前10条占比':sum(r[key] for r in ranked[:10])/total,
             '前5条参考心搏占比':sum(r['参考心搏'] for r in ranked[:5])/sum(r['参考心搏'] for r in rows)})
    with (OUT/'逐记录错误统计.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_csv[0]));w.writeheader();w.writerows(all_csv)
    (OUT/'concentration.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# ST-T 全库错误集中度分析','',
      '来源：runs/stt_16s_eval_v5_matrix、stt_32s_eval_v5_matrix、stt_64s_eval_v5_matrix；每份90条记录。仅分析已有报告。','',
      '## 模型与外部评估的界限','',
      '三份训练配置均为MIT-BIH only、随机初始化、external_database_used=false；训练脚本限定48条MIT-BIH记录。因此现有证据支持这些检查点未使用ST-T进行权重训练或微调。V5是推理后处理和评估规则版本。',
      '此前已查看ST-T结果并迭代后处理方案，因此不能把这一整套评估描述为从未参考过ST-T的盲测。它是未在ST-T训练的外部数据库评估；后续根据本分析修改模型或规则会进一步引入开发集使用。','',
      '## 统计定义','',
      '- 定位错误 = RFP + RFN；一个错位预测可能同时产生一个FP和一个FN，因此此计数不是失败心搏的唯一数量。',
      '- 分类错误 = 五分类混淆矩阵非对角元素之和，仅统计位置匹配心搏，每次错分计一次。不能将各类别FP+FN直接相加作为错分次数。',
      '- 总错误计数 = 定位错误 + 分类错误，仅用于排查排序；不是独立心搏错误率。',
      '- 前K条按对应错误数量排序；参考心搏占比作为规模对照。分类错误率分母是位置匹配心搏数。',
      '- 集中度是描述性统计，说明错误分布不均；未进行随机性假设检验，也未从报告推断具体波形原因。','',
      '## 三个模型的集中度','']
    lines+=table(['窗口','类型','错误数','前1条','前5条','前10条','前5条参考心搏占比'],
       [[r['窗口秒'],r['错误类型'],r['总数'],*[f'{r[k]:.2%}' for k in ('前1条占比','前5条占比','前10条占比','前5条参考心搏占比')]] for r in summary])
    for sec in (16,32,64):
        rows=data[str(sec)];lines += ['',f'## {sec}秒模型：总错误计数最多的10条记录','']
        ranked=sorted(rows,key=lambda r:-r['总错误计数'])[:10]
        lines+=table(['记录','参考心搏','多检','漏检','错分类','总错误计数','每千心搏定位错误','匹配后分类错误率','主要错分'],
          [[r['记录'],r['参考心搏'],r['多检'],r['漏检'],r['分类错误'],r['总错误计数'],r['每千参考心搏定位错误'],f"{r['匹配心搏分类错误率']:.2%}",r['主要错分']] for r in ranked])
    rows=data['64'];lines += ['', '## 64秒 S类：匹配后分类漏判最多的10条记录','']
    lines+=table(['记录','匹配的真实S','判对S','错判成其他类','其他类误判成S'],
      [[r['记录'],r['S支持数_已匹配'],r['STP'],r['SFN_已匹配'],r['SFP_已匹配']] for r in sorted(rows,key=lambda r:-r['SFN_已匹配'])[:10]])
    lines += ['', '## 排查优先级','',
      '1. e0501：64秒主要是N→V错分；三个窗口的N→V分别为1635、220、3369次。可对照相同时间段查看模型输出和参考类别。',
      '2. e0306、e0614、e0403：定位FP/FN多，部分记录两者接近；需要检查时序匹配、参考峰校正、区域中心与后处理拆分。仅靠总数无法确定原因。',
      '3. e0112、e0114、e0203：64秒匹配的真实S分别369、175、98个，全部错分；这是S类泛化失败的重要来源。',
      '全库结果应保留这些记录；排除难例后的结果不能替代全库泛化指标。', '']
    (OUT/'STT错误集中度分析.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved',OUT)

if __name__=='__main__':main()
