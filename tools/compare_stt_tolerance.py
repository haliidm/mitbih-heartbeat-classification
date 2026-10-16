"""Compare saved 32 s ST-T evaluation reports at 100 and 150 ms."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/stt_error_analysis'

def read(folder,name):
    with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def load(folder):
    rows={}
    for r in read(folder,'心搏事件计数.csv'):
        if r['记录']=='Total':continue
        tp,fp,fn=[int(r[k]) for k in ('RTP','RFP','RFN')]
        rows[r['记录']]={'tp':tp,'fp':fp,'fn':fn,'wrong':0,'ref':tp+fn,'pred':tp+fp}
    for r in read(folder,'五分类混淆矩阵.csv'):
        if r['记录']=='Total':continue
        rows[r['记录']]['wrong']+=sum(int(r['预测'+c]) for c in 'NSVFQ' if c!=r['真实类别'])
    for r in rows.values():r['total']=r['fp']+r['fn']+r['wrong']
    return rows

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]

def main():
    old=load(ROOT/'runs/stt_32s_eval_v5_matrix')
    new=load(ROOT/'runs/stt_32s_eval_v5_matrix_150ms')
    assert set(old)==set(new) and len(new)==90
    changed=[n for n in old if old[n]['pred']!=new[n]['pred'] or old[n]['ref']!=new[n]['ref']]
    if changed:raise ValueError('Reference/prediction totals changed; investigate before attributing to tolerance: '+str(changed))
    ro=sorted(old,key=lambda n:(-old[n]['total'],n));rn=sorted(new,key=lambda n:(-new[n]['total'],n))
    rows=[]
    for n in ro:
        a,b=old[n],new[n]
        rows.append({'记录':n,'原排名':ro.index(n)+1,'新排名':rn.index(n)+1,
          '100ms多检':a['fp'],'150ms多检':b['fp'],'100ms漏检':a['fn'],'150ms漏检':b['fn'],
          '100ms错分类':a['wrong'],'150ms错分类':b['wrong'],
          '100ms总错误':a['total'],'150ms总错误':b['total'],'减少':a['total']-b['total'],
          '减少比例':(a['total']-b['total'])/a['total'] if a['total'] else None})
    OUT.mkdir(exist_ok=True)
    with (OUT/'32秒_100与150ms逐记录对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 32秒模型：ST-T匹配容差100与150ms对比','',
       '使用相同检查点、数据库和V5后处理。150ms评估推理批量为2，与原报告保持一致；全部90条记录的参考和预测心搏总数均与旧报告一致。',
       '总错误计数=多检+漏检+位置匹配后的错分类。一个位置偏差超限可能同时计FP和FN。容差放宽并未改变模型预测的含义；分类评价纳入的已匹配心搏会增加。','',
       '## 全库统计','']
    totals=[]
    for label,data in [('100ms',old),('150ms',new)]:
        sums={k:sum(r[k] for r in data.values()) for k in ('tp','fp','fn','wrong','total')}
        f1=2*sums['tp']/(2*sums['tp']+sums['fp']+sums['fn'])
        totals.append([label,*[sums[k] for k in ('fp','fn','wrong','total')],f'{f1:.4%}'])
    lines+=table(['容差','多检','漏检','错分类','总错误计数','R峰F1'],totals)
    headers=['记录','原排名','新排名','原总错误','新总错误','减少比例','新多检','新漏检','新错分类']
    by_name={r['记录']:r for r in rows}
    for title,names in [('原来错误最多的10条',ro[:10]),('150ms下错误最多的10条',rn[:10])]:
        lines+=['',f'## {title}','']
        lines+=table(headers,[[n,by_name[n]['原排名'],by_name[n]['新排名'],old[n]['total'],new[n]['total'],
              f"{by_name[n]['减少比例']:.2%}" if by_name[n]['减少比例'] is not None else '—',new[n]['fp'],new[n]['fn'],new[n]['wrong']] for n in names])
    lines+=['','此分析用于诊断位置偏差。不能将新150ms报告与其他模型100ms报告直接比较；若据此选择容差，ST-T已参与评估设置选择。','']
    (OUT/'32秒_100与150ms对比.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()
