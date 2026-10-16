"""Generate README scientific figures from saved confusion matrices; no inference."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/images'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'path','font.size':12})
classes=list('NSVFQ')
for seconds in (16,32,64):
    src=ROOT/f'reports/published/mitbih_{seconds}s_eval_v5_matrix/五分类混淆矩阵.csv'
    with src.open(encoding='utf-8-sig') as f:
        rows={r['真实类别']:r for r in csv.DictReader(f) if r['记录']=='Total'}
    cm=np.array([[int(rows[c]['预测'+p]) for p in classes] for c in classes])
    pct=cm/cm.sum(axis=1,keepdims=True)*100
    with (OUT/f'confusion_{seconds}s_percent.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['真实类别']+['预测'+c for c in classes])
        for c,row in zip(classes,pct):w.writerow([c]+[f'{v:.2f}%' for v in row])
    for mode,values in [('counts',cm),('percent',pct)]:
        fig,ax=plt.subplots(figsize=(8.4,7.5),layout='constrained')
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=100 if mode=='percent' else cm.max())
        for i in range(5):
            for j in range(5):
                label=f'{values[i,j]:.2f}%' if mode=='percent' else f'{values[i,j]:,}'
                ax.text(j,i,label,ha='center',va='center',color='white' if values[i,j]>im.norm.vmax*.55 else '#17324d',fontsize=14,fontweight='bold' if i==j else 'normal')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=classes,yticklabels=classes,xlabel='预测心搏类别',ylabel='参考心搏类别')
        ax.set_title(f'MIT-BIH · {seconds} 秒模型\n'+('百分比混淆矩阵（按真实类别归一化）' if mode=='percent' else '数量混淆矩阵'),fontsize=17,pad=16)
        fig.colorbar(im,ax=ax,shrink=.8,label='比例（%）' if mode=='percent' else '心搏数量')
        fig.supxlabel('全库位置匹配心搏；含训练记录',fontsize=10,color='#64748b')
        for ext in ('png','svg'):fig.savefig(OUT/f'confusion_{seconds}s_{mode}.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
fig,ax=plt.subplots(figsize=(16,9));fig.subplots_adjust(left=.02,right=.98,top=.90,bottom=.04)
ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
fig.suptitle('多尺度卷积 + 双向上下文注意力：心搏逐点分类网络',fontsize=22,fontweight='bold',color='#17324d')
def box(x,y,w,h,title,detail,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',facecolor=color,edgecolor='#bacbdc',linewidth=1.2))
    ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontsize=14,fontweight='bold',color='#17324d')
    ax.text(x+w/2,y+h*.32,detail,ha='center',va='center',fontsize=11,color='#334155',linespacing=1.6)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color='#64748b',linewidth=1.6))
box(.3,3.65,2.1,1.55,'单导联 ECG','B × 1 × L\n16 / 32 / 64 秒','#e8f1ff')
for y,k in [(6.5,3),(3.65,5),(.8,7)]:
    box(3.2,y,3.35,1.65,f'五层卷积 · 卷积核 {k}','步长 2 · 通道 8→16→32→64→128\n每层：组归一化 + GELU','#e5f5f0')
    arrow((2.45,4.4),(3.1,y+.82));arrow((6.65,y+.82),(7.4,4.4))
box(7.5,3.65,2.2,1.65,'拼接与融合','384 → 192 通道\n1×1 卷积','#e8f1ff')
arrow((9.8,4.4),(10.45,4.4))
box(10.55,3.5,4.85,2,'Transformer × 3','可学习位置编码 · 窗口内双向注意力\n特征维度 192 · 6 头 · 前馈维度 384','#f0eaff')
arrow((13,3.4),(13,2.45))
box(10.55,.7,4.85,1.65,'五级上采样 + 逐点分类','每级线性上采样 ×2 + 卷积\n192→128→96→64→32→32 → 六类分数','#fff2df')
arrow((10.45,1.5),(9.8,1.5))
box(7.5,.7,2.2,1.65,'逐点输出','B × 6 × L\n背景 / N / S / V / F / Q','#e8f1ff')
ax.text(8,8.55,'L = 5760 / 11520 / 23040；下采样后序列长度 = 180 / 360 / 720',ha='center',color='#475569',fontsize=13)
for ext in ('png','svg'):fig.savefig(OUT/f'model_architecture.{ext}',dpi=180,facecolor='white')
plt.close(fig)
print('Created architecture and three pairs of count/percentage confusion matrices.')
