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
plt.rcParams.update({'font.sans-serif':['DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'path','font.size':12})
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
        ax.set(xticks=range(5),yticks=range(5),xticklabels=classes,yticklabels=classes,xlabel='Predicted class',ylabel='Reference class')
        ax.set_title(f'MIT-BIH | {seconds}-s model\n'+('Row-normalized confusion matrix' if mode=='percent' else 'Confusion matrix (counts)'),fontsize=17,pad=16)
        fig.colorbar(im,ax=ax,shrink=.8,label='Percentage (%)' if mode=='percent' else 'Number of beats')
        fig.supxlabel('Full-database, position-matched beats; includes training records',fontsize=10,color='#64748b')
        for ext in ('png','svg'):fig.savefig(OUT/f'confusion_{seconds}s_{mode}_en.{ext}',dpi=180,facecolor='white')
        plt.close(fig)

# Publication-style architecture: muted colors, explicit tensor shapes, detail panels.
fig,ax=plt.subplots(figsize=(17,8))
fig.subplots_adjust(left=.015,right=.985,top=.97,bottom=.03)
ax.set(xlim=(0,17),ylim=(0,8));ax.axis('off')
ink='#253746';blue='#dce7ef';green='#e2ece6';purple='#e8e3ee';sand='#f1e9db'
def box(x,y,w,h,title,detail='',color=blue,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.06',facecolor=color,edgecolor='#7a8993',linewidth=.9))
    ax.text(x+w/2,y+h*.70,title,ha='center',va='center',fontsize=fs,fontweight='bold',color=ink)
    if detail:ax.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=fs-1,color=ink,linespacing=1.4)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#596774',linewidth=1.15))
ax.text(.2,7.65,'(a) Multi-scale CNN–Transformer for dense heartbeat classification',fontsize=16,fontweight='bold',color=ink)
box(.2,5.1,1.6,1.3,'ECG input','B × 1 × L',blue)
for y,k in [(6.5,3),(5.1,5),(3.7,7)]:
    box(2.35,y,2.4,1.05,f'CNN branch: k = {k}','5 stride-2 Conv1d blocks',green)
    arrow((1.85,5.75),(2.3,y+.52));arrow((4.8,y+.52),(5.35,5.75))
box(5.4,5.1,1.8,1.3,'Concatenate','B × 384 × T',green)
arrow((7.25,5.75),(7.65,5.75))
box(7.7,5.1,1.65,1.3,'1×1 fusion','384 → 192',blue)
arrow((9.4,5.75),(9.85,5.75))
box(9.9,5.1,2.3,1.3,'Transformer × 3','B × T × 192',purple)
box(9.9,6.75,2.3,.7,'Learned position',color=purple,fs=10)
arrow((11.05,6.7),(11.05,6.45))
arrow((12.25,5.75),(12.65,5.75))
box(12.7,5.1,1.9,1.3,'Upsampling × 5','Linear ×2 + Conv',sand)
arrow((14.65,5.75),(15.05,5.75))
box(15.1,5.1,1.6,1.3,'1×1 classifier','B × 6 × L',blue)
ax.text(13.7,4.65,'Pointwise logits: background / N / S / V / F / Q',ha='center',fontsize=10,color=ink)
ax.text(5.9,3.9,'L = 5,760 / 11,520 / 23,040     T = L / 32 = 180 / 360 / 720',fontsize=11,color=ink)
# Detail panels
for x,w,title in [(.2,5.1,'(b) CNN branch'),(5.7,5.3,'(c) Transformer encoder block'),(11.4,5.3,'(d) Dense decoder')]:
    ax.add_patch(FancyBboxPatch((x,.3),w,3,boxstyle='round,pad=0.03,rounding_size=0.08',facecolor='#fafbfc',edgecolor='#c2cbd2',linewidth=.8))
    ax.text(x+.15,3.0,title,fontsize=12,fontweight='bold',color=ink)
box(.5,1.75,4.5,.85,'Conv1d → GroupNorm(8) → GELU','kernel k; stride 2; padding floor(k/2)',green,10)
ax.text(2.75,1.3,'Channels: 1 → 8 → 16 → 32 → 64 → 128',ha='center',fontsize=10,color=ink)
ax.text(2.75,.8,'The same kernel size is used in all five layers.',ha='center',fontsize=9,color=ink)
box(6,1.85,4.7,.8,'LayerNorm → Self-attention → Add','6 heads; residual connection',purple,10)
box(6,.7,4.7,.8,'LayerNorm → Feed-forward → Add','192 → 384 → 192; GELU; dropout 0.1',purple,10)
arrow((8.35,1.8),(8.35,1.55))
ax.text(8.35,.42,'Non-causal attention over the entire input window',ha='center',fontsize=8.5,color=ink)
box(11.7,1.75,4.7,.85,'Linear upsampling ×2 → Conv1d(k=3)','GroupNorm(8) → GELU; repeated five times',sand,10)
ax.text(14.05,1.3,'Channels: 192 → 128 → 96 → 64 → 32 → 32',ha='center',fontsize=9.5,color=ink)
ax.text(14.05,.8,'Final 1×1 convolution: 32 → 6 class logits',ha='center',fontsize=10,color=ink)
for ext in ('png','svg','pdf'):fig.savefig(OUT/f'model_architecture_en.{ext}',dpi=220,facecolor='white')
plt.close(fig)
print('English architecture and confusion-matrix figures generated.')
