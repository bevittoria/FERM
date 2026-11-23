from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ferm_mechanism_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'mathtext.fontset':'dejavusans','axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
ink='#243b4b'; grey='#a9adb0'; pale='#d9dcde'; orange='#c67b42'; teal='#237f7b'
fig=plt.figure(figsize=(9.0,3.8))
for x,letter,title in [(.035,'A','Spatial search'),(.367,'B','Features shift benefits'),(.728,'C','Sequential acceptance')]:
 fig.text(x,.94,letter,weight='bold',fontsize=10)
 fig.text(x+.028,.94,title,fontsize=8.5)
# a: schematic geometry. Node sizes schematically indicate population differences.
a=fig.add_axes([.035,.27,.285,.60]); a.set_aspect('equal'); a.set_xlim(-2.65,2.65); a.set_ylim(-2.55,2.55); a.axis('off')
coords=[(-.68,.51),(1.45,.676),(1.32,-1.88)]
# Very light nested disks and fine rings emphasize distance without visual clutter.
for radius,alpha in [(2.3,.018),(1.6,.025),(.85,.035)]:
 a.add_patch(Circle((0,0),radius,facecolor=ink,edgecolor='none',alpha=alpha,zorder=0))
for radius in [np.hypot(*xy) for xy in coords]:
 a.add_patch(Circle((0,0),radius,fill=False,lw=.75,ec='#c6cdd1',zorder=1))
for (x,y),n,col,label,offset in zip(coords,[4,7,6],[orange,teal,grey],['$j_1$','$j_2$','$j_3$'],[(-23,12),(15,8),(15,-3)]):
 a.plot([0,x],[0,y],color=col,lw=1.05,alpha=.65,zorder=2)
 a.scatter(x,y,s=n*(32.5 if n == 4 else 65),c=col,edgecolors='white',linewidths=1.3,zorder=4)
 a.annotate(label,(x,y),xytext=offset,textcoords='offset points',color=col,fontsize=9)
# Open halo distinguishes the origin; filled marker areas still encode counts.
a.scatter(0,0,s=500,facecolors='none',edgecolors=ink,linewidths=.7,zorder=4)
a.scatter(0,0,s=5*65,c=ink,edgecolors='white',linewidths=1.2,zorder=5)
a.annotate('$i$',(0,0),xytext=(-17,-22),textcoords='offset points',color=ink,fontsize=9)
fig.text(.1775,.225,'$d_{ij_1}<d_{ij_2}<d_{ij_3}$',ha='center',fontsize=8)
# Minimal vector feature glyphs: local attributes and pairwise ties.
f=fig.add_axes([.381,.755,.285,.13]); f.set_xlim(0,1); f.set_ylim(0,1); f.axis('off')
# A small bar glyph represents local attributes, without a decorative cityscape.
for x0,h in [( .13,.18),(.18,.31),(.23,.42)]:
 f.plot([x0,x0],[.48,.48+h],color=ink,lw=2.5,solid_capstyle='butt')
f.text(.18,.29,'Local attributes',ha='center',fontsize=7)
# A three-node motif represents origin--destination relations.
for u,v in [((.71,.59),(.8,.85)),((.8,.85),(.9,.59)),((.71,.59),(.9,.59))]:
 f.plot([u[0],v[0]],[u[1],v[1]],color=ink,lw=.75)
f.scatter([.71,.8,.9],[.59,.85,.59],s=15,color=ink,zorder=3)
f.text(.8,.29,'Relational attributes',ha='center',fontsize=7)
f.plot([.18,.18,.8,.8],[.15,.04,.04,.15],color=grey,lw=.7,clip_on=False)
f.plot([.49,.49],[.04,-.025],color=grey,lw=.7,clip_on=False)
f.text(.49,-.16,'Benefit shifts',ha='center',fontsize=7,color=ink)

# b: elementary densities, actual finite illustrative samples, marked maxima.
rows=[('Origin $i$',0,[-1.15,-.45,.05,.55,1.3],ink,'$T_i$'),('$j_1$',-.55,[-1.7,-.8,-.25,.3],orange,'$B_{ij_1}$'),('$j_2$',.65,[-.7,-.2,.3,.75,1.1,1.55,2.25],teal,'$B_{ij_2}$')]
x=np.linspace(-3.3,3.5,600)
for k,(label,mu,draws,col,maxlabel) in enumerate(rows):
 ax=fig.add_axes([.399,.575-k*.145,.245,.115])
 y=np.exp(-.5*(x-mu)**2)/np.sqrt(2*np.pi)
 ax.fill_between(x,0,y,color=col,alpha=.07,lw=0)
 ax.plot(x,y,color=col,lw=1.05)
 ax.plot([mu,mu],[0,.399],color=col,lw=.55,ls=(0,(2,2)),alpha=.65)
 ax.scatter(draws[:-1],[-.045]*(len(draws)-1),s=8,c=col,zorder=4)
 ax.scatter(draws[-1],-.045,s=19,c=col,marker='D',zorder=5)
 ax.annotate(maxlabel,(draws[-1],-.045),xytext=(5,3),textcoords='offset points',fontsize=7,color=col,bbox={'facecolor':'white','edgecolor':'none','pad':.2})
 ax.text(-3.3,.43,label,fontsize=7.4,color=col)
 ax.text(mu,.475,['$\\Sigma_{ii}$','$\\Sigma_{ij_1}$','$\\Sigma_{ij_2}$'][k],ha='center',fontsize=7,color=col)
 ax.set(xlim=(-3.3,3.6),ylim=(-.12,.55)); ax.set_yticks([]); ax.set_xticks([])
 for sp in ax.spines.values(): sp.set_visible(False)
 ax.axhline(0,color=pale,lw=.5,zorder=0)
 if k==2:
  ax.annotate('',xy=(3.55,-.115),xytext=(-3.3,-.115),arrowprops={'arrowstyle':'->','lw':.6,'color':ink})
  ax.set_xlabel('Elementary benefit',fontsize=7.4,labelpad=3)
seq=fig.add_axes([.765,.825,.215,.055]);seq.set_xlim(.6,3.4);seq.set_ylim(0,1);seq.axis('off')
for pos,col in [(1,orange),(2,teal),(3,grey)]:
 seq.scatter(pos,.5,s=22,color=col)
for lo,hi in [(1.15,1.85),(2.15,2.85)]:
 seq.annotate('',xy=(hi,.5),xytext=(lo,.5),arrowprops={'arrowstyle':'->','lw':.65,'color':grey})

# c: the same sampled threshold, first exceedance; no fabricated j3 draw.
c=fig.add_axes([.762,.325,.218,.48]); c.set_xlim(.55,3.45); c.set_ylim(0,2.95)
c.spines[['top','right']].set_visible(False);c.spines['bottom'].set_color(grey);c.spines['left'].set_color(grey)
c.set_xticks([1,2,3],['$j_1$','$j_2$','$j_3$']);c.set_yticks([]); c.tick_params(axis='x',length=0,pad=6)
c.hlines(1.3,.58,3.45,color=ink,ls=(0,(4,3)),lw=.8)
c.text(3.4,1.4,'Fixed $T_i$',ha='right',fontsize=8)
for xx,val,col in [(1,.3,orange),(2,2.25,teal)]:
 c.bar(xx,val,width=.32,color=col,alpha=.15,linewidth=0)
 c.vlines(xx,0,val,color=col,lw=1.3)
 c.scatter(xx,val,s=24,color=col,marker='D',zorder=3)
c.text(2,2.52,'Accept',ha='center',color=teal,fontsize=7.5)
c.text(1,.47,'Reject',ha='center',color=orange,fontsize=7.5)
c.text(3,.39,'Not\nvisited',ha='center',va='center',color='#888888',fontsize=7)
c.set_ylabel('Maximum benefit',fontsize=7.4,labelpad=5)
fig.text(.762,.225,'Increasing distance $\\rightarrow$',fontsize=7.4)
# Quiet common-limit statement, without a banner.
for ext in ['svg','pdf','png']:
 fig.savefig(OUT/f'figure1_ferm_mechanism_v3.{ext}',dpi=350, bbox_inches=Bbox.from_extents(0, .65, 9.0, 3.8))
plt.close(fig)
(OUT/'caption.tex').write_text(r'''\textbf{Feature-enriched, distance-ordered opportunity acceptance.}
\textbf{A,} Destinations are examined in increasing distance from origin $i$. Node sizes schematically indicate population differences and thus differences in opportunity counts; they are not drawn to a common quantitative scale.
\textbf{B,} Elementary benefits follow Gaussian distributions with a common variance and feature-dependent means. Local attributes and origin--destination relationships enter through the shifts $\Sigma_{ii}$ and $\Sigma_{ij}$. Curves show elementary benefit densities, circles show illustrative draws, and diamonds mark their maxima. The origin threshold is $T_i=\max_{\ell\leq m_i}X_{i\ell}$ and the destination offer is $B_{ij}=\max_{\ell\leq n_j}Y_{ij\ell}$. The illustrated opportunity counts are $m_i=5$, $n_{j_1}=4$, and $n_{j_2}=7$.
\textbf{C,} The same sampled threshold is retained throughout the search. Destination $j_1$ is rejected; $j_2$ is the first to exceed $T_i$ and is accepted, so $j_3$ is not visited. If no destination exceeds the threshold, no destination is accepted in the modelled domain. A common continuous elementary benefit distribution recovers the radiation model. Locations, shifts and draws are schematic, not empirical estimates.
''')
print(OUT)
