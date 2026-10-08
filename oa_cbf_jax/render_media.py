"""Render prepared, source-bound controller comparisons as images or video."""


import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .dataset import sha256
from .io import write_json


def read(path):
    return json.loads(Path(path).read_text())


LABELS = dict(gat='OA (GAT)', nearest_fc='FC (one nearest obstacle)',
    fixed_low='Fixed low', fixed_high='Fixed high', optimal_decay='Optimal decay',
    optimal_decay_qp='Extra OD-QP', barriernet='BarrierNet')


COLORS=dict(gat='#137a63',nearest_fc='#cf6b25',fixed_low='#526e94',fixed_high='#8a63a3',
            optimal_decay='#b29636',optimal_decay_qp='#9b514e',barriernet='#5e7379')


TITLES=dict(unicycle='Dynamic unicycle · static obstacles',quad2d='Nonlinear Quad2D · static obstacles',
            quad2d_layout='Nonlinear Quad2D · additional static layouts',
            bicycle='Kinematic bicycle DPCBF · moving obstacles',
            quad3d='Linearized Quad3D · static cylinders')


def bound(path,digest):
    if sha256(path)!=digest:raise ValueError('Changed illustration evidence: '+str(path))


def cylinder_lines(obstacle, lower_z, upper_z):
    """The plant checks xy disks at EVERY altitude, i.e. infinite cylinders."""
    angles=np.linspace(0,2*np.pi,33)
    xy=obstacle[:2]+obstacle[2]*np.column_stack((np.cos(angles),np.sin(angles)))
    lines=[np.column_stack((xy,np.full(len(xy),z))) for z in (lower_z,upper_z)]
    for k in (0,8,16,24):lines.append(np.array([[*xy[k],lower_z],[*xy[k],upper_z]]))
    return lines


def render(output,example,video=False,storage_floor_gib=140.):
    if not math.isfinite(storage_floor_gib) or storage_floor_gib<1:
        raise ValueError('Positive finite storage reserve required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle
    root=Path(output);proof=read(root/'provenance.json')
    for p,d in proof['bindings'].items():bound(p,d)
    path=root/(example+'_inputs.json');bound(path,proof['examples'][example]);meta=read(path)
    path=root/(example+'_physical.npz');bound(path,meta['physical_sha256'])
    with np.load(path,allow_pickle=False) as z:data=dict(z)
    spatial=meta['dynamics']=='quad3d';moving=meta['dynamics']=='bicycle'
    methods=[m for m in LABELS if m in meta['methods']];dt=meta['robot']['dt'];radius=meta['robot']['radius']
    end=max(meta['methods'][m]['steps'] for m in methods);mask=data['mask'];goal=data['goal']
    fig=plt.figure(figsize=(19.2,10.8));axes=[]
    kind='OA-success illustration' if example=='hero' else 'Retained FC-success/OA-failure example' if example=='counterexample' else 'Representative success (no OA-only case)'
    fig.suptitle(f"{TITLES[meta['cohort']]} | {meta['obstacle_count']} obstacles\n{kind}",fontsize=19,y=.98)
    fig.text(.5,.905,meta['parent'],ha='center',fontsize=10)
    rates=[100*meta['benchmark'][m]['success_rate'] for m in ('gat','nearest_fc')]
    fig.text(.5,.037,f'Selected after the complete benchmark: OA {rates[0]:.2f}% vs nearest FC {rates[1]:.2f}% overall. This scene does not establish aggregate superiority.\n'
        'All panels use the same physical scene and simulation time. A stopped record freezes, including its obstacles.',ha='center',fontsize=11)
    fig.subplots_adjust(left=.045,right=.98,top=.85,bottom=.12,hspace=.4,wspace=.3)
    dimension=3 if spatial else 2
    points=np.vstack([data[m+'_states'][:,:dimension] for m in methods]+[np.asarray(goal)[None,:dimension]])
    obs=data[methods[0]+'_obstacles'][mask];positions=[obs[:,:2]]
    if moving:positions.append(obs[:,:2]+end*dt*obs[:,3:5])
    padding=max(radius,float(obs[:,2].max(initial=0)))+.3
    lo=points.min(0)-padding;hi=points.max(0)+padding
    for p in positions:
        lo[:2]=np.minimum(lo[:2],p.min(0)-padding);hi[:2]=np.maximum(hi[:2],p.max(0)+padding)
    artists=[]
    for i,m in enumerate(methods):
        ax=fig.add_subplot(2,4,i+1,projection='3d' if spatial else None);axes.append(ax)
        row=meta['methods'][m];states=data[m+'_states'];color=COLORS[m]
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)' if meta['dynamics']!='quad2d' else 'z (m)')
        ax.set_title(f"{LABELS[m]}\n{row['outcome']} · {row['steps']*dt:.2f}s recorded",fontsize=11)
        if spatial:
            from matplotlib.ticker import FormatStrFormatter
            ax.set_zlim(lo[2],hi[2]);ax.set_zlabel('z (m)');ax.set_box_aspect(hi-lo);ax.view_init(24,-56)
            # Dense fields are much wider than their altitude range. Avoid
            # overlapping altitude labels without changing the physical scale.
            ax.set_zticks(np.linspace(lo[2],hi[2],3))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.tick_params(axis='z',labelsize=7,pad=0)
            for obstacle in obs:
                for xyz in cylinder_lines(obstacle,lo[2],hi[2]):ax.plot(*xyz.T,color='#777777',alpha=.35,lw=.65)
            ax.scatter(*goal[:3],marker='*',color='#267b31',s=85)
            line,=ax.plot([],[],[],color=color,lw=2);body,=ax.plot([],[],[],'o',color=color,ms=6)
            label=ax.text2D(.01,.01,'',transform=ax.transAxes,fontsize=8);circles=[]
        else:
            ax.set_aspect('equal');ax.grid(alpha=.15);ax.scatter(*goal[:2],marker='*',color='#267b31',s=85)
            circles=[]
            for obstacle in data[m+'_obstacles'][mask]:
                patch=Circle(obstacle[:2],obstacle[2],fc='#999999',ec='#777777',alpha=.65,lw=.5);ax.add_patch(patch);circles.append(patch)
            line,=ax.plot([],[],color=color,lw=1.7);body=Circle(states[0,:2],radius,fc=color,alpha=.9,zorder=5);ax.add_patch(body)
            label=ax.text(.01,.015,'',transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',alpha=.8))
        artists.append((m,line,body,label,circles))
    gain_ax=fig.add_subplot(2,4,8);gain_lines=[];max_gain=.1
    for m in ('gat','nearest_fc'):
        gains=data[m+'_gains'];max_gain=max(max_gain,float(gains.max(initial=0)))
        for j in range(gains.shape[1]):
            line,=gain_ax.plot([],[],color=COLORS[m],ls=('-','--',':','-.')[j%4],lw=1,
                label=('OA' if m=='gat' else 'FC')+f' α{j+1}')
            gain_lines.append((m,j,line))
    gain_ax.set_xlim(0,max(.05,end*dt));gain_ax.set_ylim(0,max_gain*1.08);gain_ax.set_title('Recorded class-K coefficients',fontsize=11)
    gain_ax.set_xlabel('Simulation time (s)');gain_ax.grid(alpha=.15);gain_ax.legend(fontsize=8,ncol=2)
    def frame(t):
        for m,line,body,label,circles in artists:
            states=data[m+'_states'];k=min(t,len(states)-1);line.set_data(states[:k+1,0],states[:k+1,1])
            if spatial:
                line.set_3d_properties(states[:k+1,2]);body.set_data([states[k,0]],[states[k,1]]);body.set_3d_properties([states[k,2]])
            else:
                body.center=states[k,:2]
                if moving:
                    obstacles=data[m+'_obstacles'][mask]
                    for patch,o in zip(circles,obstacles):patch.center=o[:2]+k*dt*o[3:5]
            label.set_text(f'{k*dt:.2f}s'+(' | record ended' if k==len(states)-1 else ''))
        for m,j,line in gain_lines:
            n=min(t,len(data[m+'_gains']));line.set_data(np.arange(n)*dt,data[m+'_gains'][:n,j])
    frame(end);png=root/(example+'.png');fig.savefig(png,dpi=100)
    result=dict(png=str(png.resolve()),png_sha256=sha256(png))
    if video:
        if shutil.disk_usage(root).free/2**30<storage_floor_gib:
            raise ValueError(f'{storage_floor_gib:g} GiB disk reserve reached')
        path=root/(example+'.mp4');ticks=list(range(0,end+1,2))
        if ticks[-1]!=end:ticks.append(end)
        writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','2'])
        with writer.saving(fig,str(path),100):
            for t in ticks:frame(t);writer.grab_frame()
        probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(path)],capture_output=True,text=True,check=True).stdout)
        stream=next(s for s in probe['streams'] if s['codec_type']=='video')
        if int(stream['nb_frames'])!=len(ticks) or stream['width']!=1920 or stream['height']!=1080:raise ValueError('Incomplete video')
        result.update(video=str(path.resolve()),video_sha256=sha256(path),frames=len(ticks),fps=10)
    plt.close(fig);write_json(root/(example+'_render.json'),result);return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Prepared media directory containing provenance and physical traces')
    parser.add_argument('--example', default='hero', choices=['hero', 'counterexample', 'representative'])
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--storage-floor-gib', type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(render(args.output, args.example, args.video, args.storage_floor_gib), indent=2))
