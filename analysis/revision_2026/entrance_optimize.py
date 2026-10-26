"""Find the entrance point that best explains the phase gradient: the point p on
the comb that maximizes the (count-weighted) correlation between cell distance
from p and cell mean peak-phase. Validates on colony B (should recover ~exit_pos),
then gives a principled entrance estimate for colony A. Streams once, works on the
60x60 cell grid so the p-search is cheap.
"""
from __future__ import annotations
import argparse, numpy as np, pandas as pd
import os; ZEN = os.environ.get("SPEEDTRANSFER_INTERACTIONS", "/mnt/trove/storage/beesbook/speedtransfer/data/zenodo")  # interaction CSVs live on trove; see README
EXIT_POS = {2016: (0, 250), 2019: (5, 264)}
RANGES = {2016: ((-30, 400), (-30, 270)), 2019: ((-30, 440), (-30, 330))}
USE = ["x_pos_start_bee0","y_pos_start_bee0","phase_bee0",
       "x_pos_start_bee1","y_pos_start_bee1","phase_bee1"]
G = 60
def ph_hours(p): return ((-24*p/(2*np.pi))+12) % 24
def wcorr(a,b,w):
    w=w/w.sum(); ma=(a*w).sum(); mb=(b*w).sum()
    ca=a-ma; cb=b-mb
    va=(w*ca*ca).sum(); vb=(w*cb*cb).sum()
    return (w*ca*cb).sum()/np.sqrt(va*vb) if va>0 and vb>0 else np.nan
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int,required=True)
    ap.add_argument("--side",type=int,default=0); ap.add_argument("--chunk",type=int,default=2_000_000)
    a=ap.parse_args(); (x0,x1),(y0,y1)=RANGES[a.year]
    xb=np.linspace(x0,x1,G+1); yb=np.linspace(y0,y1,G+1)
    sph=np.zeros((G,G)); cnt=np.zeros((G,G))
    f=f"{ZEN}/interactions_side{a.side}_{a.year}.csv"; print("streaming",f,flush=True); n=0
    for ch in pd.read_csv(f,usecols=USE,chunksize=a.chunk):
        x=np.concatenate([ch.x_pos_start_bee0,ch.x_pos_start_bee1]).astype(float)
        y=np.concatenate([ch.y_pos_start_bee0,ch.y_pos_start_bee1]).astype(float)
        ph=ph_hours(np.concatenate([ch.phase_bee0,ch.phase_bee1]).astype(float))
        m=np.isfinite(x)&np.isfinite(y)&np.isfinite(ph); x,y,ph=x[m],y[m],ph[m]
        ix=np.clip(np.digitize(x,xb)-1,0,G-1); iy=np.clip(np.digitize(y,yb)-1,0,G-1)
        np.add.at(sph,(ix,iy),ph); np.add.at(cnt,(ix,iy),1); n+=len(ch)
    xc=0.5*(xb[:-1]+xb[1:]); yc=0.5*(yb[:-1]+yb[1:])
    good=cnt>max(50,cnt.sum()/(G*G)*0.05)
    XX,YY=np.meshgrid(xc,yc,indexing="ij")
    gx=XX[good]; gy=YY[good]; gph=(sph/np.maximum(cnt,1))[good]; gw=cnt[good]
    print(f"rows {n:,}  good cells {good.sum()}  phase range [{gph.min():.2f},{gph.max():.2f}]",flush=True)
    # grid-search candidate entrance p over a coarse mesh spanning the comb + margin
    px=np.linspace(x0,x1,25); py=np.linspace(y0,y1,25)
    best=(-1,None)
    for pxi in px:
        for pyi in py:
            d=np.hypot(gx-pxi,gy-pyi)
            r=wcorr(d,gph,gw)                 # want phase to INCREASE with distance => r>0
            if np.isfinite(r) and r>best[0]: best=(r,(pxi,pyi))
    r,p=best
    ex,ey=EXIT_POS[a.year]
    d0=np.hypot(gx-ex,gy-ey); r0=wcorr(d0,gph,gw)
    print(f"HARD-CODED exit_pos={EXIT_POS[a.year]}  phase-vs-dist weighted r = {r0:.3f}",flush=True)
    print(f"BEST-FIT entrance   =({p[0]:.0f},{p[1]:.0f})  phase-vs-dist weighted r = {r:.3f}",flush=True)
if __name__=="__main__": main()
