"""Full-data spatial maps (phase-time, age, density) to locate the true entrance
per colony/side, streaming the whole CSV (no early-row bias). The entrance is
where peak phase is earliest, foragers/old bees and traffic concentrate."""
from __future__ import annotations
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os; ZEN = os.environ.get("SPEEDTRANSFER_INTERACTIONS", "/mnt/trove/storage/beesbook/speedtransfer/data/zenodo")  # interaction CSVs live on trove; see README
EXIT_POS = {2016: (0, 250), 2019: (5, 264)}
# generous fixed ranges (clip outliers)
RANGES = {2016: ((-30, 400), (-30, 270)), 2019: ((-30, 440), (-30, 330))}
USE = ["x_pos_start_bee0","y_pos_start_bee0","phase_bee0","age_bee0",
       "x_pos_start_bee1","y_pos_start_bee1","phase_bee1","age_bee1"]
G = 60

def ph_hours(p): return ((-24*p/(2*np.pi))+12) % 24

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--side", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    a = ap.parse_args()
    (x0,x1),(y0,y1) = RANGES[a.year]
    xb = np.linspace(x0,x1,G+1); yb = np.linspace(y0,y1,G+1)
    sph = np.zeros((G,G)); sag = np.zeros((G,G)); cnt = np.zeros((G,G))
    f = f"{ZEN}/interactions_side{a.side}_{a.year}.csv"
    print("streaming", f, flush=True)
    n=0
    for ch in pd.read_csv(f, usecols=USE, chunksize=a.chunk):
        x = np.concatenate([ch.x_pos_start_bee0, ch.x_pos_start_bee1]).astype(float)
        y = np.concatenate([ch.y_pos_start_bee0, ch.y_pos_start_bee1]).astype(float)
        ph = ph_hours(np.concatenate([ch.phase_bee0, ch.phase_bee1]).astype(float))
        ag = np.concatenate([ch.age_bee0, ch.age_bee1]).astype(float)
        m = np.isfinite(x)&np.isfinite(y)&np.isfinite(ph)&np.isfinite(ag)
        x,y,ph,ag = x[m],y[m],ph[m],ag[m]
        ix = np.clip(np.digitize(x,xb)-1,0,G-1); iy = np.clip(np.digitize(y,yb)-1,0,G-1)
        np.add.at(sph,(ix,iy),ph); np.add.at(sag,(ix,iy),ag); np.add.at(cnt,(ix,iy),1)
        n += len(ch)
    print(f"rows {n:,}  points {int(cnt.sum()):,}", flush=True)
    good = cnt > max(50, cnt.sum()/ (G*G) * 0.05)
    mph = np.where(good, sph/np.maximum(cnt,1), np.nan)
    mag = np.where(good, sag/np.maximum(cnt,1), np.nan)
    dens = np.where(cnt>0, cnt, np.nan)
    xc = 0.5*(xb[:-1]+xb[1:]); yc = 0.5*(yb[:-1]+yb[1:])
    ex,ey = EXIT_POS[a.year]
    def cell(px,py): return (np.argmin(np.abs(xc-px)), np.argmin(np.abs(yc-py)))
    ei,ej = cell(ex,ey)
    # entrance candidates
    pi,pj = np.unravel_index(np.nanargmin(mph), mph.shape)      # earliest phase
    di,dj = np.unravel_index(np.nanargmax(np.where(good,dens,np.nan)), dens.shape)  # densest
    print(f"exit_pos={EXIT_POS[a.year]}  -> cell ({xc[ei]:.0f},{yc[ej]:.0f}) phase={mph[ei,ej]:.2f} age={mag[ei,ej]:.1f} n={cnt[ei,ej]:.0f}")
    print(f"EARLIEST-PHASE cell ({xc[pi]:.0f},{yc[pj]:.0f}) phase={mph[pi,pj]:.2f}  <- entrance by phase")
    print(f"DENSEST      cell ({xc[di]:.0f},{yc[dj]:.0f}) n={dens[di,dj]:.0f} phase={mph[di,dj]:.2f}")
    print(f"phase spatial range: [{np.nanmin(mph):.2f}, {np.nanmax(mph):.2f}] h")
    ext = [xb[0],xb[-1],yb[0],yb[-1]]
    fig, ax = plt.subplots(1,3, figsize=(15,4.4))
    for k,(dat,ttl,cm) in enumerate([(mph,"mean peak phase [h]","viridis_r"),(mag,"mean age [d]","magma"),(np.log10(dens),"log10 count","cividis")]):
        im = ax[k].imshow(dat.T, origin="lower", extent=ext, aspect="auto", cmap=cm)
        ax[k].plot([ex],[ey],"r*",ms=15,mec="white")
        ax[k].set_title(f"{a.year} side{a.side}: {ttl}"); fig.colorbar(im,ax=ax[k])
    fig.tight_layout()
    out=os.path.join(os.environ.get("REVISION_OUT","."), f"spatialmap_{a.year}_side{a.side}.pdf")
    fig.savefig(out, bbox_inches="tight"); print("saved", out, flush=True)

if __name__=="__main__": main()
