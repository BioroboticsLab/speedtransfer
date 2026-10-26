"""Full-data stitch check for 2016 side0: stream the whole file, accumulate a fine
x-histogram (a bad two-camera stitch shows as a gap, a doubling ridge, or a sharp
step at the seam ~x=180) and a 2-D density. 2019 as single-camera control."""
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; ZEN = os.environ.get("SPEEDTRANSFER_INTERACTIONS", "/mnt/trove/storage/beesbook/speedtransfer/data/zenodo")  # interaction CSVs live on trove; see README
USE = ["x_pos_start_bee0","y_pos_start_bee0","x_pos_start_bee1","y_pos_start_bee1"]
RNG = {2016: ((-30,400),(-30,270)), 2019: ((-30,440),(-30,330))}
def run(year):
    (x0,x1),(y0,y1)=RNG[year]; nx=int(x1-x0); ny=int(y1-y0)
    hx=np.zeros(nx); hy=np.zeros(ny); H=np.zeros((nx//4, ny//4)); n=0
    for ch in pd.read_csv(f"{ZEN}/interactions_side0_{year}.csv", usecols=USE, chunksize=2_000_000):
        x=np.concatenate([ch.x_pos_start_bee0,ch.x_pos_start_bee1]).astype(float)
        y=np.concatenate([ch.y_pos_start_bee0,ch.y_pos_start_bee1]).astype(float)
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]; n+=len(x)
        ix=np.clip((x-x0).astype(int),0,nx-1); iy=np.clip((y-y0).astype(int),0,ny-1)
        np.add.at(hx,ix,1); np.add.at(hy,iy,1)
        np.add.at(H,(np.clip(ix//4,0,nx//4-1),np.clip(iy//4,0,ny//4-1)),1)
    print(f"{year}: n={n:,}", flush=True)
    return np.arange(x0,x1),hx,np.arange(y0,y1),hy,H,(x0,x1,y0,y1)
fig,ax=plt.subplots(2,2,figsize=(14,8))
for r,year in enumerate((2016,2019)):
    xg,hx,yg,hy,H,ext=run(year)
    ax[r,0].bar(xg,hx,width=1.0,color="steelblue")
    ax[r,0].axvline(180,color="crimson",ls=":",label="~seam x=180" if year==2016 else None)
    ax[r,0].set_title(f"{year} side0 FULL x-hist (1-unit bins)"); ax[r,0].legend()
    im=ax[r,1].imshow(np.log1p(H).T,origin="lower",extent=ext,aspect="auto",cmap="cividis")
    ax[r,1].axvline(180,color="crimson",ls=":") if year==2016 else None
    ax[r,1].set_title(f"{year} side0 FULL 2-D density (log)")
    # seam metric: median-normalized dip at x=180 vs neighborhood
    if year==2016:
        c=xg; band=(c>120)&(c<240); loc=(c>=176)&(c<=184)
        base=np.median(hx[band]); dip=np.median(hx[loc])
        print(f"   x=180 seam: local median {dip:.0f} vs band median {base:.0f} = {100*(dip-base)/base:+.0f}%", flush=True)
fig.tight_layout(); fig.savefig(os.path.join(os.environ.get("REVISION_OUT","."),"full_seam.pdf"),bbox_inches="tight")
print("saved full_seam.pdf", flush=True)
