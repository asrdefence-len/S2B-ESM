"""Selected-emitter spectrum plus conventional time-down waterfall."""

import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class EmitterFrequencyWaterfallWindow(QMainWindow):
    TIME_SPAN_S = 20.0
    TIME_BINS = 160
    FREQ_BINS = 180
    SPECTRUM_LOOKBACK_S = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("S2B ESM - Selected Emitter Frequency Waterfall")
        self.resize(900, 650)
        body=QWidget(self); layout=QVBoxLayout(body)
        self.heading=QLabel("NO EMITTER SELECTED"); layout.addWidget(self.heading)
        self.figure=Figure(figsize=(8.5,5.8))
        grid=self.figure.add_gridspec(
            2,2,height_ratios=(1,4),width_ratios=(30,1),
            left=.10,right=.94,bottom=.09,top=.93,hspace=.08,wspace=.12)
        self.spectrum_axes=self.figure.add_subplot(grid[0,0])
        self.waterfall_axes=self.figure.add_subplot(grid[1,0],sharex=self.spectrum_axes)
        self.colorbar_axes=self.figure.add_subplot(grid[1,1])
        self.canvas=FigureCanvas(self.figure); layout.addWidget(self.canvas,stretch=1)
        self.setCentralWidget(body); self._colorbar=None; self.clear_plot()

    def clear_plot(self,message="NO PDW HISTORY"):
        self.spectrum_axes.clear(); self.waterfall_axes.clear(); self.colorbar_axes.clear()
        self.spectrum_axes.set_title("SELECTED EMITTER SPECTRUM")
        self.spectrum_axes.set_ylabel("Strength (dBFS)")
        self.spectrum_axes.tick_params(labelbottom=False)
        self.waterfall_axes.set_xlabel("Frequency (MHz)")
        self.waterfall_axes.set_ylabel("Time ago (s)")
        self.waterfall_axes.text(.5,.5,message,transform=self.waterfall_axes.transAxes,
                                 ha="center",va="center")
        self.colorbar_axes.set_axis_off(); self.canvas.draw_idle()

    def update_pdws(self,emitter_id,pdw_rows):
        rows=list(pdw_rows or []); self.heading.setText(f"{emitter_id}  FREQUENCY WATERFALL")
        if not rows:self.clear_plot(f"{emitter_id}: WAITING FOR PDWs"); return
        data=np.asarray(rows,dtype=float)
        t=data[:,0]; f=data[:,1]/1e6; amp=data[:,2]
        newest=float(np.max(t)); oldest=max(float(np.min(t)),newest-self.TIME_SPAN_S)
        keep=t>=oldest; t=t[keep]; f=f[keep]; amp=amp[keep]

        fmin=float(np.min(f)); fmax=float(np.max(f))
        if fmax-fmin<2.0:
            mid=.5*(fmin+fmax); fmin,fmax=mid-1.0,mid+1.0
        else:
            pad=max(.5,.08*(fmax-fmin)); fmin-=pad; fmax+=pad

        f_edges=np.linspace(fmin,fmax,self.FREQ_BINS+1)
        age=newest-t
        age_edges=np.linspace(0.0,self.TIME_SPAN_S,self.TIME_BINS+1)
        image=np.full((self.TIME_BINS,self.FREQ_BINS),np.nan)
        fi=np.clip(np.searchsorted(f_edges,f,side="right")-1,0,self.FREQ_BINS-1)
        ai=np.clip(np.searchsorted(age_edges,age,side="right")-1,0,self.TIME_BINS-1)
        for y,x,a in zip(ai,fi,amp):
            if np.isnan(image[y,x]) or a>image[y,x]:image[y,x]=a

        vmin=float(np.percentile(amp,10)); vmax=float(np.max(amp))
        if vmax<=vmin:vmax=vmin+1.0

        recent=age<=self.SPECTRUM_LOOKBACK_S
        spectrum=np.full(self.FREQ_BINS,np.nan)
        if np.any(recent):
            for x,a in zip(fi[recent],amp[recent]):
                if np.isnan(spectrum[x]) or a>spectrum[x]:spectrum[x]=a
        centres=.5*(f_edges[:-1]+f_edges[1:])

        self.spectrum_axes.clear()
        self.spectrum_axes.plot(centres,spectrum)
        self.spectrum_axes.set_title(f"{emitter_id}  SPECTRUM (last 1 s)")
        self.spectrum_axes.set_ylabel("Strength (dBFS)")
        self.spectrum_axes.grid(True,alpha=.25)
        self.spectrum_axes.tick_params(labelbottom=False)
        self.spectrum_axes.set_xlim(fmin,fmax)

        self.waterfall_axes.clear()
        self.waterfall_axes.set_xlabel("Frequency (MHz)")
        self.waterfall_axes.set_ylabel("Time ago (s)")
        im=self.waterfall_axes.imshow(
            image,origin="upper",aspect="auto",interpolation="nearest",
            extent=[fmin,fmax,self.TIME_SPAN_S,0.0],vmin=vmin,vmax=vmax)
        self.waterfall_axes.set_ylim(self.TIME_SPAN_S,0.0)

        self.colorbar_axes.clear(); self.colorbar_axes.set_axis_on()
        self._colorbar=self.figure.colorbar(im,cax=self.colorbar_axes)
        self._colorbar.set_label("Measured strength (dBFS)")
        self.canvas.draw_idle()
