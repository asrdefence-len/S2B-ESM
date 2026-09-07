"""Block-streaming four-face IQ simulator for the unified S2B ESM front end.

The simulated receiver has four independent 90-degree faces centred at
0/90/180/270 degrees. A source is rendered into the face containing its bearing.
Downstream processing receives four IQ channels; it is never given emitter IDs.
"""
import math
from pathlib import Path
import numpy as np
from beam_model import RotatingSincBeam
from rf_link_budget import received_power_dbm
from scenario_runtime import ScenarioRuntime
from scripted_antenna_motion import ScriptedAntennaMotion
from scenarios import get_scenario

FACE_CENTRES_DEG={1:0.0,2:90.0,3:180.0,4:270.0}

def face_for_bearing(bearing_deg):
    b=float(bearing_deg)%360.0
    if b>=315.0 or b<45.0:return 1
    if b<135.0:return 2
    if b<225.0:return 3
    return 4

class SimulatedStreamingIQSource:
    def __init__(self,sample_rate_hz=40_000_000,center_frequency_hz=9_400_000_000,block_samples=40_000,noise_std=0.02,legacy_scenario_name="close_emitters"):
        self.sample_rate_hz=float(sample_rate_hz);self.center_frequency_hz=float(center_frequency_hz);self.block_samples=int(block_samples);self.noise_std=float(noise_std);self.block_duration_s=self.block_samples/self.sample_rate_hz;self.sample_index=0;self.rng=np.random.default_rng(12345);self.legacy_scenario=get_scenario(legacy_scenario_name)
        # Test geometry only: these bearings determine which physical receiver face
        # gets the signal. They are not passed downstream as emitter truth.
        self.legacy_bearings_deg=(45.0,270.0)
        root=Path(__file__).resolve().parent;self.scripted_runtime=ScenarioRuntime(root/"emitter_types.yaml",root/"scripted_scenarios"/"nav_scan_to_dwell.yaml");self.e3_motion=ScriptedAntennaMotion(self.scripted_runtime,"E3")
    @property
    def time_s(self):return self.sample_index/self.sample_rate_hz
    def reset(self):self.sample_index=0;self.rng=np.random.default_rng(12345)
    def _add_pulse(self,iq,block_start_s,pulse_toa_s,pulse_width_s,rf_frequency_hz,amplitude,modulation="CW",bandwidth_hz=0.0):
        start=int(round((pulse_toa_s-block_start_s)*self.sample_rate_hz));pulse_samples=max(1,int(round(pulse_width_s*self.sample_rate_hz)));stop=start+pulse_samples;ds=max(0,start);de=min(len(iq),stop)
        if ds>=de:return
        ss=ds-start;count=de-ds;pulse_t=np.arange(ss,ss+count,dtype=float)/self.sample_rate_hz;if_hz=float(rf_frequency_hz)-self.center_frequency_hz
        if str(modulation).upper() in ("FM","LFM") and bandwidth_hz>0.0:
            k=float(bandwidth_hz)/float(pulse_width_s);f0=if_hz-.5*float(bandwidth_hz);phase=2.*np.pi*(f0*pulse_t+.5*k*pulse_t**2)
        else:phase=2.*np.pi*if_hz*pulse_t
        iq[ds:de]+=(float(amplitude)*np.exp(1j*phase)).astype(np.complex64)
    @staticmethod
    def _periodic_toas(start_s,pri_s,block_start_s,block_end_s):
        if block_end_s<=start_s:return []
        n0=max(0,int(math.ceil((block_start_s-start_s)/pri_s-1e-12)));t=start_s+n0*pri_s;out=[]
        while t<block_end_s-1e-15:
            if t>=block_start_s-1e-15:out.append(t)
            n0+=1;t=start_s+n0*pri_s
        return out
    def _render_e1_e2(self,faces,block_start_s,block_end_s):
        for idx,emitter in enumerate(self.legacy_scenario.emitters):
            rf_hz=self.center_frequency_hz+float(emitter["if_frequency_hz"]);bearing=self.legacy_bearings_deg[idx%len(self.legacy_bearings_deg)];iq=faces[face_for_bearing(bearing)]
            for toa_s in self._periodic_toas(float(emitter["start_delay_s"]),float(emitter["pri_s"]),block_start_s,block_end_s):
                self._add_pulse(iq,block_start_s,toa_s,float(emitter["pulse_width_s"]),rf_hz,float(emitter["amplitude"]),emitter.get("modulation","CW"),float(emitter.get("lfm_bandwidth_hz",0.0)))
    def _e3_mode_segments(self):
        emitter=next(e for e in self.scripted_runtime.emitters if str(e["id"])=="E3");timeline=emitter["timeline"];segments=[]
        for i,event in enumerate(timeline):segments.append((float(event["time_s"]),float(timeline[i+1]["time_s"]) if i+1<len(timeline) else self.scripted_runtime.duration_s,str(event["mode"])))
        return segments
    def _render_e3(self,faces,block_start_s,block_end_s):
        rx_gain_dbi=float(self.scripted_runtime.esm_receiver["antenna_gain_dbi"]);threshold_dbm=float(self.scripted_runtime.esm_receiver["detection_threshold_dbm"])
        for segment_start,segment_end,_ in self._e3_mode_segments():
            start=max(block_start_s,segment_start);end=min(block_end_s,segment_end)
            if start>=end:continue
            state0=self.scripted_runtime.state("E3",start);pri_s=float(state0.mode["pri_us"])*1e-6
            for toa_s in self._periodic_toas(segment_start,pri_s,start,end):
                state=self.scripted_runtime.state("E3",toa_s);motion=self.e3_motion.state(toa_s);antenna=state.mode["antenna"];beam=RotatingSincBeam(beamwidth_deg=float(antenna.get("beamwidth_deg",3.0)),scan_rate_rpm=0.0,fixed_azimuth_deg=float(motion.azimuth_deg),sidelobe_floor_db=-50.0);pattern_db=beam.gain_db(state.aoa_deg,0.0);prx_dbm=received_power_dbm(state.tx_peak_power_w,float(antenna.get("peak_gain_dbi",0.0)),pattern_db,float(state.mode["frequency_hz"]),state.range_km,rx_gain_dbi);amplitude=min(.90,max(0.0,.10*10.**((prx_dbm-threshold_dbm)/20.0)));iq=faces[face_for_bearing(state.aoa_deg)]
                self._add_pulse(iq,block_start_s,toa_s,float(state.mode["pw_us"])*1e-6,float(state.mode["frequency_hz"]),amplitude,state.mode.get("waveform","CW"),float(state.mode.get("bandwidth_hz",0.0)))
    def read_block(self):
        bs=self.time_s;be=bs+self.block_duration_s;faces={f:np.zeros(self.block_samples,dtype=np.complex64) for f in FACE_CENTRES_DEG};self._render_e1_e2(faces,bs,be);self._render_e3(faces,bs,be)
        if self.noise_std>0.0:
            for f in faces:
                noise=(self.rng.normal(0.,self.noise_std,self.block_samples)+1j*self.rng.normal(0.,self.noise_std,self.block_samples)).astype(np.complex64);faces[f]+=noise
        metadata={"sample_rate_hz":self.sample_rate_hz,"center_frequency_hz":self.center_frequency_hz,"start_time_s":bs,"sample_index":self.sample_index,"receiver_faces":tuple(FACE_CENTRES_DEG)};self.sample_index+=self.block_samples;return faces,metadata
