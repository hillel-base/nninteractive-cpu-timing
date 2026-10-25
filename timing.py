"""Research timing test: how long one nnInteractive click takes on a CPU-only x86 machine, and its peak memory.
Downloads one public TCIA scan (RIDER Lung CT, patient RIDER-2619750334, 1.25 mm) and its RIDER-LungCT-Seg outline
from NCI Imaging Data Commons, and the official nnInteractive v1.0 weights, at run time. Neither is stored here.
Protocol: one click at the deepest point inside the reference outline, then 2 simulated corrections, each at the
deepest point of the largest remaining error. Writes results.json: machine, timings, peak memory, Dice per click.
Usage: python timing.py <window side in voxels, e.g. 160> <out_dir>"""
import glob, json, os, platform, sys, threading, time
t_start = time.time()
import numpy as np, psutil, pydicom, SimpleITK as sitk, torch
from scipy import ndimage

SIDE, OUT = int(sys.argv[1]), sys.argv[2]
assert SIDE % 32 == 0, "the network halves the window 5 times: use a multiple of 32"
os.makedirs(OUT, exist_ok=True)
CT_UID = "1.3.6.1.4.1.9328.50.1.307319453402340384126936296040122224868"  # RIDER Lung CT, RIDER-2619750334, 1.25 mm
SEG_UID = "1.2.276.0.7230010.3.1.3.8323329.12891.1554820797.597451"  # RIDER-LungCT-Seg outline for that series
me = psutil.Process()
R = {"window_voxels": [SIDE] * 3, "stages": {}}

peak = [0]
def _sample():  # resident memory every 50 ms; the operating system's own peak counter is added below
    while True:
        peak[0] = max(peak[0], me.memory_info().rss); time.sleep(0.05)
threading.Thread(target=_sample, daemon=True).start()
def os_peak():
    mi = me.memory_info()
    if hasattr(mi, "peak_wset"): return mi.peak_wset  # Windows: peak working set
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KB
def stage(name, t0):
    R["stages"][name] = {"seconds": round(time.time() - t0, 2), "peak_gb_so_far": round(max(peak[0], os_peak()) / 1e9, 2)}
    print(name, R["stages"][name], flush=True)

try:
    import cpuinfo; cpu = cpuinfo.get_cpu_info().get("brand_raw")
except Exception: cpu = platform.processor()
R["machine"] = {"os": platform.platform(), "cpu": cpu, "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "torch": torch.__version__, "onednn": torch.backends.mkldnn.is_available(),
                "python": platform.python_version()}
print(R["machine"], flush=True)

t = time.time()
from idc_index import index
index.IDCClient().download_from_selection(seriesInstanceUID=[CT_UID, SEG_UID], downloadDir="dl", dirTemplate="%Modality")
stage("download_data", t)

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.model_management import ensure_model_available, get_default_model_id
t = time.time(); folder = ensure_model_available(get_default_model_id()); stage("download_weights", t)
t = time.time()
warmup = nnInteractiveInferenceSession.warmup; nnInteractiveInferenceSession.warmup = lambda self: None
s = nnInteractiveInferenceSession(device=torch.device("cpu"), use_torch_compile=False, verbose=False,
                                  torch_n_threads=psutil.cpu_count(), do_autozoom=False, interactions_storage="tensor")
s.initialize_from_trained_model_folder(folder)
s.configuration_manager.configuration["patch_size"] = [SIDE] * 3  # set the window before the warm-up runs
net = s.network.float(); _fwd = net.forward; net.forward = lambda x, *a, **k: _fwd(x.float(), *a, **k)  # 32-bit network
R["machine"]["torch_threads"] = torch.get_num_threads()
stage("load_model", t)
t = time.time(); warmup(s); stage("warm_up", t)

t = time.time()
r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(os.path.join("dl", "CT"))); ct = r.Execute()
ds = pydicom.dcmread(glob.glob(os.path.join("dl", "SEG", "*.dcm"))[0])  # reference outline: the MANUAL segment
manual = [x.SegmentNumber for x in ds.SegmentSequence if x.get("SegmentAlgorithmType") == "MANUAL"]
assert len(manual) == 1
E = np.zeros(ct.GetSize()[::-1], bool)
for f, fg in zip(ds.pixel_array.reshape(-1, ds.Rows, ds.Columns), ds.PerFrameFunctionalGroupsSequence):
    if fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber == manual[0]:
        x, y, z = ct.TransformPhysicalPointToIndex([float(v) for v in fg.PlanePositionSequence[0].ImagePositionPatient])
        E[z] |= f.astype(bool)
sp = ct.GetSpacing()
def deepest(mask):
    lab, n = ndimage.label(mask)
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return tuple(int(i) for i in np.unravel_index(np.argmax(ndimage.distance_transform_edt(big)), mask.shape))
def dice(a, b): return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))
bb = tuple(slice(max(0, b.start - 1), b.stop + 1) for b in ndimage.find_objects(E.astype(np.uint8))[0])
c0 = tuple(c + b.start for c, b in zip(deepest(E[bb]), bb))
half = (round(60 / sp[2]), round(90 / sp[1]), round(90 / sp[0]))  # a 120 x 180 x 180 mm box around the click
lo = [max(0, a - h) for a, h in zip(c0, half)]; hi = [min(n, a + h) for a, h, n in zip(c0, half, E.shape)]
crop = tuple(slice(a, b) for a, b in zip(lo, hi)); E = E[crop]
arr = sitk.GetArrayFromImage(ct)[crop].astype(np.float32)[None]
buf = np.zeros(arr.shape[1:], np.uint8)
s.set_image(arr); s.set_target_buffer(buf)
stage("load_scan_and_set_image", t)

clicks = []
click, positive = tuple(a - l for a, l in zip(c0, lo)), True
for k in range(3):
    t = time.time(); s.add_point_interaction(click, include_interaction=positive); secs = time.time() - t
    M = buf.astype(bool)
    clicks.append({"seconds": round(secs, 2), "dice_vs_reference": round(dice(E, M), 4)})
    stage(f"click{k + 1}", t)
    fn, fp = E & ~M, M & ~E
    if not (fn.any() or fp.any()): break
    positive = bool(fn.sum() >= fp.sum()); click = deepest(fn if positive else fp)
R["clicks"] = clicks
R["median_click_seconds"] = float(np.median([c["seconds"] for c in clicks]))
R["peak_gb"] = round(max(peak[0], os_peak()) / 1e9, 2)
R["total_seconds"] = round(time.time() - t_start, 1)
json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
