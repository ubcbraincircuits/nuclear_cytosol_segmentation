import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.signal as signal
import skimage.filters as filters
import skimage.measure as measure
from IPython.display import display, clear_output
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from PIL import Image
#from pyneurotrace import filters  as pntfilters
from scipy import integrate
from scipy.stats import sem

"""
Performs fast nonnegative deconvolution on pmt signal to solve for minimum MSE photon rate
   trace  :   The data to be deconvolved
   tau    :   The time constant of the PMT, in data samples
   return :   estimated photon rate
A matlab version is also available on request.
For details on how this works, see:
  Podgorski, K., & Haas, K. (2013).
  Fast non‐negative temporal deconvolution for laser scanning microscopy.
  Journal of biophotonics, 6(2), 153-162.
"""
def nonNegativeDeconvolution(trace, tau):
    T = len(trace)
    counts = np.zeros(T)
    counts[-1] = trace[-1]
    cutoff = math.ceil(8 * tau)
    kernel = np.exp(-np.arange(cutoff + 1)/tau) # convolution kernel
    recent = np.full(1 + round(T / 2), np.nan).astype(int)
    recent[0] = T #stored locations where we assigned counts
    recentIdx = 0

    # the points that could potentially be assigned counts:
    _delayed = np.concatenate(([0], trace[:-2]))
    points = (trace[:-1] > kernel[1] * _delayed) & (trace[:-1] > 0)

    # dividing these points up into runs, for speed
    runStarts = np.where(points & ~(np.concatenate(([False], points[:-1]))))[0].astype(int)
    runEnds = np.where(points & ~(np.concatenate((points[1:], [False]))))[0].astype(int)
    runIdx = len(runEnds) - 1

    while runIdx >= 0:
        oldTop, oldBottom = 0, 0
        t = runEnds[runIdx]
        t1 = t
        accum = 0

        converged = False
        while not converged:
            if recentIdx >= 0 and recent[recentIdx] < (t+cutoff):
                t2 = recent[recentIdx] - 1
                C_max = counts[t2] / kernel[t2-t]
            else:
                t2 = min(t + cutoff, T+1) - 1
                C_max = np.inf


            b = kernel[t1-t:t2-t]
            top = np.dot(b, trace[t1:t2]) + oldTop #this is the numerator of the least squares fit for an exponential
            bottom = np.dot(b, b) + oldBottom #this is the denominator of the fit

            done = False
            while not done:
                #the error function is (data-kernel.*C)^2
                bestC = max(top/bottom, 0);  #C=top/bottom sets the derivative of the error to 0

                # does not meet nonnegative constraint. Continue to adjust previous solutions.
                if bestC > (C_max+accum):
                    accum = accum + counts[t2] / kernel[t2-t]
                    counts[t2] = 0
                    t1 = t2
                    oldTop = top
                    oldBottom = bottom
                    recentIdx -= 1
                    done = True

                else: # converged!
                    #now that we have found the MSE counts for times t<end, check if
                    #this will be swamped by the next timepoint in the run
                    if  (t == runStarts[runIdx]) or (trace[t-1] < bestC/kernel[1]): #%C_max won't necessarily get swamped
                        if recentIdx >= 0 and t2 <= t + cutoff:
                            counts[t2] = counts[t2] - (bestC - accum) * kernel[t2-t]
                        runStart = runStarts[runIdx]
                        initIdx = recentIdx + 1
                        recentIdx = recentIdx + 1 + t - runStart;

                        _skipped = 0
                        if recentIdx + 1 > len(recent):
                            _skipped = recentIdx - (len(recent) - 1)
                            recentIdx = len(recent) - 1


                        recent[initIdx:recentIdx + 1] = np.arange(t+1, runStart + _skipped, -1)
                        counts[runStart:(t+1)] = \
                               np.concatenate((trace[runStart:t], [bestC])) - \
                               np.concatenate(([0], kernel[1]*trace[runStart:t]))
                        done = True
                        converged = True
                    else: #%C_max will get swamped
                        #%in this situation, we know that this point will be removed
                        #%as we continue to process the run. To save time:
                        t -= 1
                        runEnds[runIdx] = t
                        accum = accum / kernel[1]
                        top = top * kernel[1] + trace[t] #% %this is the correct adjustment to the derivative term above
                        bottom = bottom * (kernel[1] ** 2) + 1 #% %this is the correct adjustment to the derivative term above

        runIdx -= 1
    return counts

def nndSmooth(data, hz, tau, iterFunc=None):
    tauSamples = tau * hz

    # This is the transient shape we're deconvolving against:
    # e^(x/tauSamples), for 8 times the length of tau.
    cutoff = round(8 * tauSamples)
    fitted = np.exp(-np.arange(cutoff + 1) / tauSamples)

    def _singleRowNND(samples):
        result = np.copy(samples)
        nanSamples = np.isnan(samples)
        if np.all(nanSamples):
            pass # No data
        elif not np.any(nanSamples):
            # All samples exist, so fit in one go
            result = np.convolve(nonNegativeDeconvolution(samples, tauSamples), fitted)[:len(samples)]
        else:
            # Lots of different runs of samples, fit each separately
            starts = np.where((not nanSamples) & np.isnan(np.concatenate(([1], samples[:-1]))))[0]
            ends = np.where((not nanSamples) & np.isnan(np.concatenate((samples[1:], [1]))))[0]
            for start, end in zip(starts, ends):
                tmp = np.convolve(NND(samples[start:end], tauSamples), fitted)
                result[start:end] = np.max(0, tmp[:end - start + 1])
        return result

    return _forEachTimeseries(data, _singleRowNND, iterFunc)

def deltaFOverF0(data, hz, t0=0.2, t1=0.75, t2=3.0, iterFunc=None):
    t0ratio = None if t0 is None else np.exp(-1 / (t0 * hz))
    t1samples, t2samples = round(t1 * hz), round(t2*hz)

    def _singeRowDeltaFOverF(samples):
        fBar = _windowFunc(np.mean, samples, t1samples, mid=True)
        f0 = _windowFunc(np.min, fBar, t2samples)
        result = (samples - f0) / f0
        if t0ratio is not None:
            result = _ewma(result, t0ratio)
        return result
    return _forEachTimeseries(data, _singeRowDeltaFOverF, iterFunc)


def _windowFunc(f, x, window, mid=False):
    n = len(x)
    startOffset = (window - 1) // 2 if mid else window - 1
    result = np.zeros(x.shape)
    for i in range(n):
        startIdx = i - startOffset
        endIdx = startIdx + window
        startIdx, endIdx = max(0, startIdx), min(endIdx, n)
        result[i] = f(x[startIdx:endIdx])
    return result


def _ewma(x, ratio):
    result = np.zeros(x.shape)
    weightedSum, sumOfWeights = 0.0, 0.0
    for i in range(len(x)):
        weightedSum = ratio * weightedSum + x[i]
        sumOfWeights = ratio * sumOfWeights + 1.0
        result[i] = weightedSum / sumOfWeights
    return result

# Input is either 1d (timeseries), 2d (each row is a timeseries) or 3d (x, y, timeseries)
def _forEachTimeseries(data, func, iterFunc=None):
    if iterFunc is None:
        iterFunc = lambda x: x
    dim = len(data.shape)
    result = np.zeros(data.shape)
    if dim == 1: # single timeseries
        result = func(data)
    elif dim == 2: # (node, timeseries)
        for i in iterFunc(range(data.shape[0])):
            result[i] = func(data[i])
    elif dim == 3: # (x, y, timeseries)
        for i in iterFunc(range(data.shape[0])):
            for j in iterFunc(range(data.shape[1])):
                result[i, j] = func(data[i, j])
    return result

# Take a folder of Tifs and turn it into a numpy array
def folder2tif(dir_path):
    final = []
    files = os.listdir(dir_path)
    files = sorted(files)
    movie = []
    for fname in files:
        im = Image.open(os.path.join(dir_path, fname))
        imarray = np.array(im)
        movie.append(imarray)
    movie = np.asarray(movie)
    return movie



class cell():
    def __init__(self, path, cell, hz, power, fc, std_thresh):
        self.path = path
        self.cell = cell
        self.hz = hz
        self.power = power
        self.fc = fc
        self.std_thresh = std_thresh
        print(self.cell, ': Analysis Beginning')


        # Returns change in fluorescence over average fluorescence of ROI
        def deltaF(video_mask):
            video_mask_nan = video_mask.copy()
            video_mask_nan[video_mask_nan==0] = np.nan
            mean = np.mean(np.nanmean(video_mask))
            dff = np.zeros((video_mask.shape[0]))
            for i in range(dff.shape[0]):
                delta = np.nanmean(video_mask[i, :, :])-mean
                dff[i] = delta/mean
            return dff

        # Returns raw values fluorescence in the ROI
        def rawIntensity(video_mask):
            video_mask_nan = video_mask.copy()
            video_mask_nan[video_mask_nan==0] = np.nan
            mean = np.nanmean(video_mask, axis=(1,2))
            return mean

        # Generates the ROI
        def genROI(gcamp, rcamp):

            # To create a ROI for the nucleus a STD projection is created
            # Thresholding this image creates a mask for the roi
            std_projectionG = np.std(gcamp, axis=0)
            threshold = filters.threshold_otsu(std_projectionG)
            std_projectionG[std_projectionG < threshold] = 0
            std_projectionG[std_projectionG>0]=1

            # Create a ROI for the cytosl using an STD projection
            # Thresholding this image creates a mask for the roi
            std_projectionR = np.std(rcamp, axis=0)
            threshold = filters.threshold_otsu(std_projectionR)
            std_projectionR[std_projectionR < threshold] = 0
            std_projectionR[std_projectionR>0]=1

            # Remove the Nuclear Mask from this ROI
            std_projectionR[std_projectionG==1]=0

            # Applying the masks for the two channels
            gcamp_masked = gcamp * std_projectionG
            rcamp_masked = rcamp * std_projectionR
            return gcamp_masked, rcamp_masked, std_projectionR, std_projectionG


        def Cell2Trace(self, path, cell):
            # Import the movies provide the path to the folder containing the frames
            gcamp = folder2tif(path+cell+"_G/")
            rcamp = folder2tif(path+cell+"_R/")

            # Return Masked Arrays
            gcamp_masked, rcamp_masked, mask_r, mask_g = genROI(gcamp, rcamp)

            # Return Raw Traces from ROI
            gcamp_raw = rawIntensity(gcamp_masked)
            rcamp_raw = rawIntensity(rcamp_masked)

            return gcamp, rcamp, gcamp_masked, rcamp_masked, gcamp_raw, rcamp_raw, mask_r, mask_g
        self.gcamp, self.rcamp, self.gcamp_masked, self.rcamp_masked, self.gcamp_raw, self.rcamp_raw, self.mask_r, self.mask_g = Cell2Trace(self, path, cell)
        print(self.cell, ": Images Loaded")

        # Calculate df/f and perform NND
        self.dffG = deltaFOverF0(self.gcamp_raw, self.hz)
        self.nndG = nndSmooth(self.dffG, self.hz, tau=1)
        self.dffR = deltaFOverF0(self.rcamp_raw, self.hz)
        self.nndR = nndSmooth(self.dffR, self.hz, tau=1)
        print(self.cell, ": Traces Completed")

        def peakDetect(trace, endpoint, power, fc, std_thresh):
            # Third Order Butterworth lowpass filter; 3hz cutoff

            fc = fc  # Cut-off frequency of the filter
            w = fc / (10 / 2) # Normalize the frequency
            b, a = signal.butter(power, w, 'low', analog=True)
            z = signal.lfilter(b, a, trace, axis=0)

            #plt.plot(trace, linewidth=2, color='Black')
            #plt.plot(z, linewidth=2, color='Red')
            #plt.show()


            # Detect Peaks
            threshold = std_thresh * np.std(z[:endpoint])

            peaks, _ = signal.find_peaks(trace, width=7, rel_height=.5, prominence=(threshold))

            width = signal.peak_widths(trace, peaks, rel_height=.1)

            return peaks, width[0], threshold, z

        def signal_analysis(self, cell_id, gcamp, rcamp, rawG, rawR, HZ, power, fc, std_thresh):
            str_index = int(cell_id.find("Iono_"))
            threshold_cuttoff = (int(cell_id[(str_index+5):(str_index+8)])*HZ)
            gcamp_peaks, gcamp_widths, gThreshold, gLowPass = peakDetect(gcamp, threshold_cuttoff, power, fc, std_thresh)
            rcamp_peaks, rcamp_widths, rThreshold, rLowPass = peakDetect(rcamp, threshold_cuttoff, power, fc, std_thresh)


            drug_app = np.nan
            iono_min = [np.nan, np.nan]
            iono_max = [np.nan, np.nan]
            iono_diff = [np.nan, np.nan]

            # Update: Now uses value of drug app from folder name
            # Should return -1 is no match is found for the key string 'Iono_'
            if cell_id.find("Iono_") is not -1:
                str_index = int(cell_id.find("Iono_"))
                drug_app = (int(cell_id[(str_index+5):(str_index+8)])*HZ)
                cutoffG = np.array(np.where(gcamp_peaks >= drug_app)[0])
                if cutoffG.size !=0:
                    cutoffG = np.min(cutoffG)

                if cutoffG != 0:

                    if cutoffG.size !=0:
                        cutoffG = np.min(cutoffG)

                    cutoffR = np.array(np.where(rcamp_peaks >= drug_app))
                    if cutoffR.size !=0:
                        cutoffR = np.min(cutoffR)

                    gcamp_peaks = gcamp_peaks[:cutoffG]
                    gcamp_widths = gcamp_widths[:cutoffG]
                    rcamp_peaks = rcamp_peaks[:cutoffR]
                    rcamp_widths = rcamp_widths[:cutoffR]


            if cell_id.find("_S") is not -1:
                if drug_app is not np.nan:
                    iono_min[0] = np.min(rawG[drug_app-200:drug_app+1000]>0)
                    iono_min[1] = np.min(rawR[drug_app-200:drug_app+1000]>0)

                    iono_max[0] = np.max(rawG[drug_app-200:drug_app+1000])
                    iono_max[1] = np.max(rawR[drug_app-200:drug_app+1000])

                    iono_diff[0] = iono_max[0]-iono_min[0]
                    iono_diff[1] = iono_max[1]-iono_min[1]


            # Match Peaks and puttin them in a list of tuples (g, r)
            gcamp_matched = []
            shared_rcamp = []
            shared_gcamp = []
            for g in gcamp_peaks:
                for r in rcamp_peaks:
                    a = math.isclose(g, r, abs_tol=15)
                    if a == True:
                        gcamp_matched.append((g,r))
                        shared_rcamp.append(r)
                        shared_gcamp.append(g)

            rcamp_only = list(set(rcamp_peaks)-set(shared_rcamp))
            rcamp_only.sort
            gcamp_only = list(set(gcamp_peaks)-set(shared_gcamp))
            gcamp_only.sort

            if len(gcamp_peaks) == 0:
                g_percent_shared = np.nan
            else:
                g_percent_shared = (len(gcamp_matched)/len(gcamp_peaks))

            if len(rcamp_peaks) == 0:
                r_percent_shared = np.nan
            else:
                r_percent_shared = (len(gcamp_matched)/len(rcamp_peaks))
            if len(gcamp_matched) == 0:
                r_percent_shared = np.nan
                g_percent_shared = np.nan

            # General Stats for the cell
            cell_stats = {
                                        'Cell ID': cell_id,
                                        'GCaMP Peaks':len(gcamp_peaks),
                                        'RCaMP Peaks':len(rcamp_peaks),
                                        'Shared Peaks':len(gcamp_matched),
                                        'GCaMP Percent Shared':g_percent_shared,
                                        'RCaMP Percent Shared':r_percent_shared,
                                        'Experiment Length (s)': gcamp.shape[0]/10,
                                        'Drug Application': drug_app/10,
                                        'Iono GCaMP Max': iono_max[0],
                                        'Iono GCaMP Dif': iono_diff[0],
                                        'Iono RCaMP Max': iono_max[1],
                                        'Iono RCaMP Dif': iono_diff[1],


                                       }

            cell_stats = pd.DataFrame(data=cell_stats, index=[0])

            # Peak Data for Shared Peaks
            shared_peak_data = pd.DataFrame()
            for event in gcamp_matched:

                gindex = np.where(gcamp_peaks == event[0])[0]
                rindex = np.where(rcamp_peaks == event[1])[0]


                # Integrate Under the Curve for Area
                # Note: Area from start to peak
                g_event_start = int(event[0]-gcamp_widths[gindex])
                if g_event_start < 0:
                    g_event_start = 0
                r_event_start = int(event[0]-rcamp_widths[rindex])
                if r_event_start < 0:
                    r_event_start = 0

                g_area = integrate.cumtrapz(gcamp[g_event_start:event[0]])
                if len(g_area) is not 0:
                    g_area = g_area[-1]
                r_area = integrate.cumtrapz(rcamp[r_event_start:event[1]])
                if len(r_area) is not 0:
                    r_area = r_area[-1]

                peak_stats = {          'Cell ID': cell_id,
                                        'GCaMP Loc':event[0],
                                        'GCaMP Start': event[0]-gcamp_widths[gindex][0]*HZ,
                                        'GCaMP Width':gcamp_widths[gindex][0],
                                        'GCaMP Prominence':gcamp[event[0]],
                                        'GCaMP Area':g_area,
                                        'RCaMP Loc':event[1],
                                        'RCaMP Start': event[1]-rcamp_widths[rindex][0]*HZ,
                                        'RCaMP Width':rcamp_widths[rindex][0],
                                        'RCaMP Prominence':rcamp[event[1]],
                                        'RCaMP Area':r_area,
                                        'Promicence Ratio (G/R)':(gcamp[event[1]]/rcamp[event[0]]),
                                        'Peak Time Diff (G-R)':((event[0]-event[1])*100),
                                        'Start Difference (G-R)': (event[0]-gcamp_widths[gindex] - event[1]-rcamp_widths[rindex])[0]*100
                                                                         }
                shared_peak_data = shared_peak_data.append(peak_stats, ignore_index=True)

            # Adding RCaMP peaks to the shared datatable
            for event in rcamp_only:

                rindex = np.where(rcamp_peaks == event)[0]


                # Integrate Under the Curve for Area
                # Note: Area from start to peak
                r_event_start = int(event-rcamp_widths[rindex])
                if r_event_start < 0:
                    r_event_start = 0

                r_area = integrate.cumtrapz(rcamp[r_event_start:event])
                if len(r_area) is not 0:
                    r_area = r_area[-1]

                peak_stats = {          'Cell ID': cell_id,
                                        'GCaMP Loc':np.nan,
                                        'GCaMP Start': np.nan,
                                        'GCaMP Width':np.nan,
                                        'GCaMP Prominence':np.nan,
                                        'GCaMP Area':np.nan,
                                        'RCaMP Loc':event,
                                        'RCaMP Start': event-rcamp_widths[rindex][0]*HZ,
                                        'RCaMP Width':rcamp_widths[rindex][0],
                                        'RCaMP Prominence':rcamp[event],
                                        'RCaMP Area':r_area,
                                        'Promicence Ratio (G/R)':0,
                                        'Peak Time Diff (G-R)':np.nan,
                                        'Start Difference (G-R)': np.nan,
                                                                         }
                shared_peak_data = shared_peak_data.append(peak_stats, ignore_index=True)



            # Peak Data for exclusive GCaMP Peaks
            gcamp_peak_data = pd.DataFrame()
            for event in gcamp_only:
                gindex = np.where(gcamp_peaks == event)[0]

                # Integrate Under the Curve for Area
                # Note: Area from start to peak
                g_event_start = int(event-gcamp_widths[gindex])
                if g_event_start < 0:
                    g_event_start = 0

                g_area = integrate.cumtrapz(gcamp[g_event_start:event])
                if len(g_area) is not 0:
                    g_area = g_area[-1]

                peak_stats = {          'Cell ID': cell_id,
                                        'GCaMP Loc':event,
                                        'GCaMP Start': event-gcamp_widths[gindex][0]*HZ,
                                        'GCaMP Width':gcamp_widths[gindex][0],
                                        'GCaMP Prominence':gcamp[event],
                                        'GCaMP Area':g_area,
                                                                 }

                gcamp_peak_data = gcamp_peak_data.append(peak_stats, ignore_index=True)

            # Peak Data for exclusive RCaMP Peaks
            rcamp_peak_data = pd.DataFrame()

            for event in rcamp_only:
                rindex = np.where(rcamp_peaks == event)[0]
                # Integrate Under the Curve for Area
                # Note: Area from start to peak
                r_event_start = int(event-rcamp_widths[rindex])
                if r_event_start < 0:
                    r_event_start = 0

                r_area = integrate.cumtrapz(rcamp[r_event_start:event])
                if len(r_area) is not 0:
                    r_area = r_area[-1]

                peak_stats = {          'Cell ID': cell_id,
                                        'RCaMP Loc':event,
                                        'RCaMP Start': (event-rcamp_widths[rindex][0])*HZ,
                                        'RCaMP Width':rcamp_widths[rindex][0],
                                        'RCaMP Prominence':rcamp[event],
                                        'RCaMP Area':r_area,

                                                                         }
                rcamp_peak_data = rcamp_peak_data.append(peak_stats, ignore_index=True)


                cell_stats["Prominence Ratio Mean"] = np.nanmean(np.array(shared_peak_data["Promicence Ratio (G/R)"]))
                cell_stats["Prominence Ratio SEM"] = sem(np.array(shared_peak_data["Promicence Ratio (G/R)"]))

            return cell_stats, shared_peak_data, gcamp_peak_data, rcamp_peak_data, gThreshold, rThreshold, gcamp_peaks, rcamp_peaks, rLowPass, gLowPass
        self.cell_stats, self.shared_peak_data, self.gcamp_peak_data, self.rcamp_peak_data, self.gThreshold, self.rThreshold, self.gcamp_peaks, self.rcamp_peaks, self.rLowPass, self.gLowPass = signal_analysis(self, self.cell, self.nndG, self.nndR, self.gcamp_raw, self.rcamp_raw, self.hz, self.power, self.fc, self.std_thresh)
        print(self.cell, ": Analysis Completed\n")

    def inspect_peaks(self):
        gtrace = self.nndG
        gpeaks = self.gcamp_peaks
        gthreshold = self.gThreshold
        gLowPass = self.gLowPass

        plt.axhline(y=gthreshold)
        plt.plot(gpeaks, gtrace[gpeaks],  'x')
        plt.plot(gtrace, color='green')
        plt.plot(gLowPass)
        plt.show()

        rtrace = self.nndR
        rpeaks = self.rcamp_peaks
        rthreshold = self.rThreshold
        rLowPass = self.rLowPass
        plt.axhline(y=rthreshold)
        plt.plot(rpeaks, rtrace[rpeaks],  'x')
        plt.plot(rtrace, color='red')
        plt.plot(rLowPass)
        plt.show()

    def inspect_results(self):
        def plot(Frame):
            gmask=self.mask_g.copy()
            gmask[gmask>0]=1


            channel1 = self.gcamp.copy()
            channel2 =self.rcamp.copy()


            gclosed = gmask.copy()
            gcontours = []
            #mask_values = np.unique(masks)
            gclosed = gmask.copy()
            gclosed[gclosed !=1]=0
            gcontours.append(measure.find_contours(gclosed, .4))
            gcontour = gcontours[0][0]
            fig, ax = plt.subplots(ncols=2)

            ax[0].imshow(channel1[Frame,:,:], cmap='magma')
            ax[0].plot(gcontour[:, 1],gcontour[:, 0], linewidth=4, color='white')



            rmask=self.mask_r.copy()

            rmask[rmask>0]=1

            rclosed = rmask.copy()
            rcontours = []
            #mask_values = np.unique(masks)
            rclosed = rmask.copy()
            rclosed[rclosed !=1]=0
            rcontours.append(measure.find_contours(rclosed, .4))
            rcontour = rcontours[0][0]
            ax[1].plot(gcontour[:, 1],gcontour[:, 0], linewidth=4, color='white')
            ax[1].plot(rcontour[:, 1],rcontour[:, 0], linewidth=4, color='white')
            ax[1].imshow(channel2[Frame,:,:], cmap='magma')

            fig2, ax2 = plt.subplots(2)
            gtrace = self.nndG
            gpeaks = self.gcamp_peaks

            ax2[0].axvline(x=Frame)
            ax2[0].plot(gpeaks, gtrace[gpeaks],  'x')
            ax2[0].plot(gtrace, color='green')

            rtrace =self.nndR
            rpeaks =self.rcamp_peaks
            ax2[1].axvline(x=Frame)
            ax2[1].plot(rpeaks, rtrace[rpeaks],  'x')
            ax2[1].plot(rtrace, color='red')
            plt.show()



        display(interact(plot, Frame=widgets.IntSlider(min=0, max=(self.gcamp.shape[0]-1), step=1, value=0)))
