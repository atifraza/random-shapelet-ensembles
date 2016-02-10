/**
 * 
 */
package org.kramerlab.timeseries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

/**
 * @author Atif Raza
 */
public class TimeSeriesDataset {
	protected ArrayList<TimeSeries> dataset;
	protected HashMap<Integer, Integer> instsPerClass;
	protected boolean isClassHistUpdated;

	public TimeSeriesDataset() {
		this.dataset = new ArrayList<TimeSeries>();
		this.instsPerClass = new HashMap<Integer, Integer>();
		this.isClassHistUpdated = true;
	}

	public TimeSeriesDataset(ArrayList<TimeSeries> dataset) {
		this.dataset = dataset;
		this.instsPerClass = new HashMap<Integer, Integer>();
		this.updateClassHist();
	}

	public TimeSeries get(int index) {
		if (index < 0) {
			throw new IndexOutOfBoundsException("Index can not be negative");
		} else if (index > this.dataset.size()) {
			throw new IndexOutOfBoundsException("The index is larger than the TimeSeries objects in dataset");
		} else {
			return this.dataset.get(index);
		}
	}

	public void add(TimeSeries t) {
		this.dataset.add(t);
        this.isClassHistUpdated = false;
	}

	public void add(ArrayList<TimeSeries> insts) {
		for (TimeSeries t : insts) {
			this.add(t);
		}
		this.isClassHistUpdated = false;
	}

	public int size() {
		return this.dataset.size();
	}

	public double entropy() {
		if (!this.isClassHistUpdated) {
			this.updateClassHist();
		}

		int N = this.dataset.size();
		double frac, entropy = 0;
		for (Integer count : this.instsPerClass.keySet()) {
			frac = (double) this.instsPerClass.get(count) / N;
			entropy += (frac > 0.0) ? -1 * (frac) * Math.log(frac) : 0.0;
		}
		return entropy;
	}

	public void updateClassHist() {
	    if (this.isClassHistUpdated) {
	        return;
	    }
		int currCount;
		Integer clsLabel;
		this.instsPerClass.clear();
		for (TimeSeries t : this.dataset) {
			clsLabel = t.getLabel();
			currCount = this.instsPerClass.getOrDefault(clsLabel, 0);
			this.instsPerClass.put(clsLabel, currCount + 1);
		}
		this.isClassHistUpdated = true;
	}

	public Set<Integer> getAllClasses() {
		return this.instsPerClass.keySet();
	}
    
    public int getNumOfClasses() {
        return this.instsPerClass.keySet().size();
    }
    
    public int getInstCount(int instClass) {
        return this.instsPerClass.getOrDefault(instClass, 0);
    }
    
    public HashMap<Integer, Integer> getClassHist() {
        if (!this.isClassHistUpdated) {
            this.updateClassHist();
        }
        return this.instsPerClass;
    }
    
    public void remove(int ind) {
        this.dataset.remove(ind);
    }
    
    public void shuffle() {
        Collections.shuffle(dataset);
    }
}
