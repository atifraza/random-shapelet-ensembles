package org.kramerlab.rulelearners;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import org.kramerlab.shapelets.*;
import org.kramerlab.timeseries.*;

public class Ripper {
    protected static TimeSeriesDataset trainSet;
    protected static BaseShapelets shapeletFinder;
    protected static int minLen, maxLen, stepSize;
    
    protected ArrayList<Rule> ruleSet;
    private TimeSeriesDataset growPos, growNeg;
    private TimeSeriesDataset prunePos, pruneNeg;
    private int posSize = 0, growPosSize;
    private int negSize = 0, growNegSize;
    private ArrayList<Integer> posList, negList;
    
    public Ripper(TimeSeriesDataset trainSet, int minLen, int maxLen,
                  int stepSize) {
        Ripper.trainSet = trainSet;
        Ripper.minLen = minLen;
        Ripper.maxLen = maxLen;
        Ripper.stepSize = stepSize;
        ruleSet = new ArrayList<Rule>();
        createRules();
    }
    
    public void createRules() {
        TimeSeriesDataset pos, neg;
        Map<Integer, Integer> sortedTrainSetClassMap = sortByValue(Ripper.trainSet.getClassHist());
        for (Integer key : sortedTrainSetClassMap.keySet()) {
            pos = new TimeSeriesDataset();
            neg = new TimeSeriesDataset();
            for (int i = 0; i < Ripper.trainSet.size(); i++) {
                if (Ripper.trainSet.get(i).getLabel() == key) {
                    pos.add(Ripper.trainSet.get(i));
                } else {
                    neg.add(Ripper.trainSet.get(i));
                }
            }
            startRipper(pos, neg);
            ArrayList<Integer> removalList = new ArrayList<Integer>();
            ListIterator<Integer> li;
            for (int i = 0; i < Ripper.trainSet.size(); i++) {
                if (Ripper.trainSet.get(i).getLabel() == key) {
                    removalList.add(i);
                }
            }
            li = removalList.listIterator(removalList.size());
            while (li.hasPrevious()) {
                Ripper.trainSet.remove(li.previous());
            }
        }
    }
    
    private void startRipper(TimeSeriesDataset pos, TimeSeriesDataset neg) {
        Rule r = null;
        while (pos.size() != 0) {
            split(pos, neg);
            r = growRule();
//            r = pruneRule(r);
            if (checkInsts(r) > 0.5) {
                return;
            } else {
                ruleSet.add(r);
                ArrayList<Integer> removalList = new ArrayList<Integer>();
                ListIterator<Integer> li;
                TimeSeries t;
                for (int i = 0; i < pos.size(); i++) {
                    t = pos.get(i);
                    if (r.eval(t) == t.getLabel()) {
                        removalList.add(i);
                    }
                }
                li = removalList.listIterator(removalList.size());
                while (li.hasPrevious()) {
                    pos.remove(li.previous());
                }
                removalList.clear();
                for (int i = 0; i < neg.size(); i++) {
                    t = neg.get(i);
                    if (r.eval(t) == t.getLabel()) {
                        removalList.add(i);
                    }
                }
                li = removalList.listIterator(removalList.size());
                while (li.hasPrevious()) {
                    pos.remove(li.previous());
                }
            }
        }
    }
    
    private Rule growRule() {
        Rule r = new Rule();
        Shapelet s;
        if (growNeg.size() == 0) {
            r.setConsequent(growPos.get(0).getLabel());
        } else if (growPos.size() == 0) {
            r.setConsequent(growNeg.get(0).getLabel());
        } else {
            TimeSeriesDataset growSet = new TimeSeriesDataset();
            TimeSeries t;
            int tLabel = Integer.MAX_VALUE;
            int i;
            for (i = 0; i < Math.min(growPos.size(), growNeg.size()); i++) {
                growSet.add(growPos.get(i));
                t = new TimeSeries(growNeg.get(i).getTS(), tLabel);
                growSet.add(t);
            }
            for (i = Math.min(growPos.size(), growNeg.size()); i < Math.max(growPos.size(), growNeg.size()); i++) {
                t = new TimeSeries(growNeg.get(i).getTS(), tLabel);
                growSet.add(t);
            }
//            growSet.shuffle();
            Ripper.shapeletFinder = new LegacyShapelets(growSet, Ripper.minLen, Ripper.maxLen, Ripper.stepSize);
            s = Ripper.shapeletFinder.findShapelet();
            int posCount = 0, negCount = 0;
            for (Double d : s.getHistMap().keySet()) {
                if (d < s.getSplitDist()) {
                    for (TimeSeries it : s.getHistMap().get(d)) {
                        if (it.getLabel() == tLabel) {
                            negCount++;
                        } else {
                            posCount++;
                        }
                    }
                } else {
                    break;
                }
            }
            int comparator = (posCount > negCount) ? -1 : 1; 
            Antecedent a = new Antecedent(s, comparator);
            r.addClause(a);
            r.setConsequent(growPos.get(0).getLabel());
        }
        
        return r;
    }
    
    private double checkInsts(Rule r) {
        TimeSeries t;
        int correctPos = 0, correctNeg = 0;
        double error = 0;
        int i;
        for (i = 0; i < prunePos.size(); i++) {
            t = prunePos.get(i);
            if (r.eval(t) == t.getLabel()) {
                correctPos++;
            }
        }
        for (i = 0; i < pruneNeg.size(); i++) {
            t = pruneNeg.get(i);
            if (r.eval(t) == Integer.MIN_VALUE) {
                correctNeg++;
            }
        }
        error = (double)(prunePos.size() - correctPos) / prunePos.size()
                + (double)(pruneNeg.size() - correctNeg) / pruneNeg.size();
        
        return error;
    }
    
    private void split(TimeSeriesDataset pos, TimeSeriesDataset neg) {
        int i;
        if (posSize != pos.size()) {
            posSize = pos.size();
            growPosSize = (int) Math.ceil(posSize * 0.8);
            if (posList == null) {
                posList = new ArrayList<Integer>(posSize);
            } else {
                posList.clear();
            }
            for (i = 0; i < posSize; i++) {
                posList.add(i);
            }
        }
        if (negSize != neg.size()) {
            negSize = neg.size();
            growNegSize = (int) Math.ceil(negSize * 0.8);
            if (negList == null) {
                negList = new ArrayList<Integer>(negSize);
            } else {
                negList.clear();
            }
            for (i = 0; i < negSize; i++) {
                negList.add(i);
            }
        }
        Collections.shuffle(posList);
        Collections.shuffle(negList);
        
        growPos = new TimeSeriesDataset();
        prunePos = new TimeSeriesDataset();
        growNeg = new TimeSeriesDataset();
        pruneNeg = new TimeSeriesDataset();
        for (i = 0; i < growPosSize; i++) {
            growPos.add(pos.get(posList.get(i)));
        }
        for (i = growPosSize; i < posSize; i++) {
            prunePos.add(pos.get(posList.get(i)));
        }
        for (i = 0; i < growNegSize; i++) {
            growNeg.add(neg.get(negList.get(i)));
        }
        for (i = growNegSize; i < negSize; i++) {
            pruneNeg.add(neg.get(negList.get(i)));
        }
    }
    
    public int checkInstance(TimeSeries t) {
        int result = Integer.MIN_VALUE;
        for (Rule r : this.ruleSet) {
            result = r.eval(t);
            if (result != Integer.MIN_VALUE) {
                break;
            }
        }
        return result;
    }
    
    public void printRules() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < this.ruleSet.size(); i++) {
            if (this.ruleSet.get(i).clauses.size() != 0) {
                sb.append("IF ");
                for (int j = 0; j < this.ruleSet.get(i).clauses.size(); j++) {
                    sb.append("[dist(t,r" + i + "s" + j + ")");
                    sb.append((this.ruleSet.get(i).clauses.get(j).comp==1)?" > ":" < ");
                    sb.append(this.ruleSet.get(i).clauses.get(j).sh.getSplitDist() + "]");
                    if (j < this.ruleSet.get(i).clauses.size()-1) {
                        sb.append(" AND ");
                    }
                }
                sb.append(" then CLASS = " + this.ruleSet.get(i).result + "\n");
            } else {
                sb.append("ELSE CLASS = " + this.ruleSet.get(i).result + "\n");
            }
        }
        System.out.print(sb.toString());
    }
    
//    private Rule pruneRule(Rule r) {
//        //TODO: prune rule with prunePos and pruneNeg
//        return r;
//    }
    
    private class Rule {
        ArrayList<Antecedent> clauses;
        int result;
        
        public Rule() {
            clauses = new ArrayList<Antecedent>();
            result = Integer.MIN_VALUE;
        }
        
        void addClause(Antecedent a) {
            this.clauses.add(a);
        }
        
        void setConsequent(int resultingClass) {
            result = resultingClass;
        }
        
        int eval(TimeSeries t) {
            boolean allPassed = true;
            int i = 0;
            while (allPassed && i < clauses.size()) {
                allPassed &= clauses.get(i).eval(t);
                i++;
            }
            if (allPassed) {
                return result;
            } else {
                return Integer.MIN_VALUE;
            }
        }
    }
    
    private class Antecedent {
        Shapelet sh;
        int comp; // -1 lessThan; 0 equal; 1 greaterThan
        
        public Antecedent(Shapelet sh) {
            this.sh = sh;
            this.comp = -1;
        }
        
        public Antecedent(Shapelet sh, int comp) {
            this(sh);
            this.comp = comp;
        }
        
        boolean eval(TimeSeries t) {
            double sDist = shapeletFinder.getDist(t, sh.getShapelet());
            if (this.comp == -1) {
                return sDist < sh.getSplitDist();
            } else {
                return sDist >= sh.getSplitDist();
            }
        }
    }
    
    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });
        
        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }
}
