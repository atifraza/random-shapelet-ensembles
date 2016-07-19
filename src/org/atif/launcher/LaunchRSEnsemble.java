package org.atif.launcher;

import java.util.ArrayList;
import java.util.HashMap;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsemble {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args, "RS Ensemble.csv");
        int method = 2;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet());
        long start, stop;
        double trainingTime, testingTime;
        double trainingAccuracy, testingAccuracy;
        DecisionTree tree;
//        long totalCandidates = 0, prunedCandidates = 0;
        ArrayList<DecisionTree> dtList = new ArrayList<>();
        
        start = System.currentTimeMillis();
        for (int i = 0; i<cc.getEnsembleSize(); i++) {
            tree = new DecisionTree.Builder(trainSet, method)
                                   .minLen(cc.getMinLen())
                                   .maxLen(cc.getMaxLen())
                                   .stepSize(cc.getStepSize())
                                   .leafeSize(cc.getLeafSize())
                                   .treeDepth(cc.getTreeDepth())
                                   .build();
//            totalCandidates += tree.getTotalCandidates();
//            prunedCandidates += tree.getPrunedCandidates();
            dtList.add(tree);
        }
        stop = System.currentTimeMillis();
        trainingAccuracy = getSplitAccuracy(dtList, trainSet);
        trainingTime = (stop - start) / 1e3;
        
        start = System.currentTimeMillis();
        testingAccuracy = getSplitAccuracy(dtList, testSet);
        stop = System.currentTimeMillis();
        testingTime = (stop - start) / 1e3;
        
        cc.saveResults(trainingTime, testingTime, trainingAccuracy, testingAccuracy, dtList.size());
    }
    
    public static double getSplitAccuracy(ArrayList<DecisionTree> dtList, TimeSeriesDataset split) {
        int predClass, correct = 0, majorityVote;
        HashMap<Integer, Integer> predClassCount = new HashMap<>();
        for (int ind = 0; ind < split.size(); ind++) {
            predClassCount.clear();
            for (int j = 0; j < dtList.size(); j++) {
                predClass = dtList.get(j).checkInstance(split.get(ind));
                predClassCount.put(predClass, predClassCount.getOrDefault(predClass, 0) + 1);
            }
            majorityVote = predClassCount.entrySet().stream()
                                         .max((e1, e2) -> ((e1.getValue() > e2.getValue()) ? 1 : -1)).get().getKey();
            if (majorityVote == split.get(ind).getLabel()) {
                correct++;
            }
        }
        return 100.0 * correct / split.size();
    }
}
