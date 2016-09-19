package org.atif.launcher;

import java.util.ArrayList;

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
        double trainingTime;
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
        trainingTime = (stop - start) / 1e3;
        
        cc.saveResults(dtList, trainSet, testSet, trainingTime);
    }
}
