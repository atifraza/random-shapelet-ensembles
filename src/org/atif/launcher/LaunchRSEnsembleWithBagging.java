package org.atif.launcher;

import java.util.ArrayList;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsembleWithBagging {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args, "RS Ensemble - Bagging.csv");
        int method = 2;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet()),
                          trainSetBagged;
        long start, stop;
        double trainingTime;
        DecisionTree tree;
//        long totalCandidates = 0, prunedCandidates = 0;
        ArrayList<DecisionTree> dtList = new ArrayList<>();
        Random rng = new Random();
        IntStream randIntStream;
        ArrayList<Integer> randIndices;
        
        start = System.currentTimeMillis();
        for (int i = 0; i < cc.getEnsembleSize(); i++) {
            trainSetBagged = new TimeSeriesDataset();
            randIntStream = rng.ints(trainSet.size(), 0, trainSet.size());
            randIndices = (ArrayList<Integer>)randIntStream.boxed().collect(Collectors.toList());
            for (Integer ind : randIndices) {
                trainSetBagged.add(trainSet.get(ind));
            }
            tree = new DecisionTree.Builder(trainSetBagged, method)
                                   .minLen(cc.getMinLen())
                                   .maxLen(cc.getMaxLen())
                                   .stepSize(cc.getStepSize())
                                   .leafeSize(cc.getLeafSize())
                                   .treeDepth(cc.getTreeDepth())
                                   .build();
//             totalCandidates += tree.getTotalCandidates();
//             prunedCandidates += tree.getPrunedCandidates();
            dtList.add(tree);
        }
        stop = System.currentTimeMillis();
        trainingTime = (stop - start) / 1e3;
        
        cc.saveResults(dtList, trainSet, testSet, trainingTime);
    }
}
