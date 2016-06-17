package org.atif.launcher;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsembleWithBoosting {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args);
        int method = 2;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet()),
                          trainSetResampled = new TimeSeriesDataset(cc.getTrainSet());
        long start, stop;
        double trainingTime, testingTime;
        double trainingAccuracy, testingAccuracy;
        DecisionTree tree;
//        long totalCandidates = 0, prunedCandidates = 0;
        ArrayList<DecisionTree> dtList = new ArrayList<>();
        ArrayList<Integer> correctlyClassified = new ArrayList<>();
        ArrayList<Integer> incorrectlyClassified = new ArrayList<>();
        ArrayList<Double> alphaList = new ArrayList<>();
        int predClass = Integer.MIN_VALUE;
        Random rng = new Random();
        double[] weights = new double[trainSet.size()];
        Arrays.fill(weights, 1.0 / trainSet.size());
        double error, temp;
        
        start = System.currentTimeMillis();
        for (int i = 0; i < cc.getEnsembleSize(); i++) {
            tree = new DecisionTree.Builder(trainSetResampled, method)
                                   .minLen(cc.getMinLen())
                                   .maxLen(cc.getMaxLen())
                                   .stepSize(cc.getStepSize())
                                   .leafeSize(cc.getLeafSize())
                                   .treeDepth(cc.getTreeDepth())
                                   .build();
            error = 0.0;
            for (int j = 0; j < trainSetResampled.size(); j++) {
                predClass = tree.checkInstance(trainSetResampled.get(j));
                if (predClass == trainSetResampled.get(j).getLabel()) {
                    correctlyClassified.add(j);
                } else {
                    error += weights[j];
                    incorrectlyClassified.add(j);
                }
            }
            // totalCandidates += tree.getTotalCandidates();
            // prunedCandidates += tree.getPrunedCandidates();
            dtList.add(i, tree);
            if (error >= 0.5 || error < 1e-3) {
                if (i == 0) {
                    alphaList.add(i, 1.0);
                }
                break;
            }
            alphaList.add(i, 0.5 * Math.log((1.0 - error) / error));
            temp = (2 * error);
            for (Integer ind : incorrectlyClassified) {
                weights[ind] = weights[ind] / temp;
            }
            temp = (2 * (1 - error));
            for (Integer ind : correctlyClassified) {
                weights[ind] = weights[ind] / temp;
            }
            // resample the dataset using new weights
            trainSetResampled = resampleWithWeights(rng, trainSetResampled, weights);
        }
        stop = System.currentTimeMillis();
        trainingAccuracy = getSplitAccuracy(dtList, alphaList, trainSet);
        trainingTime = (stop - start) / 1e3;
        
        start = System.currentTimeMillis();
        testingAccuracy = getSplitAccuracy(dtList, alphaList, testSet);
        stop = System.currentTimeMillis();
        testingTime = (stop - start) / 1e3;
        
        cc.saveResults("RS Ensemble - Boosting.csv", trainingTime, testingTime, trainingAccuracy, testingAccuracy,
                       dtList.size());
    }
    
    public static double getSplitAccuracy(ArrayList<DecisionTree> dtList, ArrayList<Double> alphaList, TimeSeriesDataset split) {
        int correct = 0, predClass = Integer.MIN_VALUE;
        HashMap<Integer, Integer> classMap = split.getClassHist();
        int min = classMap.keySet().stream().min((x, y) -> x.compareTo(y)).get();
        int max = classMap.keySet().stream().max((x, y) -> x.compareTo(y)).get();
        double collectiveDecision = 0;
        for (int ind = 0; ind < split.size(); ind++) {
            collectiveDecision = 0;
            for (int j = 0; j < alphaList.size(); j++) {
                predClass = dtList.get(j).checkInstance(split.get(ind));
                predClass = -1 + 2 * (predClass - min) / (max - min);
                collectiveDecision += alphaList.get(j) * predClass;
            }
            predClass = (int) Math.signum(collectiveDecision);
            if (-1 + 2 * (split.get(ind).getLabel() - min) / (max - min) == predClass) {
                correct++;
            }
        }
        return 100.0 * correct / split.size();
    }
    
    protected static TimeSeriesDataset resampleWithWeights(Random rng, TimeSeriesDataset src, double[] weights) {
        TimeSeriesDataset newSet = new TimeSeriesDataset();
        double[] probabilities = new double[src.size()];
        double sumProbs = 0, sumOfWeights = sumOfWeights(weights);
        
        for (int i = 0; i < src.size(); i++) {
            sumProbs += rng.nextDouble();
            probabilities[i] = sumProbs;
        }
        normalize(probabilities, sumProbs / sumOfWeights);
        probabilities[src.size() - 1] = sumOfWeights;
        
        int k = 0;
        int l = 0;
        sumProbs = 0;
        while ((k < src.size() && (l < src.size()))) {
            if (weights[l] < 0) {
                throw new IllegalArgumentException("Weights have to be positive.");
            }
            sumProbs += weights[l];
            while ((k < src.size()) && (probabilities[k] <= sumProbs)) {
                newSet.add(src.get(l));
                // System.out.println(k + " " + l);
                k++;
            }
            l++;
        }
        
        return newSet;
    }
    
    protected static void normalize(double[] doubles, double sum) {
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] /= sum;
        }
    }
    
    protected static double sumOfWeights(double[] weights) {
        double total = 0.0;
        for (Double d : weights) {
            total += d;
        }
        return total;
    }
}
