package org.atif.launcher;

import java.util.ArrayList;
import java.util.Random;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsembleWithBoostingBySampling {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args, "RS Ensemble - Boosting by Sampling.csv");
        int method = 2;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet()),
                          trainSetResampled = new TimeSeriesDataset(cc.getTrainSet());
        long start, stop;
        double trainingTime;
        DecisionTree tree;
//        long totalCandidates = 0, prunedCandidates = 0;
        ArrayList<DecisionTree> dtList = new ArrayList<>();
        ArrayList<Integer> incorrectlyClassified = new ArrayList<>();
        ArrayList<Double> weights = new ArrayList<>();
        int predClass = Integer.MIN_VALUE;
        Random rng = new Random();
        double error;
        double temp = 1.0 / trainSet.size();
        for (int i = 0; i < trainSet.size(); i++) {
            weights.add(temp);
        }
        
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
                if (predClass != trainSetResampled.get(j).getLabel()) {
                    error += weights.get(j);
                    incorrectlyClassified.add(j);
                }
            }
            // totalCandidates += tree.getTotalCandidates();
            // prunedCandidates += tree.getPrunedCandidates();
            dtList.add(i, tree);
//            if (error < 1e-3) {
//                break;
//            }
            temp = (2 * error);
            for (Integer ind : incorrectlyClassified) {
                weights.set(ind, weights.get(ind) / temp);
            }
            normalize(weights);
            // resample the dataset using new weights
            trainSetResampled = resampleWithWeights(rng, trainSetResampled, weights);
        }
        stop = System.currentTimeMillis();
        trainingTime = (stop - start) / 1e3;
        
        cc.saveResults(dtList, trainSet, testSet, trainingTime);
    }
    
    protected static TimeSeriesDataset resampleWithWeights(Random rng, TimeSeriesDataset src, ArrayList<Double> weights) {
        TimeSeriesDataset newSet = new TimeSeriesDataset();
        double[] probabilities = new double[src.size()];
        double sumProbs = 0, sumWeights = 0;
        
        for (Double weight : weights) {
            sumWeights += weight;
        }
        
        for (int i = 0; i < src.size(); i++) {
            sumProbs += rng.nextDouble();
            probabilities[i] = sumProbs;
        }
        normalize(probabilities, sumProbs / sumWeights);
        probabilities[src.size() - 1] = sumWeights;
        
        int k = 0;
        int l = 0;
        sumProbs = 0;
        while ((k < src.size() && (l < src.size()))) {
            if (weights.get(l) < 0) {
                throw new IllegalArgumentException("Weights have to be positive.");
            }
            sumProbs += weights.get(l);
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
    
    protected static void normalize(ArrayList<Double> weights) {
        double sum = weights.stream().mapToDouble(val -> val).sum();
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i)/sum);
        }
    }
}
