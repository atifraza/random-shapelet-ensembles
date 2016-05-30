package org.atif.launcher;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Properties;
import java.util.Random;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeries;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsembleWithBoosting {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args);
        if (args.length == 0) {
            cc.printHelp(true);
        } else {
            String dsName = cc.getDataSetName();
            int method = 2;
            String dataPath = cc.getDataPath();
            String resultsPath = Paths.get(cc.getResultsPath(), "RS Ensemble - Boosting.csv").toString();
            System.out.println("Data set: " + dsName);
            System.out.println("Method: " + method);
            System.out.println();
            ArrayList<TimeSeries> dataset;
            TimeSeriesDataset trainSet, testSet, trainSetResampled;
            int minLen, maxLen, stepSize;
            long start, stop;
            DecisionTree tree;
            int correct = 0;
            int predClass = Integer.MIN_VALUE;
            long totalCandidates = 0, prunedCandidates = 0;
            int ensembleSize = 10;
            Random rng = new Random();
            ArrayList<DecisionTree> dtList = new ArrayList<>();
            ArrayList<Double> alphaList = new ArrayList<>();
            ArrayList<Double> errorList = new ArrayList<>();
            ArrayList<Integer> correctlyClassified = new ArrayList<>();
            ArrayList<Integer> incorrectlyClassified = new ArrayList<>();
            double[] weights;
            double error;
            
            dataset = cc.loadDataset(Paths.get(dataPath, dsName + "_TRAIN"), " ");
            trainSet = new TimeSeriesDataset(dataset);
            
            Properties props = cc.constructPropertiesObject(trainSet.get(0).size());
            minLen = Integer.parseInt(props.getProperty("minLen"));
            maxLen = Integer.parseInt(props.getProperty("maxLen"));
            stepSize = Integer.parseInt(props.getProperty("stepSize"));
            
            trainSetResampled = new TimeSeriesDataset(dataset);
            weights = new double[trainSetResampled.size()];
            Arrays.fill(weights, 1.0/trainSetResampled.size());
            start = System.currentTimeMillis();
            for (int i = 0; i<ensembleSize; i++) {
                tree = new DecisionTree(trainSetResampled, minLen, maxLen, stepSize, method);
                error = 0.0;
                correct = 0;
                for (int j = 0; j < trainSetResampled.size(); j++) {
                    predClass = tree.checkInstance(trainSetResampled.get(j));
                    if (predClass == trainSetResampled.get(j).getLabel()) {
                        correct++;
                        correctlyClassified.add(j);
                    } else {
                        error += weights[j];
                        incorrectlyClassified.add(j);
                    }
                }
//                currError = (double) (trainSetResampled.size() - correct) /trainSetResampled.size();
                totalCandidates += tree.getTotalCandidates();
                prunedCandidates += tree.getPrunedCandidates();
                dtList.add(i, tree);
                errorList.add(i, error);
                if (error >= 0.5 || error < 1e-3) {
                    ensembleSize = i;
                    if (i==0) {
                        ensembleSize = 1;
                        alphaList.add(i, 1.0);
                    }
                    break;
                }
                alphaList.add(i, 0.5 * Math.log((1.0 - error)/error));
                double temp = (2*error);
                for (Integer ind : incorrectlyClassified) {
                    weights[ind] = weights[ind]/temp;
                }
                temp = (2*(1-error));
                for (Integer ind : correctlyClassified) {
                    weights[ind] = weights[ind]/temp;
                }
                // resample the dataset using new weights
                trainSetResampled = resampleWithWeights(rng, trainSetResampled, weights);
            }
            stop = System.currentTimeMillis();
            
            dataset = cc.loadDataset(Paths.get(dataPath, dsName + "_TEST"), " ");
            testSet = new TimeSeriesDataset(dataset);
            correct = 0;
            Map<Integer, Integer> classMap = trainSet.getClassHist();
            int min = classMap.keySet().stream().min((x, y) -> x.compareTo(y)).get();
            int max = classMap.keySet().stream().max((x, y) -> x.compareTo(y)).get();
            double collectiveDecision = 0;
            for (int i = 0; i < testSet.size(); i++) {
                collectiveDecision = 0;
                for (int j = 0; j < ensembleSize; j++) {
                    tree = dtList.get(j);
                    predClass = tree.checkInstance(testSet.get(i));
                    predClass = -1+2*(predClass - min)/(max - min);
                    collectiveDecision += alphaList.get(j) * predClass;
                }
                predClass = (int) Math.signum(collectiveDecision);
                if (-1+2*(testSet.get(i).getLabel() - min)/(max - min) == predClass) {
                    correct++;
                }
            }
            
            File resultsFile = new File(resultsPath);
            StringBuilder strBldr = new StringBuilder();
            if (!resultsFile.exists()) {
                strBldr.append("Dataset,TrainSize,TSLen,MinLen,MaxLen,Method,TotalCand,Pruned,Time (sec),Accuracy (%),EnsembleSize\n");
            }
            try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(resultsFile.getAbsolutePath()),
                                                             StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                strBldr.append(dsName + "," + trainSet.size() + "," + trainSet.get(0).size() + "," + minLen + "," + maxLen
                        + "," + method + "," + totalCandidates + "," + prunedCandidates + ","
                        + (stop - start) / 1e3 + "," + 100.0 * correct / testSet.size() + "," + ensembleSize + "\n");
                bw.write(strBldr.toString());
                System.out.println(strBldr.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    
    protected static TimeSeriesDataset resampleWithWeights(Random rng, TimeSeriesDataset src, double[] weights) {
        TimeSeriesDataset newSet = new TimeSeriesDataset();
        double[] probabilities = new double[src.size()];
        double sumProbs = 0, sumOfWeights = sumOfWeights(weights);
        
        for (int i = 0; i < src.size(); i++) {
            sumProbs += rng.nextDouble();
            probabilities[i] = sumProbs;
        }
        normalize(probabilities, sumProbs/sumOfWeights);
        probabilities[src.size()-1] = sumOfWeights;
        
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
//                System.out.println(k + " " + l);
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
