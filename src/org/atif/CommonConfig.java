package org.atif;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.HashMap;
import java.util.Properties;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.kramerlab.timeseries.TimeSeries;

public class CommonConfig {
    // The following static String variables are the CLI switches
    protected static String datasetSw = "dn";
    protected static String datasetSwL = "dataset-name";
    
    protected static String dataPathSw = "dd";
    protected static String dataPathSwL = "data-dir";
    
    protected static String rsltPathSw = "rd";
    protected static String rsltPathSwL = "results-dir";
    
    protected static String paramsPathSw = "pd";
    protected static String paramsPathSwL = "params-dir";
    
    protected static String ensembleSizeSw = "es";
    protected static String ensembleSizeSwL = "ensemble-size";
    
    protected CommandLine cmdLine;
    protected Options options;
    
    protected ArrayList<TimeSeries> trainSet;
    protected ArrayList<TimeSeries> testSet;
    
    protected int minLen;
    protected int maxLen;
    protected int stepSize;
    protected int leafSize;
    protected int treeDepth;
    
    public CommonConfig(String[] args) {
        this.constructCommandLine(args);
        if (this.cmdLine == null) {
            this.printHelp(true);
            System.exit(1);
        } else {
            System.out.println("Data set: " + this.getDataSetName());
            this.trainSet = this.loadDataset(this.getDataSetName() + "_TRAIN");
            this.testSet  = this.loadDataset(this.getDataSetName() + "_TEST");
            
            Properties props = new Properties();
            File propsFile;
            Path filePath;
            try {
                filePath = Paths.get(this.getParamsPath(), "default.params");
                propsFile = new File(filePath.toUri());
                if (propsFile.exists()) {
                    props.load(new FileInputStream(propsFile));
                }
                filePath = Paths.get(this.getParamsPath(), this.getDataSetName() + ".params");
                propsFile = new File(filePath.toUri());
                if (propsFile.exists()) {
                    props.load(new FileInputStream(propsFile));
                }
            } catch (Exception e) {
                System.err.println("Error opening properties file: " + e);
            }
            
            int tsLen = this.trainSet.get(0).size();
            this.minLen = (int) (tsLen / 4.0);
            this.maxLen = (int) (tsLen * 2 / 3.0);
            this.stepSize = 1;
            this.leafSize = 1;
            this.treeDepth = Integer.MAX_VALUE;
            
            if (props.containsKey("minLen")) {
                this.minLen = Integer.parseInt(props.getProperty("minLen"));
            }
            if (props.containsKey("maxLen")) {
                this.maxLen = Integer.parseInt(props.getProperty("maxLen"));
            }
            if (props.containsKey("stepSize")) {
                this.stepSize = Integer.parseInt(props.getProperty("stepSize"));
            }
            if (props.containsKey("leafSize")) {
                this.leafSize = Integer.parseInt(props.getProperty("leafSize"));
                if (this.leafSize == 0) {
                    HashMap<Integer, Integer> classMap = new HashMap<>();
                    for (TimeSeries ts : trainSet) {
                        classMap.put(ts.getLabel(), classMap.getOrDefault(ts.getLabel(), 0)+1);
                    }
                    this.leafSize = classMap.entrySet()
                                            .stream()
                                            .min((x, y) -> x.getValue() < y.getValue() ? -1 : 1)
                                            .get()
                                            .getValue();
                }
            }
            if (props.containsKey("treeDepth")) {
                this.treeDepth = Integer.parseInt(props.getProperty("treeDepth"));
            }
        }
    }
    
    protected void constructCommandLine(String[] args) {
        this.options = new Options();
        this.options.addOption(Option.builder(datasetSw)
                                     .longOpt(datasetSwL)
                                     .argName("FILE")
                                     .required()
                                     .desc("Data set name to be evaluated")
                                     .numberOfArgs(1)
                                     .build())
                    .addOption(Option.builder(dataPathSw)
                                     .longOpt(dataPathSwL)
                                     .argName("PATH/TO/DATA/DIRECTORY")
                                     .desc("Absolute or relative path to the directory with the data set file")
                                     .numberOfArgs(1)
                                     .build())
                    .addOption(Option.builder(rsltPathSw)
                                     .longOpt(rsltPathSwL)
                                     .argName("PATH/TO/RESULTS/DIRECTORY")
                                     .desc("Absolute or relative path to the directory where results will be saved"
                                           + "\nDefault: 'results' directory in the program directory")
                                     .numberOfArgs(1)
                                     .build())
                    .addOption(Option.builder(paramsPathSw)
                                     .longOpt(paramsPathSwL)
                                     .argName("PATH-TO-PARAMS-FOLDER")
                                     .desc("Absolute or relative path to the parameter files directory"
                                           + "\nDefault: 'params' directory in the program directory")
                                     .numberOfArgs(1)
                                     .build())
                    .addOption(Option.builder(ensembleSizeSw)
                                     .longOpt(ensembleSizeSwL)
                                     .argName("ENSEMBLE SIZE")
                                     .desc("The ensemble size\nDefault: 10 members")
                                     .numberOfArgs(1)
                                     .build());
        
        try {
            CommandLineParser cliParser = new DefaultParser();
            
            this.cmdLine = cliParser.parse(options, args);
        } catch (ParseException e) {
            System.err.println("Error parsing command line arguments\nInvalid or no arguments provided");
        }
    }
    
    protected ArrayList<TimeSeries> loadDataset(String fileName) {
        ArrayList<TimeSeries> dataset = null;
        TimeSeries currTS;
        ArrayList<Double> ts;
        int tsClass;
        
        String currLine, delimiter = " ";
        StringTokenizer st;
        try (BufferedReader br = Files.newBufferedReader(Paths.get(this.getDataPath(), fileName))) {
            dataset = new ArrayList<TimeSeries>();
            while ((currLine = br.readLine()) != null) {
                if (currLine.matches("\\s*")) {
                    continue;
                } else {
                    currLine = currLine.trim();
                    st = new StringTokenizer(currLine, String.valueOf(delimiter));
                    tsClass = (int) Double.parseDouble(st.nextToken());
                    ts = new ArrayList<Double>();
                    while (st.hasMoreTokens()) {
                        ts.add(Double.parseDouble(st.nextToken()));
                    }
                    currTS = new TimeSeries(ts, tsClass);
                    dataset.add(currTS);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading the data set file: " + e);
        }
        return dataset;
    }
    
    public String getDataSetName() {
        return this.cmdLine.getOptionValue(datasetSw);
    }
    
    protected String getDataPath() {
        return this.cmdLine.getOptionValue(dataPathSw, "../../data/UCR_2015");
    }
    
    protected String getResultsPath() {
        return this.cmdLine.getOptionValue(rsltPathSw, "results");
    }
    
    public String getParamsPath() {
        return this.cmdLine.getOptionValue(paramsPathSw, "params");
    }
    
    public int getEnsembleSize() {
        return Integer.parseInt(this.cmdLine.getOptionValue(ensembleSizeSw, "10"));
    }
    
    public ArrayList<TimeSeries> getTrainSet() {
        return this.trainSet;
    }
    
    public ArrayList<TimeSeries> getTestSet() {
        return this.testSet;
    }
    
    public int getMinLen() {
        return this.minLen;
    }
    
    public int getMaxLen() {
        return this.maxLen;
    }
    
    public int getStepSize() {
        return this.stepSize;
    }
    
    public int getLeafSize() {
        return this.leafSize;
    }
    
    public int getTreeDepth() {
        return this.treeDepth;
    }
    
    public void saveResults(String fileName, double trainingTime, double testingTime, double trainingAccuracy,
                            double testingAccuracy) {
        this.saveResults(fileName, trainingTime, testingTime, trainingAccuracy, testingAccuracy, false, 0);
    }
    
    public void saveResults(String fileName, double trainingTime, double testingTime, double trainingAccuracy,
                            double testingAccuracy, int enSize) {
        this.saveResults(fileName, trainingTime, testingTime, trainingAccuracy, testingAccuracy, true, enSize);
    }
    
    protected void saveResults(String fileName, double trainingTime, double testingTime, double trainingAccuracy,
                               double testingAccuracy, boolean isEnsemble, int enSize) {
        try {
            Files.createDirectories(Paths.get(this.getResultsPath()));
            File resultsFile = new File(Paths.get(this.getResultsPath(), fileName).toString());
            Formatter formatter = new Formatter();
            if (!resultsFile.exists()) {
                formatter.format("%s",
                                 "Dataset,TrainingTime,TestingTime,TrainingAccuacy,TestingAccuracy,TrainSize,TestSize,TSLen,MinLen,MaxLen");
                if (isEnsemble) {
                    formatter.format(",%s", "EnsembleSize");
                }
                formatter.format("\n");
            }
            try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(resultsFile.getAbsolutePath()),
                                                             StandardOpenOption.CREATE,
                                                             StandardOpenOption.APPEND)) {
                formatter.format("%s,%.2f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d",
                                 this.getDataSetName(),
                                 trainingTime,
                                 testingTime,
                                 trainingAccuracy,
                                 testingAccuracy,
                                 this.trainSet.size(),
                                 this.testSet.size(),
                                 this.trainSet.get(0).size(),
                                 this.getMinLen(),
                                 this.getMaxLen());
                if (isEnsemble) {
                    formatter.format(",%d", enSize);
                }
                formatter.format("\n");
                bw.write(formatter.toString());
            }
            System.out.println(formatter.toString());
            formatter.close();
        } catch (IOException e) {
            System.err.println("Error saving results: " + e);
        }
    }
    
    /**
     * Print help to provided OutputStream.
     * 
     * @param options Command-line options to be part of usage.
     * @param detailed
     * @param out OutputStream to which to write the usage information.
     */
    public void printHelp(boolean detailed) {
        String cliSyntax = "java ";
        PrintWriter writer = new PrintWriter(System.out);
        HelpFormatter helpFormatter = new HelpFormatter();
        if (detailed) {
            helpFormatter.printHelp(writer, 120, cliSyntax, "", options, 7, 1, "", true);
        } else {
            helpFormatter.printUsage(writer, 120, cliSyntax, options);
        }
        writer.flush();
    }
}
