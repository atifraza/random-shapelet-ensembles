import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.*;
import org.kramerlab.shapelets.LegacyShapelets;
import org.kramerlab.shapelets.Shapelet;
import org.kramerlab.rulelearners.Ripper;

public class StartTests {
    // The following static String variables are the CLI switches
    protected static String fileSw = "f";
    protected static String fileSwL = "file";
    protected static String methodSw = "m";
    protected static String methodSwL = "method";
    protected static String helpSw = "h";
    protected static String helpSwL = "help";
    
    /**
     * @param args
     */
    public static void main(String[] args) {
        String dsName = null;       // Time series data set name
        String path = "./data/";
        FileWriter fw;
        BufferedWriter bw;
        Properties props;
        
        ArrayList<TimeSeries> dataset;
        TimeSeriesDataset trainSet, testSet;
        
        int minLen = 5, maxLen = 20, stepSize = 1;
        int method = 1;
        
        Options options = null;
        
        try {
            // Constructs CLI options and corresponding help text
            options = constructCLIOptions();
            
            // Create a DefaultParser object
            CommandLineParser cliParser = new DefaultParser();
            
            // Parse provided CLI arguments according to available
            // options
            CommandLine cmdLine = cliParser.parse(options, args);
            
            if (cmdLine.getOptions().length == 0 || cmdLine.hasOption(helpSw)) {
                // No CLI argument provided OR Help required
                printHelp(options, true, System.out);   // Print help
                System.exit(0);                         // and exit
            }
            
            // Get the values provided with CLI arguments
            dsName = cmdLine.getOptionValue(fileSw);
            
            if (cmdLine.hasOption(methodSw)) {
                method = Integer.parseInt(cmdLine.getOptionValue(methodSw));
            }
            
            System.out.println("Dataset: " + dsName + "\nMethod: " + method);
            if (method == 0) {
                dataset = loadDataset(path + dsName + "_TRAIN", " ");
                minLen = dataset.get(0).size()/4;
                maxLen = dataset.get(0).size();
                
                ArrayList<Integer> lens = new ArrayList<Integer>();
                Shapelet currShapelet;
                LegacyShapelets shapeletFinder;
                boolean added;
                
                for (int i = 0; i < 100; i++) {
                    Collections.shuffle(dataset);
                    
                    trainSet = new TimeSeriesDataset();
                    
                    for (int j = 0; j < dataset.size()/4; j++) {  // j < dataset.size()/4
                        trainSet.add(dataset.get(j));
                    }
                    
                    shapeletFinder = new LegacyShapelets(trainSet, minLen,
                                                         maxLen, stepSize);
                    shapeletFinder.toggleEntropyPruning(false);
                    shapeletFinder.setInversedSearch();
                    
                    currShapelet = shapeletFinder.findShapelet();
                    
                    added = false;
                    for (int j = 0; j < lens.size(); j++) {
                        if (lens.get(j) <= currShapelet.getShapelet().size()) {
                            continue;
                        } else {
                            lens.add(j, currShapelet.getShapelet().size());
                            added = true;
                            break;
                        }
                    }
                    if (!added) {
                        lens.add(currShapelet.getShapelet().size());
                    }
                }
                
                minLen = lens.get(19);  // 20th entry
                maxLen = lens.get(89);  // 90th entry
                
                fw = new FileWriter("./params/" + dsName + ".params");
                bw = new BufferedWriter(fw);
                props = new Properties();
                props.put("minLen", Integer.toString(minLen));
                props.put("maxLen", Integer.toString(maxLen));
                props.store(bw, "");
            } else {
                dataset = loadDataset(path + dsName + "_TRAIN", " ");
                trainSet = new TimeSeriesDataset(dataset);
                
                dataset = loadDataset(path + dsName + "_TEST", " ");
                testSet = new TimeSeriesDataset(dataset);
                
                int correct = 0;
                int predClass = Integer.MIN_VALUE;
                long start, stop;
                
                props = new Properties();
                File propsFile = new File("./params/" + dsName + ".params");
                if (propsFile.exists()) {
                    props.load(new FileInputStream(propsFile));
                } else {
                    props.load(new FileInputStream("./params/default.params"));
                }
                
                minLen = Integer.parseInt(props.getProperty("minLen", "10"));
                maxLen = Integer.parseInt(props.getProperty("maxLen", "20"));
                stepSize = Integer.parseInt(props.getProperty("stepSize", "1"));

                start = System.currentTimeMillis();
                DecisionTree tree = new DecisionTree(trainSet, minLen, maxLen,
                                                     stepSize, method);
                for (int ind = 0; ind < testSet.size(); ind++) {
                    predClass = tree.checkInstance(testSet.get(ind));
                    if (predClass == testSet.get(ind).getLabel()) {
                        correct++;
                    }
                }
                
                stop = System.currentTimeMillis();
                System.out.println("Time (sec): " + (stop - start) / 1e3);
                System.out.println("Accuracy: " + 100.0 * correct / testSet.size());
                File resultsFile = new File("./results/Shapelets_results.csv");
                String temp = "";
                if (!resultsFile.exists()) {
                    temp = "Dataset," + "Train Size," + "TS Len," + "MinLen,"
                           + "MaxLen," + "Method," + "C_Total," + "C_Processed,"
                           + "Time (sec)," + "Accuracy (%)\n";
                }
                temp += dsName + ",\t" + trainSet.size() + ",\t"
                        + trainSet.get(0).size() + ",\t" + minLen + ",\t"
                        + maxLen + ",\t" + method + ",\t"
                        + tree.getTotalCandidates() + ",\t"
                        + tree.getPrunedCandidates() + ",\t"
                        + (stop - start) / 1e3 + ",\t"
                        + 100.0 * correct / testSet.size() + "\n";
                correct = 0;
                start = System.currentTimeMillis();
                Ripper test = new Ripper(trainSet, minLen, maxLen, stepSize);
                test.printRules();
                for (int ind = 0; ind < testSet.size(); ind++) {
                    predClass = test.checkInstance(testSet.get(ind));
                    if (predClass == testSet.get(ind).getLabel()) {
                        correct++;
                    }
                }
                stop = System.currentTimeMillis();
                System.out.println("Time (sec): " + (stop - start) / 1e3);
                System.out.println("Accuracy: " + 100.0 * correct / testSet.size());
                fw = new FileWriter(resultsFile, true);
                bw = new BufferedWriter(fw);
                bw.write(temp);
                bw.close();
                resultsFile = null;
            }
        } catch (ParseException parseException) {
            // Catch any exceptions here and print the help
            System.err.println("Encountered exception while parsing input arguments:\n"
                               + parseException.getMessage() + "\n\n");
            printHelp(options, true, System.out);
            System.exit(0);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Done");
    }
    
    public static ArrayList<TimeSeries> loadDataset(String dsPathAndName,
                                                    String delimiter) {
        ArrayList<TimeSeries> dataset = null;
        TimeSeries currTS;
        ArrayList<Double> ts;
        int tsClass;
        
        BufferedReader brDataset;
        FileReader frDataset;
        String currLine;
        StringTokenizer st;
        try {
            frDataset = new FileReader(dsPathAndName);
            brDataset = new BufferedReader(frDataset);
            dataset = new ArrayList<TimeSeries>();
            while ((currLine = brDataset.readLine()) != null) {
                if (currLine.matches("\\s*")) {
                    continue;
                } else {
                    currLine = currLine.trim();
                    st = new StringTokenizer(currLine,
                                             String.valueOf(delimiter));
                    tsClass = (int) Double.parseDouble(st.nextToken());
                    ts = new ArrayList<Double>();
                    while (st.hasMoreTokens()) {
                        ts.add(Double.parseDouble(st.nextToken()));
                    }
                    currTS = new TimeSeries(ts, tsClass);
                    dataset.add(currTS);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return dataset;
    }
    
    /**
     * Print help to provided OutputStream.
     * 
     * @param options Command-line options to be part of usage.
     * @param detailed
     * @param out OutputStream to which to write the usage
     *            information.
     */
    public static void printHelp(Options options, boolean detailed,
                                 OutputStream out) {
        String cliSyntax = "java " + StartTests.class.getSimpleName();
        PrintWriter writer = new PrintWriter(out);
        HelpFormatter helpFormatter = new HelpFormatter();
        if (detailed) {
            helpFormatter.printHelp(writer, 120, cliSyntax, "", options, 7, 1,
                                    "", true);
        } else {
            helpFormatter.printUsage(writer, 120, cliSyntax, options);
        }
        writer.flush();
    }
    
    private static Options constructCLIOptions() {
        Option help = new Option(helpSw, helpSwL, false,
                                 "prints the help for the program");
        Option fName = new Option(fileSw, fileSwL, true, "dataset name");
        fName.setArgName("FILE");
        fName.setArgs(1);
        fName.setRequired(true);
        
        Option method = new Option(methodSw, methodSwL, true,
                                   "algorithm to use, 1: Legacy");
        method.setArgName("METHOD");
        method.setArgs(1);
        
        Options opts = new Options();
        opts.addOption(help).addOption(fName).addOption(method);
        return opts;
    }
}
