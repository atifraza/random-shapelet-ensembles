import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.kramerlab.timeseries.*;
import org.kramerlab.btree.DecisionTree;

public class StartTests {
    // The following static String variables are the CLI switches
    protected static String fileSw = "f";
    protected static String fileSwL = "file";
    protected static String rangeSw = "r";
    protected static String rangeSwL = "range";
    protected static String stepSw = "s";
    protected static String stepSwL = "step";
    protected static String methodSw = "m";
    protected static String methodSwL = "method";
    protected static String helpSw = "h";
    protected static String helpSwL = "help";
    
    /**
     * @param args
     */
    public static void main(String[] args) {
        String dsName = null;       // Time series data set name
        String path = System.getProperty("user.dir") + "/data/";
        int minLen = 1;
        int maxLen = 10;
        int stepSize = 1;
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
            if (cmdLine.hasOption(rangeSw)) {
                String[] temp = cmdLine.getOptionValues(rangeSw);
                minLen = Integer.parseInt(temp[0]);
                maxLen = Integer.parseInt(temp[1]);
            }
            if (cmdLine.hasOption(stepSw)) {
                stepSize = Integer.parseInt(cmdLine.getOptionValue(stepSw));
            }
            if (cmdLine.hasOption(methodSw)) {
                method = Integer.parseInt(cmdLine.getOptionValue(methodSw));
            }
            System.out.println("Dataset: " + dsName + "\n"
                               + "MinLen: " + minLen + "\n"
                               + "MaxLen: " + maxLen + "\n"
                               + "Step size: " + stepSize + "\n"
                               + "Method: " + method + "\n\n");
        } catch (ParseException parseException) {
            // Catch any exceptions here and print the help
            System.err.println("Encountered exception while parsing input arguments:\n"
                               + parseException.getMessage() + "\n\n");
            printHelp(options, true, System.out);
            System.exit(0);
        }
        
        ArrayList<TimeSeries> dataset;
        int correct = 0;
        int predClass = Integer.MIN_VALUE;
        long start, stop;
        
        dataset = loadDataset(path + dsName + "_TRAIN", " ");
        TimeSeriesDataset trainSet = new TimeSeriesDataset(dataset);
        
        dataset = loadDataset(path + dsName + "_TEST", " ");
        TimeSeriesDataset testSet = new TimeSeriesDataset(dataset);
        
        start = System.currentTimeMillis();
        DecisionTree tree = new DecisionTree(trainSet, minLen, maxLen, stepSize,
                                             method);
        for (int ind = 0; ind < testSet.size(); ind++) {
            predClass = tree.checkInstance(testSet.get(ind));
            if (predClass == testSet.get(ind).getLabel()) {
                correct++;
            }
        }
        stop = System.currentTimeMillis();
        try {
            File resultsFile = new File("results/Shapelets_results.csv");
            String temp = "";
            if (!resultsFile.exists()) {
                temp = "Dataset,"
                       + "Train Size,"
                       + "TS Len,"
                       + "MinLen,"
                       + "MaxLen,"
                       + "Method,"
                       + "C_Total,"
                       + "C_Processed,"
                       + "Time (sec),"
                       + "Accuracy (%)\n";
            }
            temp += dsName + ",\t"
                    + trainSet.size() + ",\t"
                    + trainSet.get(0).size() + ",\t"
                    + minLen + ",\t"
                    + maxLen + ",\t"
                    + method + ",\t"
                    + tree.getTotalCandidates() + ",\t"
                    + tree.getPrunedCandidates() + ",\t"
                    + (stop - start) / 1e3 + ",\t"
                    + 100.0 * correct / testSet.size() + "\n";
            FileWriter fw = new FileWriter(resultsFile, true);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(temp);
            bw.close();
            fw.close();
            resultsFile = null;
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
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
        
        Option range = new Option(rangeSw, rangeSwL, true,
                                  "range of lengths to find shapelet");
        range.setArgName("minLen maxLen");
        range.setArgs(2);
        
        Option step = new Option(stepSw, stepSwL, true,
                                 "step size for candiate generation");
        step.setArgName("stepSize");
        step.setArgs(1);
        
        Option method = new Option(methodSw, methodSwL, true,
                                   "algorithm to use, 1: Legacy");
        method.setArgName("METHOD");
        method.setArgs(1);
        
        Options opts = new Options();
        opts.addOption(help).addOption(fName).addOption(range).addOption(step)
            .addOption(method);
        return opts;
    }
}
