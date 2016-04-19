/**
 * 
 */
package org.kramerlab.shapelets;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;

import org.kramerlab.timeseries.*;

/**
 * @author razaa
 *
 */
@SuppressWarnings("unused")
public class FastShapelets extends BaseShapelets {
    
    protected int num_class, num_obj;
    protected int R, top_k;
    protected int subseq_len;
    protected int seed;
    protected Random rand;
    protected HashMap<Integer, USAX_elm_type> USAX_Map;
    protected ArrayList<Pair<Integer, Double>> Score_List;
    ArrayList<Integer> Label;


    public FastShapelets(TimeSeriesDataset trainSet, int minLen, int maxLen,
                          int stepSize) {
        super(trainSet, minLen, maxLen, stepSize);
        rand = new Random(seed);

        num_class = this.trainSet.getNumOfClasses();
        num_obj = this.trainSet.size();
        R = 10;
        top_k = 10;
        Label = new ArrayList<>();
        for(int ind = 0; ind < this.trainSet.size(); ind++) {
            Label.add(this.trainSet.get(ind).getLabel());
        }
    }
    
    @Override
    public Shapelet findShapelet() {
        int sax_max_len, sax_len, w;
        int max_len = this.maxLen, min_len = this.minLen, step = 1; //consider whole search space.

        double percent_mask;
        Shapelet sh;

        sax_max_len = 15;
        percent_mask = 0.25;
        
        USAX_Map = new HashMap<>();
        Score_List = new ArrayList<>();
        
        for (subseq_len = min_len; subseq_len <= max_len; subseq_len += step) {
            sax_len = sax_max_len;
            /// Make w and sax_len both integer
            w = (int) Math.ceil(1.0 * subseq_len / sax_len);
            sax_len = (int) Math.ceil(1.0 * subseq_len / w);
            createSAXList(subseq_len, sax_len, w);

            randomProjection(R, percent_mask, sax_len);
            scoreAllSAX(R);

            sh = findBestSAX(top_k);
        }
        return null;
    }
    
    private void createSAXList(int subseq_len, int sax_len, int w) {
        double ex, ex2, mean, std;
        double sum_segment[] = new double[sax_len];
        int elm_segment[] = new int[sax_len];
        int series, j, j_st, k, slot;
        double d;
        int word, prev_word;
        USAX_elm_type ptr;

        //init the element segments to the W value.
        for (k = 0; k < sax_len; k++) {
            elm_segment[k] = w;
//            System.out.print(elm_segment[k] + " ");
        }

        elm_segment[sax_len - 1] = subseq_len - (sax_len - 1) * w;
//        System.out.print("\t" + elm_segment[sax_len - 1] + " ");
//        System.out.println();

        for (series = 0; series < this.trainSet.size(); series++) {
            ex = ex2 = 0;
            prev_word = -1;

            for (k = 0; k < sax_len; k++) {
                sum_segment[k] = 0;
            }

            /// Case 1: Initial
            for (j = 0; (j < this.trainSet.get(series).size()) && (j < subseq_len); j++) {
                d = this.trainSet.get(series).get(j);
                ex += d;
                ex2 += d * d;
                slot = (int) Math.floor((j) / w);
                sum_segment[slot] += d;
            }

            /// Case 2: Slightly Update
            for (; (j <= (int) this.trainSet.get(series).size()); j++) {

                j_st = j - subseq_len;
                mean = ex / subseq_len;
                std = Math.sqrt(ex2 / subseq_len - mean * mean);

                /// Create SAX from sum_segment
                word = createSAXWord(sum_segment, elm_segment, mean, std, sax_len);

                if (word != prev_word) {
                    prev_word = word;
                    //we're updating the reference so no need to re-add.
                    ptr = USAX_Map.get(word);
                    if (ptr == null) {
                        ptr = new USAX_elm_type();
                    }
                    ptr.obj_set.add(series);
                    ptr.sax_id.add(new Pair<>(series, j_st));
                    USAX_Map.put(word, ptr);
                }

                /// For next update
                if (j < this.trainSet.get(series).size()) {
                    double temp = this.trainSet.get(series).get(j_st);

                    ex -= temp;
                    ex2 -= temp * temp;

                    for (k = 0; k < sax_len - 1; k++) {
                        sum_segment[k] -= this.trainSet.get(series).get(j_st + (k) * w);
                        sum_segment[k] += this.trainSet.get(series).get(j_st + (k + 1) * w);
                    }
                    sum_segment[k] -= this.trainSet.get(series).get(j_st + (k) * w);
                    sum_segment[k] += this.trainSet.get(series).get(j_st + Math.min((k + 1) * w, subseq_len));

                    d = this.trainSet.get(series).get(j);
                    ex += d;
                    ex2 += d * d;
                }
            }
        }
    }

    /// Fixed cardinality 4 !!!
    //create a sax word of size 4 here as an int.
    int createSAXWord(double[] sum_segment, int[] elm_segment, double mean, double std, int sax_len) {
        int word = 0, val = 0;
        double d = 0;

        for (int i = 0; i < sax_len; i++) {
            d = (sum_segment[i] / elm_segment[i] - mean) / std;
//            if (d < -0.67) {
//                val = 0;
//            } else if (d < 0) {
//                val = 1;
//            } else if (d < 0.67) {
//                val = 2;
//            } else {
//                val = 3;
//            }
            if (d < 0) {
                if (d < -0.67) {
                    val = 0;
                } else {
                    val = 1;
                }
            } else if (d < 0.67) {
                val = 2;
            } else {
                val = 3;
            }
            word = (word << 2) | (val);
        }
//        System.out.println(Integer.toString(word, 4));
        return word;
    }

    private void randomProjection(int r2, double percent_mask, int sax_len) {
        HashMap<Integer, HashSet<Integer>> Hash_Mark = new HashMap<>();
        int word, mask_word, new_word;
        HashSet<Integer> obj_set, ptr;

        int num_mask = (int) Math.ceil(percent_mask * sax_len);
        System.out.println("num_mask = " + num_mask);

        for (int r = 0; r < R; r++) {
            mask_word = createMaskWord(num_mask, sax_len);
            System.out.println("mask_word = " + mask_word);

            /// random projection and mark non-duplicate object
            for (Map.Entry<Integer, USAX_elm_type> entry : USAX_Map.entrySet()) {
                word = entry.getKey();
                obj_set = entry.getValue().obj_set;
                System.out.println("obj_set = " + obj_set);

                //put the new word and set combo in the hash_mark
                new_word = word | mask_word;
                System.out.println("new_word = " + new_word);

                ptr = Hash_Mark.get(new_word);
                System.out.println("ptr = " + ptr);

                if (ptr == null) {
                    Hash_Mark.put(new_word, new HashSet<>(obj_set));
                } else {
                    //add onto our ptr, rather than overwrite.
                    ptr.addAll(obj_set);
                }
            }

            /// hash again for keep the count
            for (Map.Entry<Integer, USAX_elm_type> entry : USAX_Map.entrySet()) {
                word = entry.getKey();
                new_word = word | mask_word;
                obj_set = Hash_Mark.get(new_word);
                System.out.println("word = " + word);
                System.out.println("new_word = " + new_word);
                System.out.println("obj_set = " + obj_set);
                System.out.print("counts = ");
                //increase the histogram
                for (Integer o_it : obj_set) {
                    Integer count = entry.getValue().obj_count.get(o_it);
                    count = count == null ? 1 : count + 1;
                    System.out.print(count + " ");
                    entry.getValue().obj_count.put(o_it, count);
                }
            }

            Hash_Mark.clear();
        }
    }

    /// create mask word (two random may give same position, we ignore it)
    int createMaskWord(int num_mask, int word_len) {
        int a, b;

        a = 0;
        for (int i = 0; i < num_mask; i++) {
            b = 1 << (word_len / 2);
            //b = 1 << (rand.nextInt()%word_len); //generate a random number between 0 and the word_len
            a = a | b;
        }
        return a;
    }

    private void scoreAllSAX(int R) {
        int word;
        double score;
        USAX_elm_type usax;

        for (Map.Entry<Integer, USAX_elm_type> entry : USAX_Map.entrySet()) {
            word = entry.getKey();
            usax = entry.getValue();
            score = calcScore(usax, R);
            System.out.println("word: " + word + "score: " + score);
            Score_List.add(new Pair<>(word, score));
        }
    }

    /// ***Calc***
    double calcScore(USAX_elm_type usax, int R) {
        double score = -1;
        int cid, count;
        double[] c_in = new double[num_class];       // Count object inside hash bucket
        double[] c_out = new double[num_class];      // Count object outside hash bucket

        /// Note that if no c_in, then no c_out of that object
        for (Map.Entry<Integer, Integer> entry : usax.obj_count.entrySet()) {
            cid = Label.get(entry.getKey())/num_class;
            count = entry.getValue();
            c_in[cid] += (count);
            c_out[cid] += (R - count);
        }
        score = calcScoreFromObjCount(c_in, c_out);
        return score;
    }

    /// Score each sax in the matrix
    double calcScoreFromObjCount(double[] c_in, double[] c_out) {
        /// multi-class
        double diff, sum = 0, max_val = Double.NEGATIVE_INFINITY, min_val = Double.POSITIVE_INFINITY;
        for (int i = 0; i < num_class; i++) {
            diff = (c_in[i] - c_out[i]);
            if (diff > max_val) {
                max_val = diff;
            }
            if (diff < min_val) {
                min_val = diff;
            }
            sum += Math.abs(diff);
        }
        return (sum - Math.abs(max_val) - Math.abs(min_val)) + Math.abs(max_val - min_val);
    }

    private Shapelet findBestSAX(int top_k2) {
        //init the ArrayList with nulls.
        ArrayList<Pair<Integer, Double>> Dist = new ArrayList<>();
        for (int i = 0; i < num_obj; i++) {
            Dist.add(null);
        }

        int word;
        double gain, dist_th, gap;
        int q_obj, q_pos;
        USAX_elm_type usax;
        int label, kk, total_c_in, num_diff;
        
        Shapelet sh = new Shapelet(), bsf_sh = new Shapelet();

        if (top_k > 0) {
            Collections.sort(Score_List, new ScoreComparator());
        }
        top_k = Math.abs(top_k);

        for (int k = 0; k < Math.min(top_k, Score_List.size()); k++) {
            word = Score_List.get(k).first;
            usax = USAX_Map.get(word);
            for (kk = 0; kk < Math.min(usax.sax_id.size(), 1); kk++) {
                int[] c_in = new int[num_class];
                int[] c_out = new int[num_class];
                //init the array list with 0s
                double[] query = new double[subseq_len];

                q_obj = usax.sax_id.get(kk).first;
                q_pos = usax.sax_id.get(kk).second;

                for (int i = 0; i < num_class; i++) {
                    c_in[i] = 0;
                    c_out[i] = Class_Freq[i];
                }
//                for (int i = 0; i < subseq_len; i++) {
//                    query[i] = this.trainSet.get(q_obj).get(q_pos + i);
//                }
//
//                double dist;
//                int m = query.length;
//                double[] Q = new double[m];
//                int[] order = new int[m];
//                for (int obj = 0; obj < num_obj; obj++) {
//                    dist = nn.nearestNeighborSearch(query, this.trainSet.get(obj), obj, Q, order);
//                    Dist.set(obj, new Pair<>(obj, dist));
//                }
//
//                Collections.sort(Dist, new DistComparator());
//
//                total_c_in = 0;
//                for (int i = 0; i < Dist.size() - 1; i++) {
//                    Pair<Integer, Double> pair_i = Dist.get(i);
//                    Pair<Integer, Double> pair_ii = Dist.get(i + 1);
//
//                    dist_th = (pair_i.second + pair_ii.second) / 2.0;
//                    //gap = Dist[i+1].second - dist_th;
//                    gap = ((double) (pair_ii.second - dist_th)) / Math.sqrt(subseq_len);
//                    label = Label.get(pair_i.first);
//                    c_in[label]++;
//                    c_out[label]--;
//                    total_c_in++;
//                    num_diff = Math.abs(num_obj - 2 * total_c_in);
//                    //gain = CalInfoGain1(c_in, c_out);
//                    gain = calcInfoGain2(c_in, c_out, total_c_in, num_obj - total_c_in);
//
//                    sh.setValueFew(gain, gap, dist_th);
//                    if (bsf_sh.lessThan(sh)) {
//                        bsf_sh.setValueAll(gain, gap, dist_th, q_obj, q_pos, subseq_len, num_diff, c_in, c_out);
//                    }
//                }
            }
        }

        return null;
    }

    @Override
    public double getDist(TimeSeries t, TimeSeries s) {
        return 0;
    }
    
    @SuppressWarnings("all")
    private class Pair<A, B> {

        public A first;
        public B second;

        Pair() {
        }

        Pair(A l, B r) {
            first = l;
            second = r;
        }
    }

    @SuppressWarnings("all")
    private class USAX_elm_type {

        HashSet<Integer> obj_set;
        ArrayList<Pair<Integer, Integer>> sax_id;
        HashMap<Integer, Integer> obj_count;

        public USAX_elm_type() {
            obj_set = new HashSet<>();
            sax_id = new ArrayList<>();
            obj_count = new HashMap<>();
        }

    }

    private class ScoreComparator implements Comparator<Pair<Integer, Double>> {

        @Override
        //if the left one is bigger put it closer to the top.
        public int compare(Pair<Integer, Double> t, Pair<Integer, Double> t1) {
            return Double.compare(t1.second, t.second);
        }

    }

    private class DistComparator implements Comparator<Pair<Integer, Double>> {

        @Override
        public int compare(Pair<Integer, Double> t, Pair<Integer, Double> t1) {
            return Double.compare(t.second, t1.second);
        }

    }

}
