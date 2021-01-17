/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 *    FastForest.java
 *    Copyright (C) 2019-2021 Darren Yates, based on the following source code:
 *
 *    RandomForest.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 *    Bagging.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *    
 */
package weka.classifiers.trees;

import java.util.ArrayList;
import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;
import weka.core.WekaException;
import weka.gui.ProgrammaticProperty;

import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import static weka.classifiers.AbstractClassifier.runClassifier;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.classifiers.evaluation.Evaluation;
import weka.core.AdditionalMeasureProducer;
import weka.core.Aggregateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.PartitionGenerator;
import weka.core.Randomizable;
import weka.core.TechnicalInformationHandler;
import weka.core.WeightedInstancesHandler;

//-------------------------------------------------------------------------------------------------------------------
/**
 * <!-- globalinfo-start --> Class for constructing a FastForest of random
 * trees.<br>
 * <br>
 * For more information see: <br>
 * <br>
 * Leo Breiman (2001). Random Forests. Machine Learning. 45(1):5-32. <br>
 * <br>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;article{Breiman2001,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {5-32},
 *    title = {Random Forests},
 *    volume = {45},
 *    year = {2001}
 * }
 * </pre>
 * //--------------------------------------------------------------------------------------------------------------------------------------------
 * For more information on FastForest see: <br>
 * <br>
 * Yates, D., Islam, M.Z. (2020) FastForest: Increasing Random Forest Processing
 * Speed While Maintaining Accuracy. Information Sciences.<br>
 * <br>
 *
 * RandomForest.java modified by (C) 2019-2021 Darren Yates<br>
 *
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;article{Yates2020,
 *    author = {Darren Yates, Md Zahidul Islam},
 *    journal = {Information Sciences},
 *    title = {FastForest: Increasing Random Forest Processing Speed While Maintaining Accuracy},
 *    year = {2020}
 * }
 * </pre>
 *
 * <br>
 * <br>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p>
 *
 * <pre>
 * -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)
 * </pre>
 *
 * <pre>
 * -O
 *  Calculate the out of bag error.
 * </pre>
 *
 * <pre>
 * -store-out-of-bag-predictions
 *  Whether to store out of bag predictions in internal evaluation object.
 * </pre>
 *
 * <pre>
 * -output-out-of-bag-complexity-statistics
 *  Whether to output complexity-based statistics when out-of-bag evaluation is performed.
 * </pre>
 *
 * <pre>
 * -print
 *  Print the individual classifiers in the output
 * </pre>
 *
 * <pre>
 * -attribute-importance
 *  Compute and output attribute importance (mean impurity decrease method)
 * </pre>
 *
 * <pre>
 * -I &lt;num&gt;
 *  Number of iterations.
 *  (current value 100)
 * </pre>
 *
 * <pre>
 * -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)
 *  (use 0 to auto-detect number of cores)
 * </pre>
 *
 * <pre>
 * -K &lt;number of attributes&gt;
 *  Number of attributes to randomly investigate. (default 0)
 *  (&lt;1 = int(log_2(#predictors)+1)).
 * </pre>
 *
 * <pre>
 * -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 1)
 * </pre>
 *
 * <pre>
 * -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).
 * </pre>
 *
 * <pre>
 * -S &lt;num&gt;
 *  Seed for random number generator.
 *  (default 1)
 * </pre>
 *
 * <pre>
 * -depth &lt;num&gt;
 *  The maximum depth of the tree, 0 for unlimited.
 *  (default 0)
 * </pre>
 *
 * <pre>
 * -N &lt;num&gt;
 *  Number of folds for backfitting (default 0, no backfitting).
 * </pre>
 *
 * <pre>
 * -U
 *  Allow unclassified instances.
 * </pre>
 *
 * <pre>
 * -B
 *  Break ties randomly when several attributes look equally good.
 * </pre>
 *
 * <pre>
 * -output-debug-info
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <pre>
 * -do-not-check-capabilities
 *  If set, classifier capabilities are not checked before classifier is built
 *  (use with caution).
 * </pre>
 *
 * <pre>
 * -num-decimal-places
 *  The number of decimal places for the output of numbers in the model (default 2).
 * </pre>
 *
 * <pre>
 * -batch-size
 *  The desired batch size for batch prediction  (default 100).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 13295 $
 */
public class FastForest 
            extends RandomizableParallelIteratedSingleClassifierEnhancer
            implements WeightedInstancesHandler, AdditionalMeasureProducer,
            TechnicalInformationHandler, PartitionGenerator, Aggregateable<FastForest> {

    /**
     * for serialization
     */
    static final long serialVersionUID = 1116839470751428698L;

    /**
     * True to compute attribute importance
     */
    protected boolean m_computeAttributeImportance;

    /**
     * The default number of iterations to perform.
     */
    @Override
    protected int defaultNumberOfIterations() {
        return 100;
    }

    /**
     * Constructor that sets base classifier to RandomTreeFF and
     * default number of iterations to 100.
     */
    public FastForest() {

        RandomTreeFF rTree = new RandomTreeFF();
        rTree.setDoNotCheckCapabilities(true);
        setClassifier(rTree);
        setRepresentCopiesUsingWeights(true);
        setNumIterations(defaultNumberOfIterations());
    }

    /**
     * Returns default capabilities of the base classifier.
     *
     * @return the capabilities of the base classifier
     */
    public Capabilities getCapabilities() {

        // Cannot use the main RandomTreeFF object because capabilities checking has
        // been turned off
        // for that object.
        return (new RandomTreeFF()).getCapabilities();
    }

    /**
     * String describing default classifier.
     *
     * @return the default classifier classname
     */
    @Override
    protected String defaultClassifierString() {

        return "weka.classifiers.trees.RandomTreeFF";
    }

    /**
     * String describing default classifier options.
     *
     * @return the default classifier options
     */
    @Override
    protected String[] defaultClassifierOptions() {

        String[] args = {"-do-not-check-capabilities"};
        return args;
    }

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {

        return "Class for constructing a forest of random trees.\n\n"
                + "For more information see: \n\n" + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Leo Breiman");
        result.setValue(Field.YEAR, "2001");
        result.setValue(Field.TITLE, "Random Forests");
        result.setValue(Field.JOURNAL, "Machine Learning");
        result.setValue(Field.VOLUME, "45");
        result.setValue(Field.NUMBER, "1");
        result.setValue(Field.PAGES, "5-32");

        return result;
    }

    /**
     * This method only accepts RandomTreeFF arguments.
     *
     * @param newClassifier the RandomTreeFF to use.
     * @exception if argument is not a RandomTreeFF
     */
    @Override
    @ProgrammaticProperty
    public void setClassifier(Classifier newClassifier) {
        if (!(newClassifier instanceof RandomTreeFF)) {
            throw new IllegalArgumentException(
                    "FastForest: Argument of setClassifier() must be a RandomTreeFF.");
        }
        super.setClassifier(newClassifier);
    }

    /**
     * This method only accepts true as its argument
     *
     * @param representUsingWeights must be set to true.
     * @exception if argument is not true
     */
    @ProgrammaticProperty
    public void setRepCopiesUsingWeights(boolean representUsingWeights) {
        if (!representUsingWeights) {
            throw new IllegalArgumentException(
                    "FastForest: Argument of setRepCopiesUsingWeights() must be true.");
        }
        setRepresentCopiesUsingWeights(representUsingWeights);
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String numFeaturesTipText() {
        return ((RandomTreeFF) getClassifier()).KValueTipText();
    }

    /**
     * Get the number of features used in random selection.
     *
     * @return Value of numFeatures.
     */
    public int getNumFeatures() {

        return ((RandomTreeFF) getClassifier()).getKValue();
    }

    /**
     * Set the number of features to use in random selection.
     *
     * @param newNumFeatures Value to assign to numFeatures.
     */
    public void setNumFeatures(int newNumFeatures) {

        ((RandomTreeFF) getClassifier()).setKValue(newNumFeatures);
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String computeAttributeImportanceTipText() {
        return "Compute attribute importance via mean impurity decrease";
    }

    /**
     * Set whether to compute and output attribute importance scores
     *
     * @param computeAttributeImportance true to compute attribute importance
     * scores
     */
    public void setComputeAttributeImportance(boolean computeAttributeImportance) {
        m_computeAttributeImportance = computeAttributeImportance;
        ((RandomTreeFF) m_Classifier).setComputeImpurityDecreases(computeAttributeImportance);
    }

    /**
     * Get whether to compute and output attribute importance scores
     *
     * @return true if computing attribute importance scores
     */
    public boolean getComputeAttributeImportance() {
        return m_computeAttributeImportance;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String maxDepthTipText() {
        return ((RandomTreeFF) getClassifier()).maxDepthTipText();
    }

    /**
     * Get the maximum depth of trh tree, 0 for unlimited.
     *
     * @return the maximum depth.
     */
    public int getMaxDepth() {
        return ((RandomTreeFF) getClassifier()).getMaxDepth();
    }

    /**
     * Set the maximum depth of the tree, 0 for unlimited.
     *
     * @param value the maximum depth.
     */
    public void setMaxDepth(int value) {
        ((RandomTreeFF) getClassifier()).setMaxDepth(value);
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String breakTiesRandomlyTipText() {
        return ((RandomTreeFF) getClassifier()).breakTiesRandomlyTipText();
    }

    /**
     * Get whether to break ties randomly.
     *
     * @return true if ties are to be broken randomly.
     */
    public boolean getBreakTiesRandomly() {

        return ((RandomTreeFF) getClassifier()).getBreakTiesRandomly();
    }

    /**
     * Set whether to break ties randomly.
     *
     * @param newBreakTiesRandomly true if ties are to be broken randomly
     */
    public void setBreakTiesRandomly(boolean newBreakTiesRandomly) {

        ((RandomTreeFF) getClassifier()).setBreakTiesRandomly(newBreakTiesRandomly);
    }

    /**
     * Set debugging mode.
     *
     * @param debug true if debug output should be printed
     */
    public void setDebug(boolean debug) {

        super.setDebug(debug);
        ((RandomTreeFF) getClassifier()).setDebug(debug);
    }

    /**
     * Set the number of decimal places.
     */
    public void setNumDecimalPlaces(int num) {

        super.setNumDecimalPlaces(num);
        ((RandomTreeFF) getClassifier()).setNumDecimalPlaces(num);
    }

    /**
     * Set the preferred batch size for batch prediction.
     *
     * @param size the batch size to use
     */
    @Override
    public void setBatchSize(String size) {

        super.setBatchSize(size);
        ((RandomTreeFF) getClassifier()).setBatchSize(size);
    }

    /**
     * Sets the seed for the random number generator.
     *
     * @param s the seed to be used
     */
    public void setSeed(int s) {

        super.setSeed(s);
        ((RandomTreeFF) getClassifier()).setSeed(s);
    }

    /**
     * Returns description of the bagged classifier.
     *
     * @return description of the bagged classifier as a string
     */
    @Override
    public String toString() {

        if (m_Classifiers == null) {
            return "FastForest: No model built yet.";
        }
        StringBuilder buffer = new StringBuilder("FastForest\n\n");
        buffer.append(super.toString());

        if (getComputeAttributeImportance()) {
            try {
                double[] nodeCounts = new double[m_data.numAttributes()];
                double[] impurityScores
                        = computeAverageImpurityDecreasePerAttribute(nodeCounts);
                int[] sortedIndices = Utils.sort(impurityScores);
                buffer
                        .append("\n\nAttribute importance based on average impurity decrease "
                                + "(and number of nodes using that attribute)\n\n");
                for (int i = sortedIndices.length - 1; i >= 0; i--) {
                    int index = sortedIndices[i];
                    if (index != m_data.classIndex()) {
                        buffer
                                .append(
                                        Utils.doubleToString(impurityScores[index], 10,
                                                getNumDecimalPlaces())).append(" (")
                                .append(Utils.doubleToString(nodeCounts[index], 6, 0))
                                .append(")  ").append(m_data.attribute(index).name())
                                .append("\n");
                    }
                }
            } catch (WekaException ex) {
                // ignore
            }
        }

        return buffer.toString();
    }

    /**
     * Computes the average impurity decrease per attribute over the trees
     *
     * @param nodeCounts an optional array that, if non-null, will hold the
     * count of the number of nodes at which each attribute was used for
     * splitting
     * @return the average impurity decrease per attribute over the trees
     */
    public double[] computeAverageImpurityDecreasePerAttribute(
            double[] nodeCounts) throws WekaException {

        if (m_Classifiers == null) {
            throw new WekaException("Classifier has not been built yet!");
        }

        if (!getComputeAttributeImportance()) {
            throw new WekaException("Stats for attribute importance have not "
                    + "been collected!");
        }

        double[] impurityDecreases = new double[m_data.numAttributes()];
        if (nodeCounts == null) {
            nodeCounts = new double[m_data.numAttributes()];
        }
        for (Classifier c : m_Classifiers) {
            double[][] forClassifier = ((RandomTreeFF) c).getImpurityDecreases();
            for (int i = 0; i < m_data.numAttributes(); i++) {
                impurityDecreases[i] += forClassifier[i][0];
                nodeCounts[i] += forClassifier[i][1];
            }
        }
        for (int i = 0; i < m_data.numAttributes(); i++) {
            if (nodeCounts[i] > 0) {
                impurityDecreases[i] /= nodeCounts[i];
            }
        }

        return impurityDecreases;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>();

        newVector.addElement(new Option(
                "\tSize of each bag, as a percentage of the\n"
                + "\ttraining set size. (default 100)", "P", 1, "-P"));

        newVector.addElement(new Option("\tCalculate the out of bag error.", "O",
                0, "-O"));

        newVector
                .addElement(new Option(
                        "\tWhether to store out of bag predictions in internal evaluation object.",
                        "store-out-of-bag-predictions", 0, "-store-out-of-bag-predictions"));

        newVector
                .addElement(new Option(
                        "\tWhether to output complexity-based statistics when out-of-bag evaluation is performed.",
                        "output-out-of-bag-complexity-statistics", 0,
                        "-output-out-of-bag-complexity-statistics"));

        newVector
                .addElement(new Option(
                        "\tPrint the individual classifiers in the output", "print", 0,
                        "-print"));

        newVector.addElement(new Option(
                "\tCompute and output attribute importance (mean impurity decrease "
                + "method)", "attribute-importance", 0, "-attribute-importance"));

        newVector.addElement(new Option("\tNumber of iterations.\n"
                + "\t(current value " + getNumIterations() + ")", "I", 1, "-I <num>"));

        newVector.addElement(new Option("\tNumber of execution slots.\n"
                + "\t(default 1 - i.e. no parallelism)\n"
                + "\t(use 0 to auto-detect number of cores)", "num-slots", 1,
                "-num-slots <num>"));

        // Add base classifier options
        List<Option> list
                = Collections.list(((OptionHandler) getClassifier()).listOptions());
        newVector.addAll(list);

        return newVector.elements();
    }

    /**
     * Gets the current settings of the forest.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        result.add("-P");
        result.add("" + getBagSizePercent());

        if (getCalcOutOfBag()) {
            result.add("-O");
        }

        if (getStoreOutOfBagPredictions()) {
            result.add("-store-out-of-bag-predictions");
        }

        if (getOutputOutOfBagComplexityStatistics()) {
            result.add("-output-out-of-bag-complexity-statistics");
        }

        if (getPrintClassifiers()) {
            result.add("-print");
        }

        if (getComputeAttributeImportance()) {
            result.add("-attribute-importance");
        }

        result.add("-I");
        result.add("" + getNumIterations());

        result.add("-num-slots");
        result.add("" + getNumExecutionSlots());

        if (getDoNotCheckCapabilities()) {
            result.add("-do-not-check-capabilities");
        }

        // Add base classifier options
        Vector<String> classifierOptions = new Vector<String>();
        Collections.addAll(classifierOptions,
                ((OptionHandler) getClassifier()).getOptions());
        Option.deleteFlagString(classifierOptions, "-do-not-check-capabilities");
        result.addAll(classifierOptions);

        return result.toArray(new String[result.size()]);
    }

    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start --> Valid options are:
     * <p>
     *
     * <pre>
     * -P
     *  Size of each bag, as a percentage of the
     *  training set size. (default 100)
     * </pre>
     *
     * <pre>
     * -O
     *  Calculate the out of bag error.
     * </pre>
     *
     * <pre>
     * -store-out-of-bag-predictions
     *  Whether to store out of bag predictions in internal evaluation object.
     * </pre>
     *
     * <pre>
     * -output-out-of-bag-complexity-statistics
     *  Whether to output complexity-based statistics when out-of-bag evaluation is performed.
     * </pre>
     *
     * <pre>
     * -print
     *  Print the individual classifiers in the output
     * </pre>
     *
     * <pre>
     * -attribute-importance
     *  Compute and output attribute importance (mean impurity decrease method)
     * </pre>
     *
     * <pre>
     * -I &lt;num&gt;
     *  Number of iterations.
     *  (current value 100)
     * </pre>
     *
     * <pre>
     * -num-slots &lt;num&gt;
     *  Number of execution slots.
     *  (default 1 - i.e. no parallelism)
     *  (use 0 to auto-detect number of cores)
     * </pre>
     *
     * <pre>
     * -K &lt;number of attributes&gt;
     *  Number of attributes to randomly investigate. (default 0)
     *  (&lt;1 = int(log_2(#predictors)+1)).
     * </pre>
     *
     * <pre>
     * -M &lt;minimum number of instances&gt;
     *  Set minimum number of instances per leaf.
     *  (default 1)
     * </pre>
     *
     * <pre>
     * -V &lt;minimum variance for split&gt;
     *  Set minimum numeric class variance proportion
     *  of train variance for split (default 1e-3).
     * </pre>
     *
     * <pre>
     * -S &lt;num&gt;
     *  Seed for random number generator.
     *  (default 1)
     * </pre>
     *
     * <pre>
     * -depth &lt;num&gt;
     *  The maximum depth of the tree, 0 for unlimited.
     *  (default 0)
     * </pre>
     *
     * <pre>
     * -N &lt;num&gt;
     *  Number of folds for backfitting (default 0, no backfitting).
     * </pre>
     *
     * <pre>
     * -U
     *  Allow unclassified instances.
     * </pre>
     *
     * <pre>
     * -B
     *  Break ties randomly when several attributes look equally good.
     * </pre>
     *
     * <pre>
     * -output-debug-info
     *  If set, classifier is run in debug mode and
     *  may output additional info to the console
     * </pre>
     *
     * <pre>
     * -do-not-check-capabilities
     *  If set, classifier capabilities are not checked before classifier is built
     *  (use with caution).
     * </pre>
     *
     * <pre>
     * -num-decimal-places
     *  The number of decimal places for the output of numbers in the model (default 2).
     * </pre>
     *
     * <pre>
     * -batch-size
     *  The desired batch size for batch prediction  (default 100).
     * </pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        String bagSize = Utils.getOption('P', options);
        if (bagSize.length() != 0) {
            setBagSizePercent(Integer.parseInt(bagSize));
        } else {
            setBagSizePercent(100);
        }

        setCalcOutOfBag(Utils.getFlag('O', options));

        setStoreOutOfBagPredictions(Utils.getFlag("store-out-of-bag-predictions",
                options));

        setOutputOutOfBagComplexityStatistics(Utils.getFlag(
                "output-out-of-bag-complexity-statistics", options));

        setPrintClassifiers(Utils.getFlag("print", options));

        setComputeAttributeImportance(Utils
                .getFlag("attribute-importance", options));

        String iterations = Utils.getOption('I', options);
        if (iterations.length() != 0) {
            setNumIterations(Integer.parseInt(iterations));
        } else {
            setNumIterations(defaultNumberOfIterations());
        }

        String numSlots = Utils.getOption("num-slots", options);
        if (numSlots.length() != 0) {
            setNumExecutionSlots(Integer.parseInt(numSlots));
        } else {
            setNumExecutionSlots(1);
        }

        RandomTreeFF classifier
                = ((RandomTreeFF) AbstractClassifier.forName(defaultClassifierString(),
                        options));
        classifier.setComputeImpurityDecreases(m_computeAttributeImportance);
        setDoNotCheckCapabilities(classifier.getDoNotCheckCapabilities());
        setSeed(classifier.getSeed());
        setDebug(classifier.getDebug());
        setNumDecimalPlaces(classifier.getNumDecimalPlaces());
        setBatchSize(classifier.getBatchSize());
        classifier.setDoNotCheckCapabilities(true);


            setCalcOutOfBag(Utils.getFlag('O', options));

            setStoreOutOfBagPredictions(Utils.getFlag("store-out-of-bag-predictions", options));

            setOutputOutOfBagComplexityStatistics(Utils.getFlag("output-out-of-bag-complexity-statistics", options));

            setRepresentCopiesUsingWeights(Utils.getFlag("represent-copies-using-weights", options));

            setPrintClassifiers(Utils.getFlag("print", options));
        
        // Set base classifier and options
        setClassifier(classifier);

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 13295 $");
    }

    /**
     * Main method for this class.
     *
     * @param argv the options
     */
    public static void main(String[] argv) {
        runClassifier(new FastForest(), argv);
    }

    
//---------------------------------------------------------------------------------------------------------------------------------------------------
/*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
*    Modifications Copyright (C) 2019-2021 Darren Yates, using:
*
*    Bagging.java
*    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
*
 */
/**
 * <!-- globalinfo-start -->
 * Class for bagging a classifier to reduce variance. Can do classification
 * and regression depending on the base learner. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Leo Breiman (1996). Bagging predictors. Machine Learning. 24(2):123-140.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman1996,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {2},
 *    pages = {123-140},
 *    title = {Bagging predictors},
 *    volume = {24},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 *
 * //-------------------------------------------------------------------------------------------------------------------
 * Bagging.java modified by (C) 2019-2021 Darren Yates<br>
 *
 * For more information on FastForest see: <br>
 * <br>
 * Yates, D., Islam, M.Z. (2020) FastForest: Increasing Random Forest
 * Processing Speed While Maintaining Accuracy. Information Sciences.<br>
 * <br>
 *
 *
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;article{Yates2020,
 *    author = {Darren Yates, Md Zahidul Islam},
 *    journal = {Information Sciences},
 *    title = {FastForest: Increasing Random Forest Processing Speed While Maintaining Accuracy},
 *    year = {2020}
 * }
 * </pre>
 * //-------------------------------------------------------------------------------------------------------------------
 *
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * Valid options are:
 * <p/>
 *
 * <pre> -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)</pre>
 *
 * <pre> -O
 *  Calculate the out of bag error.</pre>
 *
 * <pre> -print
 *  Print the individual classifiers in the output</pre>
 *
 * <pre> -store-out-of-bag-predictions
 *  Whether to store out of bag predictions in internal evaluation object.</pre>
 *
 * <pre> -output-out-of-bag-complexity-statistics
 *  Whether to output complexity-based statistics when out-of-bag evaluation is performed.</pre>
 *
 * <pre> -represent-copies-using-weights
 *  Represent copies of instances using weights rather than explicitly.</pre>
 *
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 *
 * <pre> -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)</pre>
 *
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.REPTree)</pre>
 *
 * <pre>
 * Options specific to classifier weka.classifiers.trees.REPTree:
 * </pre>
 *
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf (default 2).</pre>
 *
 * <pre> -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).</pre>
 *
 * <pre> -N &lt;number of folds&gt;
 *  Number of folds for reduced error pruning (default 3).</pre>
 *
 * <pre> -S &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 *
 * <pre> -P
 *  No pruning.</pre>
 *
 * <pre> -L
 *  Maximum tree depth (default -1, no maximum)</pre>
 *
 * <pre> -I
 *  Initial class value count (default 0)</pre>
 *
 * <pre> -R
 *  Spread initial count over all class values (i.e. don't use 1 per value)</pre>
 *
 * <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (len@reeltwo.com)
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 14879 $
 */

    /**
     * The size of each bag sample, as a percentage of the training size
     */
    protected int m_BagSizePercent = 100;

    /**
     * Whether to calculate the out of bag error
     */
    protected boolean m_CalcOutOfBag = false;

    /**
     * Whether to represent copies of instances using weights rather than
     * explicitly
     */
    protected boolean m_RepresentUsingWeights = false;

    /**
     * The evaluation object holding the out of bag error, etc.
     */
    protected Evaluation m_OutOfBagEvaluationObject = null;

    /**
     * Whether to store the out of bag predictions in the evaluation object.
     */
    private boolean m_StoreOutOfBagPredictions = false;

    /**
     * Whether to output complexity-based statistics when OOB-evaluation is
     * performed.
     */
    private boolean m_OutputOutOfBagComplexityStatistics;

    /**
     * Whether class is numeric.
     */
    private boolean m_Numeric = false;

    /**
     * Whether to print individual ensemble members in output.
     */
    private boolean m_printClassifiers;

    /**
     * Random number generator
     */
    protected Random m_random;

    /**
     * Used to indicate whether an instance is in a bag or not
     */
    protected boolean[][] m_inBag;

    /**
     * Reference to the training data
     */
    protected Instances m_data;


    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String bagSizePercentTipText() {
        return "Size of each bag, as a percentage of the training set size.";
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public int getBagSizePercent() {

        return m_BagSizePercent;
    }

    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newBagSizePercent the bag size, as a percentage.
     */
    public void setBagSizePercent(int newBagSizePercent) {

        m_BagSizePercent = newBagSizePercent;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String representCopiesUsingWeightsTipText() {
        return "Whether to represent copies of instances using weights rather than explicitly.";
    }

    /**
     * Set whether copies of instances are represented using weights rather
     * than explicitly.
     *
     * @param representUsingWeights whether to represent copies using
     * weights
     */
    public void setRepresentCopiesUsingWeights(boolean representUsingWeights) {

        m_RepresentUsingWeights = representUsingWeights;
    }

    /**
     * Get whether copies of instances are represented using weights rather
     * than explicitly.
     *
     * @return whether copies of instances are represented using weights
     * rather than explicitly
     */
    public boolean getRepresentCopiesUsingWeights() {

        return m_RepresentUsingWeights;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String storeOutOfBagPredictionsTipText() {
        return "Whether to store the out-of-bag predictions.";
    }

    /**
     * Set whether the out of bag predictions are stored.
     *
     * @param storeOutOfBag whether the out of bag predictions are stored
     */
    public void setStoreOutOfBagPredictions(boolean storeOutOfBag) {

        m_StoreOutOfBagPredictions = storeOutOfBag;
    }

    /**
     * Get whether the out of bag predictions are stored.
     *
     * @return whether the out of bag predictions are stored
     */
    public boolean getStoreOutOfBagPredictions() {

        return m_StoreOutOfBagPredictions;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String calcOutOfBagTipText() {
        return "Whether the out-of-bag error is calculated.";
    }

    /**
     * Set whether the out of bag error is calculated.
     *
     * @param calcOutOfBag whether to calculate the out of bag error
     */
    public void setCalcOutOfBag(boolean calcOutOfBag) {

        m_CalcOutOfBag = calcOutOfBag;
    }

    /**
     * Get whether the out of bag error is calculated.
     *
     * @return whether the out of bag error is calculated
     */
    public boolean getCalcOutOfBag() {

        return m_CalcOutOfBag;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String outputOutOfBagComplexityStatisticsTipText() {

        return "Whether to output complexity-based statistics when out-of-bag evaluation is performed.";
    }

    /**
     * Gets whether complexity statistics are output when OOB estimation is
     * performed.
     *
     * @return whether statistics are calculated
     */
    public boolean getOutputOutOfBagComplexityStatistics() {

        return m_OutputOutOfBagComplexityStatistics;
    }

    /**
     * Sets whether complexity statistics are output when OOB estimation is
     * performed.
     *
     * @param b whether statistics are calculated
     */
    public void setOutputOutOfBagComplexityStatistics(boolean b) {

        m_OutputOutOfBagComplexityStatistics = b;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String printClassifiersTipText() {
        return "Print the individual classifiers in the output";
    }

    /**
     * Set whether to print the individual ensemble classifiers in the
     * output
     *
     * @param print true if the individual classifiers are to be printed
     */
    public void setPrintClassifiers(boolean print) {
        m_printClassifiers = print;
    }

    /**
     * Get whether to print the individual ensemble classifiers in the
     * output
     *
     * @return true if the individual classifiers are to be printed
     */
    public boolean getPrintClassifiers() {
        return m_printClassifiers;
    }

    /**
     * Gets the out of bag error that was calculated as the classifier was
     * built. Returns error rate in classification case and mean absolute
     * error in regression case.
     *
     * @return the out of bag error; -1 if out-of-bag-error has not be
     * estimated
     */
    public double measureOutOfBagError() {

        if (m_OutOfBagEvaluationObject == null) {
            return -1;
        }
        if (m_Numeric) {
            return m_OutOfBagEvaluationObject.meanAbsoluteError();
        } else {
            return m_OutOfBagEvaluationObject.errorRate();
        }
    }

    /**
     * Returns an enumeration of the additional measure names.
     *
     * @return an enumeration of the measure names
     */
    @Override
    public Enumeration<String> enumerateMeasures() {

        Vector<String> newVector = new Vector<String>(1);
        newVector.addElement("measureOutOfBagError");
        return newVector.elements();
    }

    /**
     * Returns the value of the named measure.
     *
     * @param additionalMeasureName the name of the measure to query for its
     * value
     * @return the value of the named measure
     * @throws IllegalArgumentException if the named measure is not
     * supported
     */
    @Override
    public double getMeasure(String additionalMeasureName) {

        if (additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")) {
            return measureOutOfBagError();
        } else {
            throw new IllegalArgumentException(additionalMeasureName
                    + " not supported");
        }
    }

    /**
     * Returns a training set for a particular iteration.
     *
     * @param iteration the number of the iteration for the requested
     * training set.
     * @return the training set for the supplied iteration number
     * @throws Exception if something goes wrong when generating a training
     * set.
     */
    @Override
    protected synchronized Instances getTrainingSet(int iteration) throws Exception {

        Random r = new Random(m_Seed + iteration);
        Instances bagData = null;

        // create the in-bag indicator array if necessary
        if (m_CalcOutOfBag) {
            m_inBag[iteration] = new boolean[m_data.numInstances()];
            return m_data.resampleWithWeights(r, m_inBag[iteration], getRepresentCopiesUsingWeights());
        } else {
            // //-----------------------------------------------------------------------------------------------------------------------------------------------
            // FastForest modification //    
            Instances tempData = new Instances(m_data);
            tempData.randomize(r);
            return (new Instances(tempData, 0, (int) (0.5 * m_data.numInstances())));
            // FastForest modification //    
            // //-----------------------------------------------------------------------------------------------------------------------------------------------
        }
    }

    /**
     * Returns the out-of-bag evaluation object.
     *
     * @return the out-of-bag evaluation object; null if out-of-bag error
     * hasn't been calculated
     */
    public Evaluation getOutOfBagEvaluationObject() {

        return m_OutOfBagEvaluationObject;
    }

    /**
     * Bagging method.
     *
     * @param data the training data to be used for generating the bagged
     * classifier.
     * @throws Exception if the classifier could not be built successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // Has user asked to represent copies using weights?
        if (getRepresentCopiesUsingWeights() && !(m_Classifier instanceof WeightedInstancesHandler)) {
            throw new IllegalArgumentException("Cannot represent copies using weights when "
                    + "base learner in FastForest-Bagging does not implement "
                    + "WeightedInstancesHandler.");
        }

        // get fresh Instances object
        m_data = new Instances(data);

        //----------------------------------------------------------------------------------------
        super.buildClassifier(m_data);

        m_random = new Random(m_Seed);

        m_inBag = null;
        if (m_CalcOutOfBag) {
            m_inBag = new boolean[m_Classifiers.length][];
        }

        for (int j = 0; j < m_Classifiers.length; j++) {
            if (m_Classifier instanceof Randomizable) {
                ((Randomizable) m_Classifiers[j]).setSeed(m_random.nextInt());
            }
        }

        m_Numeric = m_data.classAttribute().isNumeric();

        buildClassifiers();

        // calc OOB error?
        if (getCalcOutOfBag()) {
            m_OutOfBagEvaluationObject = new Evaluation(m_data);

            for (int i = 0; i < m_data.numInstances(); i++) {
                double[] votes;
                if (m_Numeric) {
                    votes = new double[1];
                } else {
                    votes = new double[m_data.numClasses()];
                }

                // determine predictions for instance
                int voteCount = 0;
                for (int j = 0; j < m_Classifiers.length; j++) {
                    if (m_inBag[j][i]) {
                        continue;
                    }

                    if (m_Numeric) {
                        double pred = m_Classifiers[j].classifyInstance(m_data.instance(i));
                        if (!Utils.isMissingValue(pred)) {
                            votes[0] += pred;
                            voteCount++;
                        }
                    } else {
                        voteCount++;
                        double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
                        // sum the probability estimates
                        for (int k = 0; k < newProbs.length; k++) {
                            votes[k] += newProbs[k];
                        }
                    }
                }

                // "vote"
                if (m_Numeric) {
                    if (voteCount > 0) {
                        votes[0] /= voteCount;
                        m_OutOfBagEvaluationObject.evaluationForSingleInstance(votes, m_data.instance(i), getStoreOutOfBagPredictions());
                    }
                } else {
                    double sum = Utils.sum(votes);
                    if (sum > 0) {
                        Utils.normalize(votes, sum);
                        m_OutOfBagEvaluationObject.evaluationForSingleInstance(votes, m_data.instance(i), getStoreOutOfBagPredictions());
                    }
                }
            }
        } else {
            m_OutOfBagEvaluationObject = null;
        }

        // save memory
        m_inBag = null;
        m_data = new Instances(m_data, 0);
    }

    /*
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance the instance to be classified
     * @return preedicted class probability distribution
     * @throws Exception if distribution can't be computed successfully
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        double[] sums = new double[instance.numClasses()], newProbs;

        double numPreds = 0;
        for (int i = 0; i < m_NumIterations; i++) {
            if (m_Numeric) {
                double pred = m_Classifiers[i].classifyInstance(instance);
                if (!Utils.isMissingValue(pred)) {
                    sums[0] += pred;
                    numPreds++;
                }
            } else {
                newProbs = m_Classifiers[i].distributionForInstance(instance);
                for (int j = 0; j < newProbs.length; j++) {
                    sums[j] += newProbs[j];
                }
            }
        }
        if (m_Numeric) {
            if (numPreds == 0) {
                sums[0] = Utils.missingValue();
            } else {
                sums[0] /= numPreds;
            }
            return sums;
        } else if (Utils.eq(Utils.sum(sums), 0)) {
            return sums;
        } else {
            Utils.normalize(sums);
            return sums;
        }
    }

    /**
     * Builds the classifier to generate a partition.
     */
    @Override
    public void generatePartition(Instances data) throws Exception {

        if (m_Classifier instanceof PartitionGenerator) {
            buildClassifier(data);
        } else {
            throw new Exception("Classifier: " + getClassifierSpec()
                    + " cannot generate a partition");
        }
    }

    /**
     * Computes an array that indicates leaf membership
     */
    @Override
    public double[] getMembershipValues(Instance inst) throws Exception {

        if (m_Classifier instanceof PartitionGenerator) {
            ArrayList<double[]> al = new ArrayList<double[]>();
            int size = 0;
            for (int i = 0; i < m_Classifiers.length; i++) {
                double[] r = ((PartitionGenerator) m_Classifiers[i]).
                        getMembershipValues(inst);
                size += r.length;
                al.add(r);
            }
            double[] values = new double[size];
            int pos = 0;
            for (double[] v : al) {
                System.arraycopy(v, 0, values, pos, v.length);
                pos += v.length;
            }
            return values;
        } else {
            throw new Exception("Classifier: " + getClassifierSpec()
                    + " cannot generate a partition");
        }
    }

    /**
     * Returns the number of elements in the partition.
     */
    @Override
    public int numElements() throws Exception {

        if (m_Classifier instanceof PartitionGenerator) {
            int size = 0;
            for (int i = 0; i < m_Classifiers.length; i++) {
                size += ((PartitionGenerator) m_Classifiers[i]).numElements();
            }
            return size;
        } else {
            throw new Exception("Classifier: " + getClassifierSpec()
                    + " cannot generate a partition");
        }
    }

    protected List<Classifier> m_classifiersCache;

    /**
     * Aggregate an object with this one
     *
     * @param toAggregate the object to aggregate
     * @return the result of aggregation
     * @throws Exception if the supplied object can't be aggregated for some
     * reason
     */
    @Override
    public FastForest aggregate(FastForest toAggregate) throws Exception {
        if (!m_Classifier.getClass().isAssignableFrom(toAggregate.m_Classifier.getClass())) {
            throw new Exception("Can't aggregate because base classifiers differ");
        }

        if (m_classifiersCache == null) {
            m_classifiersCache = new ArrayList<Classifier>();
            m_classifiersCache.addAll(Arrays.asList(m_Classifiers));
        }
        m_classifiersCache.addAll(Arrays.asList(toAggregate.m_Classifiers));

        return this;
    }

    /**
     * Call to complete the aggregation process. Allows implementers to do
     * any final processing based on how many objects were aggregated.
     *
     * @throws Exception if the aggregation can't be finalized for some
     * reason
     */
    @Override
    public void finalizeAggregation() throws Exception {
        m_Classifiers = m_classifiersCache.toArray(new Classifier[1]);
        m_NumIterations = m_Classifiers.length;

        m_classifiersCache = null;
    }
}
