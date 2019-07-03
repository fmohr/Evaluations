package gridsearch.classifier;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.TreeNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.serialization.ComponentInstanceDeserializer;
import ai.libs.jaicore.basic.ILoggingCustomizable;
import ai.libs.jaicore.basic.MathExt;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.WekaUtil;
import ai.libs.jaicore.ml.cache.InstructionGraph;
import ai.libs.jaicore.ml.core.dataset.weka.WekaInstances;
import ai.libs.jaicore.timing.TimedComputation;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WekaPipelineFactory;
import weka.classifiers.Classifier;
import weka.classifiers.meta.MultiClassClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifierGridSearchEvaluator implements IExperimentSetEvaluator, ILoggingCustomizable {

	private Logger logger = LoggerFactory.getLogger(ClassifierGridSearchEvaluator.class);

	private final long timeoutInSeconds;

	private final Collection<Component> components;

	public ClassifierGridSearchEvaluator(final Collection<Component> components, final long timeoutInSeconds) {
		this.timeoutInSeconds = timeoutInSeconds;
		this.components = components;
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
		try {

			/* get algorithm */
			TreeNode tn = new ObjectMapper().readTree(keys.get("algorithm"));
			ComponentInstance ci = new ComponentInstanceDeserializer(this.components).readAsTree(tn) ;
			WekaPipelineFactory factory = new WekaPipelineFactory();
			Classifier c = factory.getComponentInstantiation(ci);
			boolean isBinaryClassifier = WekaUtil.getBinaryClassifiers().contains(c.getClass().getSimpleName());

			/* get dataset */
			InstructionGraph graph = new ObjectMapper().readValue(keys.get("dataset"), InstructionGraph.class);
			Instances train = ((WekaInstances<Object>)graph.getDataForUnit(new Pair<>("split", 0))).getList();
			Instances test = ((WekaInstances<Object>)graph.getDataForUnit(new Pair<>("split", 1))).getList();
			boolean isBinaryDataset = train.numClasses() == 2;

			/* forbid evaluation of non-native MCC versions on non-binary problems */
			if (!isBinaryDataset && isBinaryClassifier) {
				throw new IllegalArgumentException("Only evaluate native multi-class classifiers on non-binary datasets!");
			}
			if (isBinaryDataset && c instanceof MultiClassClassifier) {
				throw new IllegalArgumentException("Use MultiClassClassifier only for datasets with more than two classes.");
			}

			/* train classifier */
			long startTrain = System.currentTimeMillis();
			this.logger.info("Starting evaluation of classifier {} on dataset {} with timeout {}s", c, train.relationName(), this.timeoutInSeconds);
			TimedComputation.compute(() -> {
				c.buildClassifier(train);
				return null;
			}, this.timeoutInSeconds * 1000, "Classifier training timed out");
			Map<String, Object> results = new HashMap<>();
			results.put("time_train", System.currentTimeMillis() - startTrain);
			processor.processResults(results);

			/* compute confusion matrix on training data */
			results.clear();
			results.put("predictions_train", new ObjectMapper().writeValueAsString(this.getPredictions(c, train)));
			processor.processResults(results);

			/* compute zero-one loss on test data */
			results.clear();
			long startEvaluation = System.currentTimeMillis();
			results.put("time_predict", System.currentTimeMillis() - startEvaluation);
			results.put("predictions_test", new ObjectMapper().writeValueAsString(this.getPredictions(c, test)));
			processor.processResults(results);
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	private List<List<Number>> getPredictions(final Classifier trainedClassifier, final Instances data) throws Exception {
		List<List<Number>> list = new ArrayList<>(data.size());
		int numClasses = data.numClasses();
		for (Instance i : data) {
			List<Number> vals = new ArrayList<>(2 + numClasses);
			vals.add(i.classValue());
			vals.add((int)trainedClassifier.classifyInstance(i));
			for (double prob : trainedClassifier.distributionForInstance(i)) {
				vals.add(MathExt.round(prob, 4));
			}
			list.add(vals);
		}
		return list;
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}
}
