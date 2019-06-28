package gridsearch.classifier;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.TreeNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.serialization.ComponentInstanceDeserializer;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.WekaUtil;
import ai.libs.jaicore.ml.cache.InstructionGraph;
import ai.libs.jaicore.ml.core.dataset.weka.WekaInstances;
import ai.libs.jaicore.timing.TimedComputation;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.MultiClassClassifier;
import weka.core.Instances;

public class ClassifierGridSearchEvaluator implements IExperimentSetEvaluator {

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
			WEKAPipelineFactory factory = new WEKAPipelineFactory();
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

			/* prepare evaluation */
			Evaluation eval = new Evaluation(train);

			/* train classifier */
			long startTrain = System.currentTimeMillis();
			TimedComputation.compute(() -> {
				c.buildClassifier(train);
				return null;
			}, this.timeoutInSeconds * 1000, "Classifier training timed out");
			Map<String, Object> results = new HashMap<>();
			results.put("time_train", System.currentTimeMillis() - startTrain);
			processor.processResults(results);

			/* compute zero-one loss on training data */
			results.clear();
			eval.evaluateModel(c, train);
			results.put("zoloss_train", eval.errorRate());
			processor.processResults(results);

			/* compute zero-one loss on test data */
			results.clear();
			long startEvaluation = System.currentTimeMillis();
			eval.evaluateModel(c, test);
			results.put("time_predict", System.currentTimeMillis() - startEvaluation);
			results.put("zoloss_test", eval.errorRate());
			processor.processResults(results);
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}
}
