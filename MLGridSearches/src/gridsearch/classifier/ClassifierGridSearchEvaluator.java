package gridsearch.classifier;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.TreeNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.serialization.ComponentInstanceDeserializer;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.cache.ReproducibleInstances;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ClassifierGridSearchEvaluator implements IExperimentSetEvaluator {

	private final Collection<Component> components;
	private final DatasetDescriptionGenerator generator = new DatasetDescriptionGenerator();

	public ClassifierGridSearchEvaluator(final Collection<Component> components) {
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

			/* get dataset */
			ArrayNode splitDescription = (ArrayNode)new ObjectMapper().readTree(keys.get("dataset"));
			Instances train = ReproducibleInstances.fromHistory(this.generator.getInstructionsForTrainSet(splitDescription), "");
			Instances test = ReproducibleInstances.fromHistory(this.generator.getInstructionsForTestSet(splitDescription), "");

			/* prepare evaluation */
			Evaluation eval = new Evaluation(train);

			/* train classifier */
			long startTrain = System.currentTimeMillis();
			c.buildClassifier(train);
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
