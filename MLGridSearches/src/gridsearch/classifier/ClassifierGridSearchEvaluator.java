package gridsearch.classifier;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;

public class ClassifierGridSearchEvaluator implements IExperimentSetEvaluator {

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {

		System.out.println(experimentEntry);
	}
}
