package gridsearch.classifier;

import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.experiments.exceptions.ExperimentEvaluationFailedException;

public class ClassifierGridSearchEvaluator implements IExperimentSetEvaluator {

	@Override
	public void evaluate(ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		
		System.out.println(experimentEntry);
	}
}
