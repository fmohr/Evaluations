package gridsearch.classifier;

import org.aeonbits.owner.ConfigCache;

import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;

public class ClassifierGridSearch {
	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {

		/* create config, db handle, and evaluator for evaluation */
		IClassifierGridSearchConfig config = ConfigCache.getOrCreate(IClassifierGridSearchConfig.class);
		IExperimentDatabaseHandle dbHandle = new ExperimenterSQLHandle(config);
		IExperimentSetEvaluator evaluator = new ClassifierGridSearchEvaluator();

		/* run evaluation */
		new ExperimentRunner(config, evaluator, dbHandle).randomlyConductExperiments(1, false);
	}
}
