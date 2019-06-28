package gridsearch.classifier.setup;

import org.aeonbits.owner.ConfigCache;

import ai.libs.jaicore.basic.algorithm.AlgorithmExecutionCanceledException;
import ai.libs.jaicore.basic.algorithm.exceptions.AlgorithmTimeoutedException;
import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import gridsearch.classifier.IClassifierGridSearchConfig;

public class ClassifierGridSearchPreparer {
	public static void main(final String[] args)
			throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, AlgorithmTimeoutedException, InterruptedException, AlgorithmExecutionCanceledException {

		/* create config, db handle, and evaluator for evaluation */
		IClassifierGridSearchConfig config = ConfigCache.getOrCreate(IClassifierGridSearchConfig.class);
		IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(config);

		/* run evaluation */
		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(config, dbHandle);
		preparer.setLoggerName("gridsearch");
		preparer.synchronizeExperiments();
	}
}
