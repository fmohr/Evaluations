package gridsearch.classifier;

import java.io.IOException;

import org.aeonbits.owner.ConfigCache;

import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;

public class ClassifierGridSearch {
	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException, IOException {

		/* create config, db handle, and evaluator for evaluation */
		IClassifierGridSearchConfig config = ConfigCache.getOrCreate(IClassifierGridSearchConfig.class);
		IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(config);
		IExperimentSetEvaluator evaluator = new ClassifierGridSearchEvaluator(new ComponentLoader(ClassifierDescriptionGenerator.CONF_FILE).getComponents());

		/* run evaluation */
		ExperimentRunner runner = new ExperimentRunner(config, evaluator, dbHandle);
		//		runner.createExperiments();
		runner.randomlyConductExperiments(1);
	}
}



