package gridsearch.classifier;

import java.io.IOException;

import org.aeonbits.owner.ConfigCache;

import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import gridsearch.classifier.setup.ClassifierDescriptionGenerator;

public class GridSearchExperimentRunner {
	public static void main(final String[] args) throws IOException, ExperimentDBInteractionFailedException {

		/* read limit from standard input */
		int limit = Integer.parseInt(args[0]);

		/* create config, db handle, and evaluator for evaluation */
		IClassifierGridSearchConfig config = ConfigCache.getOrCreate(IClassifierGridSearchConfig.class);
		IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(config);
		IExperimentSetEvaluator evaluator = new ClassifierGridSearchEvaluator(new ComponentLoader(ClassifierDescriptionGenerator.CONF_FILE).getComponents(), config.getEvaluationTimeoutInSeconds());

		/* run evaluation */
		ExperimentRunner preparer = new ExperimentRunner(config, evaluator, dbHandle);
		preparer.setLoggerName("gridsearch");
		preparer.randomlyConductExperiments(limit);
	}
}
