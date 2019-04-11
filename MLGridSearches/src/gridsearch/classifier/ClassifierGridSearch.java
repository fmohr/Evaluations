package gridsearch.classifier;

import org.aeonbits.owner.ConfigCache;

import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentDatabaseHandle;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import jaicore.experiments.exceptions.IllegalExperimentSetupException;

public class ClassifierGridSearch {
	public static void main(String[] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		
		/* create config, db handle, and evaluator for evaluation */
		IClassifierGridSearchConfig config = ConfigCache.getOrCreate(IClassifierGridSearchConfig.class);
		IExperimentDatabaseHandle dbHandle = new ExperimenterSQLHandle(config);
		IExperimentSetEvaluator evaluator = new ClassifierGridSearchEvaluator();
		
		/* run evaluation */
		new ExperimentRunner(config, evaluator, dbHandle).randomlyConductExperiments(1, false);
	}
}
