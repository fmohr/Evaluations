package gridsearch.classifier;

import java.io.File;

import org.aeonbits.owner.Config.LoadPolicy;
import org.aeonbits.owner.Config.LoadType;
import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.basic.IDatabaseConfig;
import ai.libs.jaicore.experiments.IExperimentSetConfig;

@LoadPolicy(LoadType.MERGE)
@Sources({ "file:./conf/gridsearch.cfg", "file:./conf/database.cfg" })
public interface IClassifierGridSearchConfig extends IExperimentSetConfig, IDatabaseConfig {

	public static final String GS_EVALUATION_TIMEOUT = "gridsearch.evaluation.timeout";
	public static final String GS_HASCOFILE = "gridsearch.components";

	@Key(GS_EVALUATION_TIMEOUT)
	@DefaultValue("60")
	public int getEvaluationTimeoutInSeconds();

	@Key(GS_HASCOFILE)
	public File getComponentDescriptionFile();
}
