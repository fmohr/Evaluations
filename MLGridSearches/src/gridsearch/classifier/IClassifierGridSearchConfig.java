package gridsearch.classifier;

import org.aeonbits.owner.Config.LoadPolicy;
import org.aeonbits.owner.Config.LoadType;
import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.basic.IDatabaseConfig;
import ai.libs.jaicore.experiments.IExperimentSetConfig;

@LoadPolicy(LoadType.MERGE)
@Sources({ "file:./conf/gridsearch.cfg", "file:./conf/database.cfg" })
public interface IClassifierGridSearchConfig extends IExperimentSetConfig, IDatabaseConfig {
}
