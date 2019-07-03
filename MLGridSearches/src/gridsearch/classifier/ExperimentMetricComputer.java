package gridsearch.classifier;

import java.io.IOException;
import java.util.List;

import org.aeonbits.owner.ConfigCache;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.ml.cache.InstructionFailedException;
import ai.libs.jaicore.ml.cache.InstructionGraph;
import ai.libs.jaicore.ml.core.dataset.weka.WekaInstances;

public class ExperimentMetricComputer {

	private IClassifierGridSearchConfig config = ConfigCache.getOrCreate(IClassifierGridSearchConfig.class);
	private IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(this.config);

	public ExperimentMetricComputer() throws ExperimentDBInteractionFailedException {
		super();
		this.dbHandle.setup(this.config);
	}

	public WekaInstances<Object> getInstances(final ExperimentDBEntry expEntry) throws JsonParseException, JsonMappingException, InstructionFailedException, InterruptedException, IOException {
		return (WekaInstances<Object>)new ObjectMapper().readValue(expEntry.getExperiment().getValuesOfKeyFields().get("dataset"), InstructionGraph.class).getDataForUnit(new Pair<>("load", 0));
	}

	public List<List<Number>> getTestPredictions(final ExperimentDBEntry expEntry) throws JsonParseException, JsonMappingException, InstructionFailedException, InterruptedException, IOException {
		return new ObjectMapper().readValue(expEntry.getExperiment().getValuesOfResultFields().get("predictions_test").toString(), List.class);
	}

	public int[][] getConfusionMatrix(final int experimentId) throws JsonParseException, JsonMappingException, IOException, ExperimentDBInteractionFailedException, InstructionFailedException, InterruptedException {
		ExperimentDBEntry entry = this.dbHandle.getExperimentWithId(experimentId);
		List<List<Number>> predictions = this.getTestPredictions(entry);
		int numClasses = this.getInstances(entry).getList().numClasses();
		int[][] matrix = new int[numClasses][numClasses];
		for (List<Number> predictionEntry : predictions) {
			int groundTruth = (int)predictionEntry.get(0);
			int prediction = (int)predictionEntry.get(1);
			matrix[groundTruth][prediction] ++;
		}
		return matrix;
	}

	public double getErrorRate(final int experimentId) throws JsonParseException, JsonMappingException, IOException, ExperimentDBInteractionFailedException, InstructionFailedException, InterruptedException {
		int[][] confusionMatrix = this.getConfusionMatrix(experimentId);
		int mistakes = 0;
		int numPredictions = 0;
		int n = confusionMatrix.length;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				numPredictions += confusionMatrix[i][j];
				if (i != j) {
					mistakes += confusionMatrix[i][j];
				}
			}
		}
		return mistakes * 1.0 / numPredictions;
	}
}
