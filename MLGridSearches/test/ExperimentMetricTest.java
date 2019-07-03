import static org.junit.Assert.assertEquals;

import java.io.IOException;

import org.junit.Test;

import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.ml.cache.InstructionFailedException;
import gridsearch.classifier.ExperimentMetricComputer;

public class ExperimentMetricTest {

	@Test
	public void testConfusionMatrix() throws ExperimentDBInteractionFailedException, IOException, InstructionFailedException, InterruptedException {
		assertEquals(2, new ExperimentMetricComputer().getConfusionMatrix(62).length);
	}

	@Test
	public void testErrorRate() throws ExperimentDBInteractionFailedException, IOException, InstructionFailedException, InterruptedException {
		assertEquals(0.03226, new ExperimentMetricComputer().getErrorRate(62), 0.0001);
	}
}
