import java.util.List;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import ai.libs.jaicore.ml.WekaUtil;
import ai.libs.jaicore.ml.cache.InstructionFailedException;
import ai.libs.jaicore.ml.cache.ReproducibleInstances;
import ai.libs.jaicore.ml.core.evaluation.measure.singlelabel.ZeroOneLoss;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.MonteCarloCrossValidationEvaluator;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.splitevaluation.SimpleSLCSplitBasedClassifierEvaluator;
import ai.libs.jaicore.timing.TimedComputation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;

public class TestSMODefaultConfig {

	private MonteCarloCrossValidationEvaluator mccv;
	private Instances data;

	@Before
	public void init() throws InstructionFailedException, InterruptedException {
		this.data = ReproducibleInstances.fromOpenML(6, "");
		this.mccv = new MonteCarloCrossValidationEvaluator(new SimpleSLCSplitBasedClassifierEvaluator(new ZeroOneLoss()), 10, this.data, .7, 0);
	}

	@Test
	@Ignore
	public void testAll() {
		WekaUtil.getBasicLearners().parallelStream().forEach(cName -> {
			try {
				Classifier c = AbstractClassifier.forName(cName, new String[] {});
				long start = System.currentTimeMillis();
				double score = TimedComputation.compute(() -> this.mccv.evaluate(c), 900 * 1000, "Classifier killed");
				System.out.println(cName + ": " + score + " (time: " + (System.currentTimeMillis() - start) + ")");
			} catch (Exception e) {
				System.err.println(cName + ": " + e);
			}
		});
	}

	@Ignore
	@Test
	public void testFromDatabase() throws Exception {
		SMO smo = new SMO();
		smo.setOptions(new String[] {"-C", "0.1",  "-N", "1"});
		Kernel kernel = new PolyKernel();
		kernel.setOptions(new String[] {"-E", "3"});
		smo.setKernel(kernel);

		List<Instances> split = WekaUtil.getStratifiedSplit(this.data, 9, .7);
		System.out.println("Split " + this.data.size() + " into " + split.get(0).size() + " + " + split.get(1).size());

		smo.buildClassifier(split.get(0));
		Evaluation eval = new Evaluation(this.data);
		eval.evaluateModel(smo, split.get(1));

		System.out.println(WekaUtil.getClassifierDescriptor(smo) + ": " + eval.errorRate());

	}
}















































