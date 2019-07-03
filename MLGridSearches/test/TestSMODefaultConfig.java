import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.aeonbits.owner.ConfigFactory;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.serialization.ComponentInstanceDeserializer;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.IDatabaseConfig;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentSetConfig;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.ml.WekaUtil;
import ai.libs.jaicore.ml.cache.InstructionFailedException;
import ai.libs.jaicore.ml.cache.InstructionGraph;
import ai.libs.jaicore.ml.cache.ReproducibleInstances;
import ai.libs.jaicore.ml.core.dataset.weka.WekaInstances;
import ai.libs.jaicore.ml.core.evaluation.measure.singlelabel.ZeroOneLoss;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.FixedSplitClassifierEvaluator;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.IClassifierEvaluator;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.MonteCarloCrossValidationEvaluator;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.splitevaluation.SimpleSLCSplitBasedClassifierEvaluator;
import ai.libs.jaicore.ml.weka.dataset.splitter.SplitFailedException;
import ai.libs.jaicore.timing.TimedComputation;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WekaPipelineFactory;
import gridsearch.classifier.ClassifierGridSearchEvaluator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.MultiClassClassifier;
import weka.core.Instances;

public class TestSMODefaultConfig {

	private MonteCarloCrossValidationEvaluator mccv;
	private IClassifierEvaluator sccv;
	private Instances data;
	private Collection<Component> components;

	@Before
	public void init() throws InstructionFailedException, InterruptedException, IOException, SplitFailedException {
		this.data = ReproducibleInstances.fromOpenML(12, "");
		List<Instances> split = WekaUtil.getStratifiedSplit(this.data, 7, .7);
		this.sccv = new FixedSplitClassifierEvaluator(split.get(0), split.get(1));
		this.mccv = new MonteCarloCrossValidationEvaluator(new SimpleSLCSplitBasedClassifierEvaluator(new ZeroOneLoss()), 10, this.data, .7, 0);
		this.components = new ComponentLoader(new File("resources/searchmodels/weka/gridsearch.json")).getComponents();
	}

	@Ignore
	@Test
	public void testAll() {
		WekaUtil.getBasicLearners().parallelStream().forEach(cName -> {
			try {
				Classifier c = AbstractClassifier.forName(cName, new String[] {});
				long start = System.currentTimeMillis();
				double score = TimedComputation.compute(() -> this.sccv.evaluate(c), 300 * 1000, "Classifier killed");
				System.out.println(cName + ": " + score + " (time: " + (System.currentTimeMillis() - start) + ")");
			} catch (Exception e) {
				System.err.println(cName + ": " + e);
			}
		});
	}

	@Ignore
	@Test
	public void testManualConfiguration() throws Exception {

		SMO smo = new SMO();
		smo.setOptions(new String[] {"-C", "0.25", "-N", "0"});
		Kernel kernel = new PolyKernel();
		kernel.setOptions(new String[] {"-E", "2"});
		smo.setKernel(kernel);

		List<Instances> split = WekaUtil.getStratifiedSplit(this.data, 7, .7);
		System.out.println("Split " + this.data.size() + " into " + split.get(0).size() + " + " + split.get(1).size());

		MultiClassClassifier mmc = new MultiClassClassifier();
		mmc.setClassifier(smo);
		mmc.setOptions(new String[] {"-M", "3"});


		mmc.buildClassifier(split.get(0));
		Evaluation eval = new Evaluation(this.data);
		eval.evaluateModel(mmc, split.get(1));

		System.out.println(WekaUtil.getClassifierDescriptor(mmc) + ": " + eval.errorRate());

	}

	@Test
	public void testByExperimentSimulation() throws Exception {
		final IDatabaseConfig conf = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("conf/database.cfg"));
		final IExperimentSetConfig expConf = (IExperimentSetConfig) ConfigFactory.create(IExperimentSetConfig.class).loadPropertiesFromFile(new File("conf/gridsearch.cfg"));
		final IExperimentDatabaseHandle handle = new ExperimenterMySQLHandle(conf);
		handle.setup(expConf);
		ClassifierGridSearchEvaluator evaluator = new ClassifierGridSearchEvaluator(this.components, 10);
		evaluator.setLoggerName("testedalgorithm");
		evaluator.evaluate(handle.getExperimentWithId(19228), m -> System.out.println("Observed updates: " + m));

	}

	@Test
	@Ignore
	public void testFromJSON() throws Exception {

		String classifierDescription = "{\"params\": {\"M\": \"3\"}, \"component\": \"weka.classifiers.meta.MultiClassClassifier\", \"requiredInterfaces\": {\"W\": {\"params\": {\"C\": \"0.25\", \"M\": \"false\", \"N\": \"0\"}, \"component\": \"weka.classifiers.functions.SMO\", \"requiredInterfaces\": {\"K\": {\"params\": {\"E\": \"2\"}, \"component\": \"weka.classifiers.functions.supportVector.PolyKernel\", \"requiredInterfaces\": {}}}}}}";
		Classifier c = new WekaPipelineFactory().getComponentInstantiation(new ComponentInstanceDeserializer(this.components).readFromJson(classifierDescription));

		String datasetDescription = "[{\"name\": \"load\", \"inputs\": [], \"instruction\": {\"id\": \"12\", \"apiKey\": \"\", \"command\": \"LoadDatasetInstructionForOpenML\"}}, {\"name\": \"split\", \"inputs\": [{\"x\": \"load\", \"y\": 0}], \"instruction\": {\"seed\": 7, \"command\": \"StratifiedSplitSubsetInstruction\", \"portionOfFirstFold\": 0.67}}]";
		InstructionGraph graph = InstructionGraph.fromJson(datasetDescription);

		Instances train = ((WekaInstances<Object>)graph.getDataForUnit(new Pair<>("split", 0))).getList();
		Instances test = ((WekaInstances<Object>)graph.getDataForUnit(new Pair<>("split", 1))).getList();

		c.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(c, train);
		eval.evaluateModel(c, test);

		System.out.println(WekaUtil.getClassifierDescriptor(c) + ": " + eval.errorRate());

	}
}















































