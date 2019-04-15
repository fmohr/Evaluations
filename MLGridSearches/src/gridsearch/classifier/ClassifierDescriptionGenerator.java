package gridsearch.classifier;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import hasco.core.HASCO;
import hasco.core.HASCOSolutionCandidate;
import hasco.core.RefinementConfiguredSoftwareConfigurationProblem;
import hasco.events.HASCOSolutionEvent;
import hasco.model.ComponentInstance;
import hasco.serialization.CompositionSerializer;
import hasco.variants.forwarddecomposition.HASCOViaFD;
import jaicore.basic.FileUtil;
import jaicore.basic.algorithm.events.AlgorithmEvent;
import jaicore.basic.algorithm.reduction.AlgorithmicProblemReduction;
import jaicore.basic.algorithm.reduction.IdentityReduction;
import jaicore.experiments.IExperimentJSONKeyGenerator;
import jaicore.planning.hierarchical.algorithms.forwarddecomposition.graphgenerators.tfd.TFDNode;
import jaicore.search.algorithms.standard.auxilliary.iteratingoptimizer.IteratingGraphSearchOptimizerFactory;
import jaicore.search.algorithms.standard.dfs.DepthFirstSearchFactory;
import jaicore.search.core.interfaces.IOptimalPathInORGraphSearchFactory;
import jaicore.search.model.other.EvaluatedSearchGraphPath;
import jaicore.search.probleminputs.GraphSearchWithPathEvaluationsInput;

public class ClassifierDescriptionGenerator implements IExperimentJSONKeyGenerator {

	private final HASCO<?, ?, ?, ?> hasco;
	private List<ObjectNode> configurations;
	private final RefinementConfiguredSoftwareConfigurationProblem<Double> problem;

	public ClassifierDescriptionGenerator() throws IOException {
		super();

		/* create a HASCO obejct */
		problem = new RefinementConfiguredSoftwareConfigurationProblem<>(new File("resources/searchmodels/weka/weka-classifiers-smo.json"), "AbstractClassifier", n -> 0.0);
		IOptimalPathInORGraphSearchFactory<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, TFDNode, String, Double> searchFactory = new IteratingGraphSearchOptimizerFactory<>(new DepthFirstSearchFactory<>());
		AlgorithmicProblemReduction<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, EvaluatedSearchGraphPath<TFDNode, String, Double>, GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, EvaluatedSearchGraphPath<TFDNode, String, Double>> searchProblemTransformer = new IdentityReduction<>();
		HASCOViaFD<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, Double> hasco = new HASCOViaFD<>(problem, searchFactory, searchProblemTransformer);
		this.hasco = hasco;
	}

	private void initFreshHASCO() {

	}

	@Override
	public int getNumberOfValues() {

		try {

			/* if the number of values has not been computed before, run a full DFS */
			if (configurations == null) {
				readConfigs();
			}
			return configurations.size();
		}
		catch (Exception e) {
			e.printStackTrace();
			return -1;
		}
	}

	private void readConfigs() throws IOException {
		if (configurations != null) {
			throw new UnsupportedOperationException("Cannot reload configurations");
		}
		Set<ComponentInstance> solutions = new HashSet<>();
		File cachedFile = new File(problem.hashCode() + ".configs");
		StringBuilder sb = new StringBuilder();
		int numSolutions = 0;
		configurations = new ArrayList<>();
		if (!cachedFile.exists()) {
			//			new JFXPanel();
			//			Platform.runLater(new AlgorithmVisualizationWindow(hasco, new GraphViewPlugin(), new NodeInfoGUIPlugin<>(new TFDNodeInfoGenerator())));
			for (AlgorithmEvent e : hasco) {
				if (e instanceof HASCOSolutionEvent){
					ComponentInstance solution = ((HASCOSolutionCandidate<?>)((HASCOSolutionEvent) e).getSolutionCandidate()).getComponentInstance();
					assert !solutions.contains(solution) : "Found solution " + solution + "twice!";

					ObjectNode on = CompositionSerializer.serializeComponentInstance(solution);
					configurations.add(on);
					sb.append(on.toString() + "\n");
					numSolutions ++;
					if (numSolutions % 100 == 0) {
						System.out.println("Found " + numSolutions + " so far.");
					}
				}
			}
			try (FileWriter fw = new FileWriter(cachedFile)) {
				fw.write(sb.toString());
			}
		}
		else {
			ObjectMapper om = new ObjectMapper();
			for (String line : FileUtil.readFileAsList(cachedFile)) {
				configurations.add((ObjectNode)om.readTree(line));
			}
		}
	}

	@Override
	public ObjectNode getValue(final int i) {
		try {
			readConfigs();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return configurations.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		try {
			return configurations.contains(new ObjectMapper().readTree(value));
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
	}

	public static void main(final String[] args) throws IOException {
		ClassifierDescriptionGenerator gen = new ClassifierDescriptionGenerator();
		gen.readConfigs();
		System.out.println(gen.getNumberOfValues());
		//		for (ObjectNode configuration : gen.configurations) {
		//			System.out.println(configuration);
		//		}
	}
}
