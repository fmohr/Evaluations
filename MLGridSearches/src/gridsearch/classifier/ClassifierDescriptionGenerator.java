package gridsearch.classifier;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import com.fasterxml.jackson.databind.node.ObjectNode;

import hasco.core.HASCO;
import hasco.core.HASCOSolutionCandidate;
import hasco.core.RefinementConfiguredSoftwareConfigurationProblem;
import hasco.events.HASCOSolutionEvent;
import hasco.model.ComponentInstance;
import hasco.variants.forwarddecomposition.HASCOViaFD;
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
	private int numValues = -1;

	public ClassifierDescriptionGenerator() throws IOException {
		super();

		/* create a HASCO obejct */
		RefinementConfiguredSoftwareConfigurationProblem<Double> problem = new RefinementConfiguredSoftwareConfigurationProblem<>(new File("resources/searchmodels/weka/weka-classifiers-base.json"), "AbstractClassifier", n -> 0.0);
		IOptimalPathInORGraphSearchFactory<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, TFDNode, String, Double> searchFactory = new IteratingGraphSearchOptimizerFactory<>(new DepthFirstSearchFactory<>());
		AlgorithmicProblemReduction<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, EvaluatedSearchGraphPath<TFDNode, String, Double>, GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, EvaluatedSearchGraphPath<TFDNode, String, Double>> searchProblemTransformer = new IdentityReduction<>();
		HASCOViaFD<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, Double> hasco = new HASCOViaFD<>(problem, searchFactory, searchProblemTransformer);
		this.hasco = hasco;
	}

	@Override
	public int getNumberOfValues() {
		
		/* if the number of values has not been computed before, run a full DFS */
		if (numValues == -1) {
			Set<ComponentInstance> solutions = new HashSet<>();
			int numSolutions = 0;
			for (AlgorithmEvent e : hasco) {
				if (e instanceof HASCOSolutionEvent){
					numSolutions++;
					ComponentInstance solution = ((HASCOSolutionCandidate<?>)((HASCOSolutionEvent) e).getSolutionCandidate()).getComponentInstance();
					assert !solutions.contains(solution) : "Found solution " + solution + "twice!";
					solutions.add(solution);
					System.out.println(solution.getComponent().getName() + ": " + solution.getParameterValues());
					if (numSolutions % 100 == 0) {
						System.out.println("Found " + numSolutions + " so far.");
					}
				}
			}
			numValues = numSolutions;
		}
		return numValues;
	}

	@Override
	public ObjectNode getValue(int i) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isValueValid(String value) {
		// TODO Auto-generated method stub
		return false;
	}

	public static void main(String[] args) throws IOException {
		ClassifierDescriptionGenerator gen = new ClassifierDescriptionGenerator();
		System.out.println(gen.getNumberOfValues());
	}
}
