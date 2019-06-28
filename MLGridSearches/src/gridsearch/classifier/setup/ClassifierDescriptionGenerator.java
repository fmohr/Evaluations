package gridsearch.classifier.setup;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import ai.libs.hasco.core.HASCO;
import ai.libs.hasco.core.HASCOSolutionCandidate;
import ai.libs.hasco.core.RefinementConfiguredSoftwareConfigurationProblem;
import ai.libs.hasco.events.HASCOSolutionEvent;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.serialization.CompositionSerializer;
import ai.libs.hasco.variants.forwarddecomposition.HASCOViaFD;
import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.basic.algorithm.events.AlgorithmEvent;
import ai.libs.jaicore.basic.algorithm.reduction.AlgorithmicProblemReduction;
import ai.libs.jaicore.basic.algorithm.reduction.IdentityReduction;
import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentJSONKeyGenerator;
import ai.libs.jaicore.planning.hierarchical.algorithms.forwarddecomposition.graphgenerators.tfd.TFDNode;
import ai.libs.jaicore.search.algorithms.standard.auxilliary.iteratingoptimizer.IteratingGraphSearchOptimizer;
import ai.libs.jaicore.search.algorithms.standard.auxilliary.iteratingoptimizer.IteratingGraphSearchOptimizerFactory;
import ai.libs.jaicore.search.algorithms.standard.dfs.DepthFirstSearch;
import ai.libs.jaicore.search.algorithms.standard.dfs.DepthFirstSearchFactory;
import ai.libs.jaicore.search.core.interfaces.IOptimalPathInORGraphSearchFactory;
import ai.libs.jaicore.search.model.other.EvaluatedSearchGraphPath;
import ai.libs.jaicore.search.probleminputs.GraphSearchWithPathEvaluationsInput;

public class ClassifierDescriptionGenerator implements IExperimentJSONKeyGenerator {

	private final Logger logger = LoggerFactory.getLogger(ClassifierDescriptionGenerator.class);
	public static final File CONF_FILE = new File("resources/searchmodels/weka/gridsearch.json");
	private final HASCO<?, ?, ?, ?> hasco;
	private List<ObjectNode> configurations;
	private final RefinementConfiguredSoftwareConfigurationProblem<Double> problem;
	private final int maxSolutionsPerRun = 50000;

	public ClassifierDescriptionGenerator() throws IOException {
		super();

		/* create a HASCO object */
		this.problem = new RefinementConfiguredSoftwareConfigurationProblem<>(CONF_FILE, "AbstractClassifier", n -> 0.0);
		IOptimalPathInORGraphSearchFactory<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, TFDNode, String, Double> searchFactory = new IteratingGraphSearchOptimizerFactory<>(new DepthFirstSearchFactory<>());
		AlgorithmicProblemReduction<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, EvaluatedSearchGraphPath<TFDNode, String, Double>, GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, EvaluatedSearchGraphPath<TFDNode, String, Double>> searchProblemTransformer = new IdentityReduction<>();
		HASCOViaFD<GraphSearchWithPathEvaluationsInput<TFDNode, String, Double>, Double> hasco = new HASCOViaFD<>(this.problem, searchFactory, searchProblemTransformer);
		this.hasco = hasco;
		this.logger.info("Created new ClassifierDescriptionGenerator.");
	}

	@Override
	public int getNumberOfValues() {

		try {

			/* if the number of values has not been computed before, run a full DFS */
			if (this.configurations == null) {
				this.logger.debug("Reading configs to get number of values.");
				this.readConfigs();
			}
			return this.configurations.size();
		} catch (Exception e) {
			e.printStackTrace();
			return -1;
		}
	}

	@SuppressWarnings("rawtypes")
	private void readConfigs() throws IOException {
		if (this.configurations != null) {
			throw new UnsupportedOperationException("Cannot reload configurations");
		}
		Set<ComponentInstance> solutions = new HashSet<>();
		File cachedFile = new File(this.problem.hashCode() + ".configs");
		File currentSearchFile = new File("dfs.state");
		StringBuilder sb = new StringBuilder();
		int numSolutions = 0;
		this.configurations = new ArrayList<>();
		this.logger.info("HashCode of problem in file {} with {} components is {}. Present: {}", this.CONF_FILE, this.problem.getComponents().size(), this.problem.hashCode(), cachedFile.exists());
		if (!cachedFile.exists() || currentSearchFile.exists()) {

			/* get HASCO's depth first search in order to retrieve or set the current index  */
			DepthFirstSearch<?, ?> dfs = (DepthFirstSearch) ((IteratingGraphSearchOptimizer) this.hasco.getSearch()).getBaseAlgorithm();

			/* if a search state file exists, set the current state of the DFS according to the sotred decision array */
			if (currentSearchFile.exists()) {
				List<Integer> decisions = SetUtil.unserializeList(FileUtil.readFileAsString(currentSearchFile).trim()).stream().map(Integer::valueOf).collect(Collectors.toList());
				int[] decisionsAsArray = new int[decisions.size()];
				for (int i = 0; i < decisionsAsArray.length; i++) {
					decisionsAsArray[i] = decisions.get(i);
				}
				dfs.setCurrentPath(decisionsAsArray);
				Files.delete(currentSearchFile.toPath());
			}

			/* identify new solutions with HASCO */
			for (AlgorithmEvent e : this.hasco) {
				if (e instanceof HASCOSolutionEvent) {

					/* get the solution object */
					ComponentInstance solution = ((HASCOSolutionCandidate<?>) ((HASCOSolutionEvent) e).getSolutionCandidate()).getComponentInstance();
					assert !solutions.contains(solution) : "Found solution " + solution + "twice!";
					ObjectNode on = CompositionSerializer.serializeComponentInstance(solution);
					this.configurations.add(on);
					sb.append(on.toString() + "\n");
					numSolutions++;
					if (numSolutions % 100 == 0) {
						this.logger.info("Found {} so far.", numSolutions);
					}

					/* if we have the limit of solutions reached, write the candidates into the output file and quit */
					if (numSolutions == this.maxSolutionsPerRun) {
						this.logger.info("Serializing to {}", currentSearchFile.getAbsolutePath());
						try (FileWriter fw = new FileWriter(currentSearchFile)) {
							fw.write(Arrays.toString(dfs.getDecisionIndicesForCurrentPath()));
						}
						this.logger.info("Serialization completed");
						break;
					}
				}
			}
			try (FileWriter fw = new FileWriter(cachedFile, true)) {
				fw.write(sb.toString());
			}
		} else {
			ObjectMapper om = new ObjectMapper();
			for (String line : FileUtil.readFileAsList(cachedFile)) {
				this.configurations.add((ObjectNode) om.readTree(line));
			}
		}
	}

	@Override
	public ObjectNode getValue(final int i) {
		if (this.configurations == null) {
			try {
				this.logger.debug("Reading config to get value for {}-th field", i);
				this.readConfigs();
			} catch (IOException e) {
				this.logger.error("Encountered error: {}", e);
			}
		}
		return this.configurations.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		try {
			Object o = new ObjectMapper().readTree(value);
			if (this.configurations == null) {
				this.readConfigs();
			}
			this.logger.trace("Checking whether configurations {} contains {}", this.configurations, o);
			return this.configurations.contains(o);
		} catch (IOException e) {
			this.logger.error("Encountered error: {}", e);
			return false;
		}
	}

	public static void main(final String[] args) throws IOException {
		ClassifierDescriptionGenerator gen = new ClassifierDescriptionGenerator();
		gen.readConfigs();
		System.out.println(gen.getNumberOfValues());
	}
}
