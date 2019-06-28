package gridsearch.classifier.setup;

import java.io.IOException;
import java.util.Arrays;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.experiments.IExperimentJSONKeyGenerator;
import ai.libs.jaicore.ml.cache.InstructionGraph;
import ai.libs.jaicore.ml.cache.LoadDatasetInstructionForOpenML;
import ai.libs.jaicore.ml.cache.StratifiedSplitSubsetInstruction;

public class DatasetDescriptionGenerator implements IExperimentJSONKeyGenerator {

	private final int SEEDS = 10;
	private final int[] openMLIDs = { 3, 6, 12, 14, 16, 18, 21, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 44, 46, 57, 60, 179, 180, 181, 182, 184, 185, 300, //389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 554, 679,
			//			715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913,
			//			914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166
	};

	@Override
	public int getNumberOfValues() {
		return this.openMLIDs.length * this.SEEDS;
	}

	@Override
	public JsonNode getValue(final int i) {
		int seed = i % this.SEEDS;
		int datasetId = Math.floorDiv(i, this.SEEDS);
		InstructionGraph graph = new InstructionGraph();
		graph.addNode("load", new LoadDatasetInstructionForOpenML("", this.openMLIDs[datasetId]));
		graph.addNode("split", new StratifiedSplitSubsetInstruction(seed, .67), Arrays.asList(new Pair<>("load", 0)));
		return new ObjectMapper().valueToTree(graph);
	}

	@Override
	public boolean isValueValid(final String value) {
		ObjectMapper om = new ObjectMapper();
		try {
			ObjectNode node = (ObjectNode) om.readTree(value);
			if (node.size() != 3) {
				return false;
			}
			if (!node.has("base") || !node.get("base").isArray() || node.get("base").size() != 1) {
				return false;
			}
			if (!node.has("train") || !node.get("train").isArray() || node.get("train").size() != 1) {
				return false;
			}
			if (!node.has("test") || !node.get("test").isArray() || node.get("test").size() != 1) {
				return false;
			}
		} catch (ClassCastException | IOException e) {
			return false;
		}
		return true;
	}
}
