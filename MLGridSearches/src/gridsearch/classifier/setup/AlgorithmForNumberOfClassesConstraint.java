package gridsearch.classifier.setup;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import org.aeonbits.owner.ConfigCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.serialization.ComponentInstanceDeserializer;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.ILoggingCustomizable;
import ai.libs.jaicore.ml.WekaUtil;
import ai.libs.jaicore.ml.cache.InstructionGraph;
import ai.libs.jaicore.ml.cache.LoadDatasetInstructionForOpenML;
import ai.libs.jaicore.ml.openml.OpenMLHelper;
import gridsearch.classifier.IClassifierGridSearchConfig;
import weka.core.Instances;

public class AlgorithmForNumberOfClassesConstraint implements Predicate<List<String>>, ILoggingCustomizable {

	private Logger logger = LoggerFactory.getLogger(AlgorithmForNumberOfClassesConstraint.class);
	private final IClassifierGridSearchConfig config = ConfigCache.getOrCreate(IClassifierGridSearchConfig.class);
	private final Map<Integer, Integer> cache = new HashMap<>();
	private final Collection<Component> components;



	public AlgorithmForNumberOfClassesConstraint() throws IOException {
		super();
		this.components = new ComponentLoader(this.config.getComponentDescriptionFile()).getComponents();
	}

	@Override
	public boolean test(final List<String> t) {
		if (t.size() < 2) {
			return true;
		}
		try {

			String strClassifier = t.get(0);
			String strDataset = t.get(1);

			/* determine whether the dataset is a binary classification problem */
			LoadDatasetInstructionForOpenML loadInstruction = (LoadDatasetInstructionForOpenML)InstructionGraph.fromJson(strDataset).get(0).getInstruction();
			int dsID = Integer.parseInt(loadInstruction.getId());
			int numClasses = this.cache.computeIfAbsent(dsID, this::getNumberOfClassesForDataset);
			boolean binaryDataset = numClasses == 2;

			/* determine whether the classifier is a native binary classifier */
			ComponentInstance inst = new ComponentInstanceDeserializer(this.components).readFromJson(strClassifier);
			String classifierName = inst.getComponent().getName();
			boolean binaryClassifier = WekaUtil.getBinaryClassifiers().contains(classifierName);

			if (!binaryDataset && binaryClassifier) {
				this.logger.info("Rejecting tuple {}, because this is a multi-class dataset combined with a binary classifier.", t);
				return false;
			}
			else if (binaryDataset && classifierName.equals("weka.classifiers.meta.MultiClassClassifier")) {
				this.logger.info("Rejecting tuple {}, because this is a binary dataset together with MultiClassClassifier, which should ONLY be used for multi class datasets.", t);
				return false;
			}
			return true;
		} catch (IOException e) {
			this.logger.error("Could not test tuple {} due to exception: {}", t, e);
			return false;
		}
	}

	/**
	 * We do NOT use the Feature class of openml here, because it does not work properly
	 *
	 * @param dsId
	 * @return
	 */
	private int getNumberOfClassesForDataset(final int dsId) {

		try {
			Instances inst = OpenMLHelper.getInstancesById(dsId);
			return inst.numClasses();
		}
		catch (Exception e) {
			throw new IllegalArgumentException("Could not identify number of classes for dataset id " + dsId, e);
		}
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}
}
