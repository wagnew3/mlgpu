package learningRule;

import validation.Validator;
import costFunctions.CostFunction;
import nDimensionalMatrices.Matrix;
import network.Network;
import network.SplitNetwork;

public abstract class MPLearningRule 
{
	
	public MPLearningRule()
	{
		
	}
	
	public abstract void trainNetwork(SplitNetwork network, Matrix[][] inputs, 
			Matrix[][] desiredOutputs, CostFunction costFunction, Validator validator);

}
