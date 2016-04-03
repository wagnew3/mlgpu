package learningRule;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;

import costFunctions.CostFunction;
import network.Network;

public abstract class LearningRule 
{
    
	public LearningRule()
	{
		
	}
	
	public abstract void trainNetwork(Network network, ArrayRealVector[] inputs, 
			ArrayRealVector[] desiredOutputs, CostFunction costFunction);

}
