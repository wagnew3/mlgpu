package costFunctions;

import org.apache.commons.math3.linear.ArrayRealVector;

import network.Network;
import network.SplitNetwork;

public abstract class PreprocessCostFunction extends CostFunction
{
	
	public abstract void preprocessDerivatives(ArrayRealVector[] inputs, ArrayRealVector[] desiredOutputs, SplitNetwork network);
	
	public abstract void preprocessCosts(ArrayRealVector[] inputs, ArrayRealVector[] desiredOutputs, SplitNetwork network);

	public abstract double totalCost();
}
