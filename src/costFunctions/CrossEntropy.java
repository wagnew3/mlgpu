package costFunctions;

import nDimensionalMatrices.*;

public class CrossEntropy extends CostFunction
{

	@Override
	public float getCost(Matrix input, Matrix networkOutput, Matrix desiredOutput) 
	{
		double cost=0.0;
		for(int entryInd=0; entryInd<networkOutput.getRows(); entryInd++)
		{
				cost+=((FDMatrix)desiredOutput).data[entryInd]*Math.log(((FDMatrix)networkOutput).data[entryInd])
						+(1-((FDMatrix)desiredOutput).data[entryInd])*Math.log(1-((FDMatrix)networkOutput).data[entryInd]);
		}
		cost/=networkOutput.getRows();
		cost=-cost;
		return (float)cost;
	}

	@Override
	public Matrix getCostDerivative(Matrix input, Matrix networkOutput, Matrix desiredOutput) 
	{
		return networkOutput.msub(desiredOutput);
	}

}
