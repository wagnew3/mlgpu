package costFunctions;

import activationFunctions.ScaleSoftMax;
import nDimensionalMatrices.Matrix;

public class SoftMaxCrossEntropy extends CostFunction
{
	
	ScaleSoftMax scaleSoftMax;
	CrossEntropy crossEntropy;
	
	public SoftMaxCrossEntropy()
	{
		scaleSoftMax=new ScaleSoftMax();
		crossEntropy=new CrossEntropy();
	}

	@Override
	public float getCost(Matrix[] inputs, Matrix[] networkOutput,
			Matrix[] desiredOutput) 
	{
		for(int netOutputInd=0; netOutputInd<networkOutput.length; netOutputInd++)
		{
			networkOutput[netOutputInd]=scaleSoftMax.applyActivationFunction(networkOutput[netOutputInd]);
		}
		return crossEntropy.getCost(inputs, networkOutput, desiredOutput);
	}

	@Override
	public Matrix[] getCostDerivative(Matrix[] inputs, Matrix[] networkOutput,
			Matrix[] desiredOutput, Matrix[] results) 
	{
		Matrix[] softMaxDerivatives=new Matrix[networkOutput.length];
		for(int netOutputInd=0; netOutputInd<networkOutput.length; netOutputInd++)
		{
			softMaxDerivatives[netOutputInd]=scaleSoftMax.getDerivatives(networkOutput[netOutputInd]);
		}
		for(int netOutputInd=0; netOutputInd<networkOutput.length; netOutputInd++)
		{
			networkOutput[netOutputInd]=scaleSoftMax.applyActivationFunction(networkOutput[netOutputInd]);
		}
		results=crossEntropy.getCostDerivative(inputs, networkOutput, desiredOutput, results);
		for(int netOutputInd=0; netOutputInd<networkOutput.length; netOutputInd++)
		{
			results[netOutputInd]=results[netOutputInd].oebemult(softMaxDerivatives[netOutputInd]);
		}
		return results;
	}

}
