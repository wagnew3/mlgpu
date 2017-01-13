package costFunctions;

import nDimensionalMatrices.Matrix;

public class ClassificationCostFunction extends CostFunction
{

	float k;
	EuclideanDistanceCostFunction eDist;
	
	public ClassificationCostFunction(float k)
	{
		super();
		this.k=k;
		eDist=new EuclideanDistanceCostFunction();
	}
	
	@Override
	public float getCost(Matrix[] inputs, Matrix[] networkOutput,
			Matrix[] desiredOutput) 
	{
		return eDist.getCost(inputs, networkOutput, desiredOutput);
	}

	@Override
	public Matrix[] getCostDerivative(Matrix[] inputs, Matrix[] networkOutput,
			Matrix[] desiredOutput, Matrix[] results) 
	{
		for(int outputSectionsInd=0; outputSectionsInd<desiredOutput.length; outputSectionsInd++)
		{
			float totalSum=0.0f;
			float positiveClassExampleSum=0.0f;
			for(int outputInd=0; outputInd<desiredOutput[outputSectionsInd].getLen(); outputInd++)
			{
				totalSum+=networkOutput[outputSectionsInd].get(outputInd, 0);
				if(desiredOutput[outputSectionsInd].get(outputInd, 0)==0.9f)
				{
					positiveClassExampleSum+=networkOutput[outputSectionsInd].get(outputInd, 0);
				}
			}
			if(positiveClassExampleSum>0.000001f)
			{
				for(int outputInd=0; outputInd<desiredOutput[outputSectionsInd].getLen(); outputInd++)
				{
					results[outputSectionsInd].set(outputInd, 0,
							-(float)(Math.signum(desiredOutput[outputSectionsInd].get(outputInd, 0)
									-networkOutput[outputSectionsInd].get(outputInd, 0))
									*Math.abs(Math.pow((networkOutput[outputSectionsInd].get(outputInd, 0)
											-desiredOutput[outputSectionsInd].get(outputInd, 0))/(positiveClassExampleSum), k))));
				}
			}
			else
			{
				results[outputSectionsInd].omscal(0.0f);
			}
		}
		return results;
	}

}
