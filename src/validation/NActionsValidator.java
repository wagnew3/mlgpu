package validation;

import nDimensionalMatrices.Matrix;
import network.SplitNetwork;

public class NActionsValidator extends Validator
{

	int numberBestActions;
	
	public NActionsValidator(Matrix[][] validationInputs,
			Matrix[][] validationOutputs, int numberBestActions)
	{
		super(validationInputs, validationOutputs);
		this.numberBestActions=numberBestActions;
	}

	@Override
	public double validate(SplitNetwork network)
	{
		double numberWithPositiveExamples=0;
		double numberCorrect=0;
		for(int validationInputInd=0; validationInputInd<validationInputs.length; 
				validationInputInd++)
		{
			Matrix[] validationInput=validationInputs[validationInputInd];
			Matrix[] validationOutput=validationOutputs[validationInputInd];
			Matrix[] networkOutput=network.getOutput(validationInput);
			
			for(int n=0; n<numberBestActions; n++)
			{
				float largestPositiveWeight=Float.NEGATIVE_INFINITY;
				boolean hasPositiveExample=false;
				int largestPositiveWeightInd=-1;
				for(int outputInd=0; outputInd<validationOutput[0].getLen(); outputInd++)
				{
					if(validationOutput[0].get(outputInd, 0)>0.34f)
					{
						hasPositiveExample=true;
						if(networkOutput[0].get(outputInd, 0)>largestPositiveWeight)
						{
							largestPositiveWeight=networkOutput[0].get(outputInd, 0);
							largestPositiveWeightInd=outputInd;
						}
					}
				}
				if(hasPositiveExample)
				{
					numberWithPositiveExamples++;
					int numberGreaterThan=0;
					for(int outputInd=0; outputInd<validationOutput[0].getLen(); outputInd++)
					{
						if(networkOutput[0].get(outputInd, 0)>largestPositiveWeight)
						{
							numberGreaterThan++;
						}
					}
					if(numberGreaterThan<=numberBestActions-n-1)
					{
						numberCorrect++;
					}
					networkOutput[0].set(largestPositiveWeightInd, 0, -1.0f);
				}
			}
			
		}
		return numberCorrect/validationInputs.length;
	}

}
