package costFunctions;

import layer.SparseArrayRealVector;
import nDimensionalMatrices.Matrix;

public class EuclideanDistanceCostFunction extends CostFunction
{

	@Override
	public float getCost(Matrix[] inputs, Matrix[] networkOutput, Matrix[] desiredOutput) 
	{
		float total=0.0f;
		for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
		{
			Matrix difference=null;
			if(false) //SparseArrayRealVector
			{
				/*
				SparseArrayRealVector sparseDesiredOutput=(SparseArrayRealVector)desiredOutput;
				
				difference=new ArrayRealVector(sparseDesiredOutput.getDimension());
				
				difference=difference.add(networkOutput);
				for(int entryInd=0; entryInd<sparseDesiredOutput.nonZeroEntries.length; entryInd++)
				{
					difference.addToEntry((int)sparseDesiredOutput.nonZeroEntries[entryInd], 
							-1.0*sparseDesiredOutput.sparseData[entryInd]);
				}
				*/
			}
			else
			{
				difference=networkOutput[outputInd].msub(desiredOutput[outputInd], networkOutput[outputInd]);
			}
			total+=(float)(0.5*difference.dot(difference));
			difference.clear();
		}
		return total;
	}

	@Override
	public Matrix[] getCostDerivative(Matrix[] inputs, Matrix[] networkOutput,
			Matrix[] desiredOutput, Matrix[] results) 
	{
		/*
		if(networkOutput instanceof SparseArrayRealVector
				&& desiredOutput instanceof SparseArrayRealVector)
		{
			SparseArrayRealVector sparseNetworkOutput=(SparseArrayRealVector)networkOutput;
			SparseArrayRealVector sparseDesiredOutput=(SparseArrayRealVector)desiredOutput;
			
			result=new ArrayRealVector(sparseNetworkOutput.getDimension());
			
			for(int entryInd=0; entryInd<sparseNetworkOutput.nonZeroEntries.length; entryInd++)
			{
				result.addToEntry((int)sparseNetworkOutput.nonZeroEntries[entryInd], 
						sparseNetworkOutput.sparseData[entryInd]);
			}
			for(int entryInd=0; entryInd<sparseDesiredOutput.nonZeroEntries.length; entryInd++)
			{
				result.addToEntry((int)sparseDesiredOutput.nonZeroEntries[entryInd], 
						-1.0*sparseDesiredOutput.sparseData[entryInd]);
			}
		}
		else if(networkOutput instanceof SparseArrayRealVector)
		{
			SparseArrayRealVector sparseNetworkOutput=(SparseArrayRealVector)networkOutput;
			
			result=new ArrayRealVector(sparseNetworkOutput.getDimension());
			
			for(int entryInd=0; entryInd<sparseNetworkOutput.nonZeroEntries.length; entryInd++)
			{
				result.addToEntry((int)sparseNetworkOutput.nonZeroEntries[entryInd], 
						sparseNetworkOutput.sparseData[entryInd]);
			}
			result=result.subtract(desiredOutput);
		}
		else if(desiredOutput instanceof SparseArrayRealVector)
		{
			SparseArrayRealVector sparseDesiredOutput=(SparseArrayRealVector)desiredOutput;
			
			result=new ArrayRealVector(sparseDesiredOutput.getDimension());
			
			result=result.add(networkOutput);
			for(int entryInd=0; entryInd<sparseDesiredOutput.nonZeroEntries.length; entryInd++)
			{
				result.addToEntry((int)sparseDesiredOutput.nonZeroEntries[entryInd], 
						-1.0*sparseDesiredOutput.sparseData[entryInd]);
			}
		}
		else
		{
		*/
			
			/*
		}
		*/
		for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
		{
			results[outputInd]=networkOutput[outputInd].msub(desiredOutput[outputInd], results[outputInd]);
			for(int subOutputInd=0; subOutputInd<results[outputInd].getLen(); subOutputInd++)
			{
				if(Float.isInfinite(results[outputInd].get(subOutputInd, 0)) || Float.isNaN(results[outputInd].get(subOutputInd, 0))
						|| results[outputInd].get(subOutputInd, 0)>100000)
				{
					int u=0;
				}
			}
		}
		return results;
	}
	
}
