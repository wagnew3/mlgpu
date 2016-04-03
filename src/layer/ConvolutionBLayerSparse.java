package layer;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import activationFunctions.ActivationFunction;

public class ConvolutionBLayerSparse extends ConvolutionBLayer
{

	public ConvolutionBLayerSparse(ActivationFunction activationFunction,
			BLayer[] inputLayers, int[] inputsDimensions, int stride,
			int[] boxSize) 
	{
		super(activationFunction, inputLayers, inputsDimensions, stride, boxSize);
	}
	
	protected double vectorMatrixDotProduct(BlockRealMatrix vectorMat, SparseArrayRealVector matrix)
	{
		double dotProduct=0.0;
		RealVector vector=vectorMat.getRowVector(0);
		for(int nonZeroInd=0; nonZeroInd<matrix.nonZeroEntries.length; nonZeroInd++)
		{
			dotProduct+=vector.getEntry((int)matrix.nonZeroEntries[nonZeroInd])
					*matrix.getSparseData(nonZeroInd);
		}
		return dotProduct;
	}
	
	@Override
	public ArrayRealVector getActivations(ArrayRealVector input, int weightIndex)
	{
		SparseArrayRealVector sparseInput=(SparseArrayRealVector)input;
		ArrayRealVector z=new ArrayRealVector(getOutputSize());
		
		int neuronIndex=0;
		for(int colInd=0; colInd<numberColumns; colInd+=stride)
		{
			z.setEntry(neuronIndex, vectorMatrixDotProduct(weights[weightIndex], (SparseArrayRealVector)sparseInput.getSubVector(colInd, boxSize[1])));
			neuronIndex++;
		}
		z=(ArrayRealVector)z.mapAdd(biases.getEntry(0));
		return z;
	}
	
	@Override
	public RealMatrix getWeightPDs(ArrayRealVector previousLayerOutputs, ArrayRealVector deltas) 
	{
		SparseArrayRealVector sparsePreviousLayerOutputs=(SparseArrayRealVector)previousLayerOutputs;
		sparsePreviousLayerOutputs.sparseData=sparsePreviousLayerOutputs.sparseData.clone();
		
		ArrayRealVector condensedPreviousLayerOutputs=new ArrayRealVector(boxSize[1]);
		for(int entryInd=0; entryInd<sparsePreviousLayerOutputs.nonZeroEntries.length; entryInd++)
		{
			condensedPreviousLayerOutputs
				.addToEntry((int)sparsePreviousLayerOutputs.nonZeroEntries[entryInd]%condensedPreviousLayerOutputs.getDimension(),
						sparsePreviousLayerOutputs.sparseData[entryInd]);
		}
		
		double deltasSum=0.0;
		for(double entry: deltas.getDataRef())
		{
			deltasSum+=entry;
		}
		
		condensedPreviousLayerOutputs.mapMultiplyToSelf(deltasSum);
		
		double[][] weightsPD=new double[][]{condensedPreviousLayerOutputs.getDataRef()};
		
		return new BlockRealMatrix(weightsPD);
	}
	
	//weightpds=deltas*activations
	// sum and average columns to get weight pds with single delta (single bias derivative)
	//result is inputs, add to appropriate weight variable for weights pd 
	public void updateWeights(BlockRealMatrix[] weightPDs, double learningRate)
	{
		for(int inputInd=0; inputInd<inputLayers.length; inputInd++)
		{
			weights[inputInd]=weights[inputInd].subtract(weightPDs[inputInd].scalarMultiply(learningRate));
		}
	}
	
	@Override
	public BlockRealMatrix[] getWeights()
	{
		return weights;
	}

}
