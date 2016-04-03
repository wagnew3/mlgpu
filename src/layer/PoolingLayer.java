package layer;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import activationFunctions.ActivationFunction;
import activationFunctions.PoolingActivationFunction;

public class PoolingLayer extends ConvolutionLayer
{

	public PoolingLayer(PoolingActivationFunction activationFunction, int[] inputsDimesions, int stride, int boxSize) 
	{
		super(activationFunction, inputsDimesions, stride, boxSize);
		
		weights=new BlockRealMatrix(1, boxSize*boxSize);
		for(int weightInd=0; weightInd<boxSize; weightInd++)
		{
			weights.setEntry(0, weightInd, 1.0);
		}
		
		biases=new ArrayRealVector(1, 0.0);
	}
	
	@Override
	public ArrayRealVector getOutput(ArrayRealVector input) 
	{
		ArrayRealVector z=getActivations(input);
		ArrayRealVector output=new ArrayRealVector(z.getDimension());
		for(int zInd=0; zInd<z.getDimension(); zInd++)
		{
			output.setEntry(zInd, 
					((PoolingActivationFunction)activationFunction).
					applyPoolingActivationFunction(new ArrayRealVector(new double[]{z.getEntry(zInd)})));
		}
		return output;
	}
	
	@Override
	public ArrayRealVector getActivations(ArrayRealVector input)
	{
		BlockRealMatrix matrixInputs=new BlockRealMatrix(inputsDimesions[0], inputsDimesions[1]);
		ArrayRealVector z=new ArrayRealVector(getOutputSize());
		for(int rowInd=0; rowInd<matrixInputs.getRowDimension(); rowInd++)
		{
			matrixInputs.setRow(rowInd, input.getSubVector(rowInd*inputsDimesions[1], inputsDimesions[1]).toArray());
		}
		
		int neuronIndex=0;
		for(int rowInd=0; rowInd<numberRows; rowInd+=stride)
		{
			for(int colInd=0; colInd<numberColumns; colInd+=stride)
			{
				ArrayRealVector neuronInputs=null;
				try
				{
				neuronInputs=matrixToVector(matrixInputs.getSubMatrix(rowInd, rowInd+boxSize-1, colInd, colInd+boxSize-1));
				}
				catch(Exception e)
				{
					e.printStackTrace();
				}
				z.setEntry(neuronIndex, ((PoolingActivationFunction)activationFunction).applyPoolingActivationFunction(neuronInputs));
				neuronIndex++;
			}
		}
		return z;
	}
	
	@Override
	public ArrayRealVector getOutputDerivatives(ArrayRealVector input) 
	{
		BlockRealMatrix matrixInputs=new BlockRealMatrix(inputsDimesions[0], inputsDimesions[1]);
		ArrayRealVector z=new ArrayRealVector(getOutputSize());
		for(int rowInd=0; rowInd<matrixInputs.getRowDimension(); rowInd++)
		{
			matrixInputs.setRow(rowInd, input.getSubVector(rowInd*inputsDimesions[1], inputsDimesions[1]).toArray());
		}
		
		int neuronIndex=0;
		for(int rowInd=0; rowInd<numberRows; rowInd+=stride)
		{
			for(int colInd=0; colInd<numberColumns; colInd+=stride)
			{
				ArrayRealVector neuronInputs=matrixToVector(matrixInputs.getSubMatrix(rowInd, rowInd+boxSize-1, colInd, colInd+boxSize-1));
				z.setEntry(neuronIndex, ((PoolingActivationFunction)activationFunction).getPoolingDerivatives(neuronInputs));
				neuronIndex++;
			}
		}
		return z;
	}
	
	@Override
	public RealMatrix getWeightPDs(ArrayRealVector previousLayerOutputs, ArrayRealVector deltas) 
	{
		return deltas.outerProduct(previousLayerOutputs);
	}
	
	public void updateWeights(BlockRealMatrix weightPDs, double learningRate)
	{

	}
	
	public void updateBiases(ArrayRealVector biasesPDs, double learningRate)
	{

	}

}
