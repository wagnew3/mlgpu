package layer;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import activationFunctions.ActivationFunction;
import jcuda.Pointer;
import learningRule.MPBackPropGradientDescent;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import nDimensionalMatrices.SparseFMatrix;

public class ConvolutionBLayerSparseVector extends FullyConnectedBLayer
{

	public ConvolutionBLayerSparseVector(ActivationFunction activationFunction, 
			BLayer[] inputLayers,
			int windowSize,
			int size) 
	{
		super(activationFunction, inputLayers);
		this.size=size;
		
		weights=new Matrix[inputLayers.length];
		
		double totalOutputSize=0.0;
		for(int intputLayerInd=0; intputLayerInd<inputLayers.length; intputLayerInd++)
		{
			totalOutputSize+=inputLayers[intputLayerInd].getOutputSize();
		}
		
		for(int intputLayerInd=0; intputLayerInd<inputLayers.length; intputLayerInd++)
		{
			weights[intputLayerInd]=new FDMatrix(new float[windowSize][1]);
			NormalDistribution nInvGaussian=new NormalDistribution(0.0, 1.0/Math.sqrt(totalOutputSize));
			for(int rowIndex=0; rowIndex<weights[intputLayerInd].getRows(); rowIndex++)
			{
				weights[intputLayerInd].set(rowIndex, 0, (float)nInvGaussian.sample());
			}
		}
		
		biases=new FDMatrix(new float[size][1]);
		NormalDistribution zeroGuassian=new NormalDistribution(0.0, 1.0);
		float bias=(float)zeroGuassian.sample();
		for(int rowInd=0; rowInd<biases.getRows(); rowInd++)
		{
			biases.set(rowInd, 0, bias);
		}
		
	}
	
	@Override
	public Matrix getOutput(Matrix[] inputs, BLayer[] inputLayer, Matrix result) 
	{
		Matrix activation=result;
		inputs[0].omatVecMultScaleAdd(inputs[0], weights[0], 1.0f, biases, activation, 0.0f);
		for(int inputInd=1; inputInd<inputs.length; inputInd++)
		{
			activation=inputs[inputInd].omatVecMultScaleAdd(inputs[inputInd], weights[inputInd], 1.0f, biases, activation, 1.0f);
		}
		
		return activationFunction.applyActivationFunction(activation);
	}
	
	@Override
	public Matrix getOutputDerivatives(Matrix[] inputs, Matrix result) 
	{
		Matrix activation=result;
		inputs[0].omatVecMultScaleAdd(inputs[0], weights[0], 1.0f, biases, activation, 0.0f);
		for(int inputInd=1; inputInd<inputs.length; inputInd++)
		{
			activation=inputs[inputInd].omatVecMultScaleAdd(inputs[inputInd], weights[inputInd], 1.0f, biases, activation, 1.0f);
		}
		
		return activationFunction.getDerivatives(activation);
	}

	@Override
	public Matrix getWeightPDs(Matrix previousLayerOutputs, Matrix deltas, Matrix result) 
	{
		SparseFMatrix sparsePreviousLayerOutputs=(SparseFMatrix)previousLayerOutputs;
		float prevOutputsSum=sparsePreviousLayerOutputs.getSum();
		prevOutputsSum/=sparsePreviousLayerOutputs.getLen();
		return deltas.scal(prevOutputsSum, 1, result);
	}
	
	@Override
	public Matrix[] getWeights()
	{
		return weights;
	}

}
