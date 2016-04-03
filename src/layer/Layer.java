package layer;

import java.io.Serializable;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import activationFunctions.ActivationFunction;

public abstract class Layer implements Serializable
{
	
	ActivationFunction activationFunction;
	BlockRealMatrix weights;
	ArrayRealVector biases;
	Layer[] outputLayers;
	
	public Layer(ActivationFunction activationFunction)
	{
		this.activationFunction=activationFunction;
	}
	
	public abstract ArrayRealVector getOutput(ArrayRealVector input);
	
	public abstract int getOutputSize();
	
	public abstract ArrayRealVector getActivations(ArrayRealVector input);
	
	public abstract ArrayRealVector getOutputDerivatives(ArrayRealVector intput);
	
	//weights.ebemult(connectionMatrix)^T * nextLayerDeltas.ebemult(activationDerivative)
	public abstract ArrayRealVector getDeltas(BlockRealMatrix nextLayerWeights, ArrayRealVector nextLayerDeltas, ArrayRealVector activationDerivative);
	
	public abstract RealMatrix getWeightPDs(ArrayRealVector previousLayerOutputs, ArrayRealVector deltas);
	
	public void updateWeights(BlockRealMatrix weightPDs, double learningRate)
	{
		if(Double.isNaN(weightPDs.getEntry(0, 0)))
		{
			int i=0;
		}
		weights=weights.subtract(weightPDs.scalarMultiply(learningRate));
	}
	
	public void updateBiases(ArrayRealVector biasesPDs, double learningRate)
	{
		if(Double.isNaN(biasesPDs.getEntry(0)))
		{
			int i=0;
		}
		biases=biases.subtract(biasesPDs.mapMultiply(learningRate));
	}
	
	public void setWeights(BlockRealMatrix weights)
	{
		if(Double.isNaN(weights.getEntry(0, 0)))
		{
			int i=0;
		}
		this.weights=weights;
	}
	
	public BlockRealMatrix getWeights()
	{
		return weights;
	}
	
	public ArrayRealVector getWeights(int neuron)
	{
		return new ArrayRealVector(weights.getRow(neuron));
	}
	
	public ArrayRealVector getBiases()
	{
		return biases;
	}
	
	public void setBiases(ArrayRealVector biases)
	{
		if(Double.isNaN(weights.getEntry(0, 0)))
		{
			int i=0;
		}
		this.biases=biases;
	}
	
	public double getBiases(int neuron)
	{
		return biases.getEntry(neuron);
	}
	
	protected BlockRealMatrix outerProduct(ArrayRealVector vectorA, ArrayRealVector vectorB)
    {
		BlockRealMatrix matrix=new BlockRealMatrix(vectorA.getDimension(), vectorB.getDimension());
        for(int row=0; row<vectorA.getDimension(); row++)
        {
            for(int col=0; col<vectorB.getDimension(); col++)
            {
                matrix.addToEntry(row, col, vectorA.getEntry(row)*vectorB.getEntry(col));
            }
        }
        return matrix;
    }
	
	public void setOutputLayers(Layer[] outputLayers)
	{
		this.outputLayers=outputLayers;
	}
	
	public Layer[] getOutputLayers(Layer[] outputLayers)
	{
		return outputLayers;
	}
	
	public abstract Layer clone();
	
}
