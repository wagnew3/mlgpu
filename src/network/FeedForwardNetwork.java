package network;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealVector;

import activationFunctions.ActivationFunction;
import filters.ScaleFilter;
import layer.Layer;

public class FeedForwardNetwork extends Network
{
	
	
	public FeedForwardNetwork(Layer[] layers)
	{
		super(layers);
	}

	@Override
	public ArrayRealVector getOutput(ArrayRealVector input) 
	{
		for(int layerIndex=0; layerIndex<layers.length; layerIndex++)
		{
			input=layers[layerIndex].getOutput(input);
		}
		return input;
	}
	
	public FeedForwardNetwork clone()
	{
		Layer[] newLayers=new Layer[layers.length];
		for(int layerInd=0; layerInd<newLayers.length; layerInd++)
		{
			newLayers[layerInd]=layers[layerInd].clone();
		}
		return new FeedForwardNetwork(newLayers);
	}

}
