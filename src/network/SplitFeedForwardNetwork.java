package network;

import java.util.HashMap;

import layer.BLayer;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import nDimensionalMatrices.SparseFMatrix;

public class SplitFeedForwardNetwork extends SplitNetwork
{

	public SplitFeedForwardNetwork(BLayer[][] layers) 
	{
		super(layers);
	}
	
	@Override
	public Matrix[] getOutput(Matrix[] inputs) 
	{
		HashMap<BLayer, Matrix> layerOutputs=new HashMap<>();
		
		for(int netLayerIndex=0; netLayerIndex<layers[0].length; netLayerIndex++)
		{
			Matrix resultMatrix=null;
			if(inputs[netLayerIndex] instanceof SparseFMatrix)
			{
				resultMatrix=null;
			}
			else
			{
				resultMatrix=new FDMatrix(layers[0][netLayerIndex].getOutputSize(), 1);
			}
			layerOutputs.put(layers[0][netLayerIndex], layers[0][netLayerIndex].getOutput(new Matrix[]{inputs[netLayerIndex]}, null, resultMatrix));
		}
		
		for(int layerIndex=1; layerIndex<layers.length; layerIndex++)
		{
			for(int netLayerIndex=0; netLayerIndex<layers[layerIndex].length; netLayerIndex++)
			{
				Matrix[] nextLayerInputs=new Matrix[layers[layerIndex][netLayerIndex].getInputLayers().length];
				for(int inputInd=0; inputInd<nextLayerInputs.length; inputInd++)
				{
					nextLayerInputs[inputInd]=layerOutputs.get(layers[layerIndex][netLayerIndex].getInputLayers()[inputInd]);
				}
				layerOutputs.put(layers[layerIndex][netLayerIndex], layers[layerIndex][netLayerIndex].getOutput(nextLayerInputs, layers[layerIndex][netLayerIndex].getInputLayers(), new FDMatrix(layers[layerIndex][netLayerIndex].getOutputSize(), 1)));
			}
		}
		
		Matrix[] output=new Matrix[layers[layers.length-1].length];
		for(int outputIndex=0; outputIndex<layers[layers.length-1].length; outputIndex++)
		{
			output[outputIndex]=layerOutputs.get(layers[layers.length-1][outputIndex]);
		}
		
		return output;
	}
	
	@Override
	public HashMap<BLayer, Matrix>[] getOutputs(Matrix[] inputs, HashMap<BLayer, Matrix>[] outputs) 
	{
		for(int netLayerIndex=0; netLayerIndex<layers[0].length; netLayerIndex++)
		{
			outputs[0].put(layers[0][netLayerIndex], layers[0][netLayerIndex].getOutput(new Matrix[]{inputs[netLayerIndex]}, null, outputs[0].get(layers[0][netLayerIndex])));
		}

		for(int layerIndex=1; layerIndex<layers.length; layerIndex++)
		{
			for(int netLayerIndex=0; netLayerIndex<layers[layerIndex].length; netLayerIndex++)
			{
				Matrix[] nextLayerInputs=new Matrix[layers[layerIndex][netLayerIndex].getInputLayers().length];
				for(int inputInd=0; inputInd<nextLayerInputs.length; inputInd++)
				{
					nextLayerInputs[inputInd]=outputs[layerIndex-1].get(layers[layerIndex][netLayerIndex].getInputLayers()[inputInd]);
				}
				outputs[layerIndex].put(layers[layerIndex][netLayerIndex], layers[layerIndex][netLayerIndex].getOutput(nextLayerInputs, layers[layerIndex][netLayerIndex].getInputLayers(), outputs[layerIndex].get(layers[layerIndex][netLayerIndex])));
			} 
		}
		
		return outputs;
	}

	@Override
	public Network clone() {
		// TODO Auto-generated method stub
		return null;
	}

}
