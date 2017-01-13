package cNetworks;

import java.io.File;

import layer.BLayer;
import network.SplitFeedForwardNetwork;

public class CNetParamSet
{

	static String fileName="/home/willie/workspace/SAT_Solvers_GPU/data/NetworkPIndusFixConfClausesInformed4043c1100v9a750000ns";
	
	public static void main(String[] args)
	{
		SplitFeedForwardNetwork net=(SplitFeedForwardNetwork)SplitFeedForwardNetwork.loadNetwork(new File(fileName));
		String pSet="int numberLayers="+net.layers.length+";\n";
		pSet+="int* newLayers=(int*)malloc(numberLayers*sizeof(int));\n";
		pSet+="float** biases=(float**)malloc(numberLayers*sizeof(float*));\n";
		pSet+="float*** weights=(float***)malloc(numberLayers*sizeof(float**));\n";
		
		for(int layerInd=0; layerInd<net.layers.length; layerInd++)
		{	
			if(layerInd>0)
			{
				pSet+="newLayers["+layerInd+"]="+net.layers[layerInd][0].getOutputSize()+";\n";
				pSet+="biases["+layerInd+"]=(float*)malloc("+net.layers[layerInd][0].getOutputSize()+"*sizeof(float));\n";
				pSet+="weights["+layerInd+"]=(float**)malloc("+net.layers[layerInd][0].getOutputSize()+"*sizeof(float*));\n";
				for(int neuronInd=0; neuronInd<net.layers[layerInd][0].getOutputSize(); neuronInd++)
				{
					pSet+="biases["+layerInd+"]["+neuronInd+"]="+net.layers[layerInd][0].biases.get(neuronInd, 0)+";\n";
					
					int totalInputSize=0;
					for(BLayer layer: net.layers[layerInd-1])
					{
						totalInputSize+=layer.getOutputSize();
					}
					pSet+="weights["+layerInd+"]["+neuronInd+"]=(float*)malloc("+totalInputSize+"*sizeof(float));\n";
					
					int offset=0;
					for(int prevLayerInd=0; prevLayerInd<net.layers[layerInd-1].length; prevLayerInd++)
					{
						for(int inputNeuronInd=0; inputNeuronInd<net.layers[layerInd-1][prevLayerInd].getOutputSize(); inputNeuronInd++)
						{
							pSet+="weights["+layerInd+"]["+neuronInd+"]["+(inputNeuronInd+offset)+"]="+net.layers[layerInd][0].getWeights()[prevLayerInd].get(neuronInd, inputNeuronInd)+";\n";
						}
						offset+=net.layers[layerInd-1][prevLayerInd].getOutputSize();
					}
				}
			}
			else
			{
				int totalInputSize=0;
				for(BLayer layer: net.layers[layerInd])
				{
					totalInputSize+=layer.getOutputSize();
				}
				pSet+="newLayers["+layerInd+"]="+totalInputSize+";\n";
			}
		}
		
		System.out.println(pSet);
	}
	
}
