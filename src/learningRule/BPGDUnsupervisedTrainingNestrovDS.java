package learningRule;

import java.util.ArrayList;
import java.util.HashMap;

import activationFunctions.RectifiedLinearActivationFunction;
import activationFunctions.Sigmoid;
import costFunctions.CostFunction;
import layer.BInputLayer;
import layer.BLayer;
import layer.ConvolutionBLayerSparseVector;
import layer.FullyConnectedBLayer;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import nDimensionalMatrices.SparseFMatrix;
import network.SplitFeedForwardNetwork;
import network.SplitNetwork;
import regularization.L2Regularization;

public class BPGDUnsupervisedTrainingNestrovDS extends MPBackPropGradientDescentNestrovDS
{
	
	protected int unsupervisedEpochs;
	
	public BPGDUnsupervisedTrainingNestrovDS(int batchSize, int epochs, int unsupervisedEpochs, float learningRate, float momentumFactor) 
	{
		super(batchSize, epochs, learningRate, momentumFactor);
		this.unsupervisedEpochs=unsupervisedEpochs;
	}

	public SplitNetwork unsupervisedTrain(SplitNetwork network, Matrix[][] inputs,
			Matrix[][] desiredOutputs, CostFunction costFunction)
	{
		HashMap<BLayer, Matrix[]> layerOutputs=new HashMap<>();
		
		for(int netLayerIndex=0; netLayerIndex<network.layers[0].length; netLayerIndex++)
		{
			Matrix[] outputs=new Matrix[inputs.length];
			for(int inputInd=0; inputInd<inputs.length; inputInd++)
			{
				Matrix resultMatrix=null;
				if(inputs[inputInd][netLayerIndex] instanceof SparseFMatrix)
				{
					resultMatrix=null;
				}
				else
				{
					resultMatrix=new FDMatrix(network.layers[0][netLayerIndex].getOutputSize(),1);
				}
				outputs[inputInd]=network.layers[0][netLayerIndex].getOutput(new Matrix[]{inputs[inputInd][netLayerIndex]}, null, resultMatrix);
			}
			layerOutputs.put(network.layers[0][netLayerIndex], outputs);
		}
		
		Matrix[][] savedInputs=null;
		
		for(int layerIndex=1; layerIndex<network.layers.length; layerIndex++)
		{
			for(int netLayerIndex=0; netLayerIndex<network.layers[layerIndex].length; netLayerIndex++)
			{
				Matrix[] currentLayerOutputs=new Matrix[inputs.length];
				Matrix[][] layerInputs=new Matrix[inputs.length][network.layers[layerIndex][netLayerIndex].getInputLayers().length];
				for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
				{
					for(int inputInd=0; inputInd<network.layers[layerIndex][netLayerIndex].getInputLayers().length; inputInd++)
					{
						layerInputs[sampleInd][inputInd]=layerOutputs.get(network.layers[layerIndex][netLayerIndex].getInputLayers()[inputInd])[sampleInd];
					}
					
					currentLayerOutputs[sampleInd]=network.layers[layerIndex][netLayerIndex].getOutput(layerInputs[sampleInd], network.layers[layerIndex][netLayerIndex].getInputLayers(), new FDMatrix(network.layers[layerIndex][netLayerIndex].getOutputSize(), 1));
				}
				
				BInputLayer[] inputLayers=new BInputLayer[network.layers[layerIndex][netLayerIndex].getInputLayers().length];
				for(int inputLayerInd=0; inputLayerInd<inputLayers.length; inputLayerInd++)
				{
					inputLayers[inputLayerInd]=new BInputLayer(null, null, layerInputs[0][inputLayerInd].getLen());	
				}

				BLayer[] oldInputLayers=network.layers[layerIndex][netLayerIndex].getInputLayers();
				BLayer[] oldOutputLayers=network.layers[layerIndex][netLayerIndex].getOutputLayers();
				network.layers[layerIndex][netLayerIndex].outputLayers=new ArrayList<>();
				network.layers[layerIndex][netLayerIndex].setInputLayers(inputLayers);
				
				if(layerIndex<network.layers.length-1)
				{
					if(savedInputs!=null)
					{
						network.layers[layerIndex][netLayerIndex].outputLayers=new ArrayList<>();
						network.layers[layerIndex][netLayerIndex].setInputLayers(oldInputLayers);
						
						inputLayers=new BInputLayer[oldInputLayers.length];
						for(int inputLayerInd=0; inputLayerInd<inputLayers.length; inputLayerInd++)
						{
							inputLayers[inputLayerInd]=new BInputLayer(null, null, layerInputs[0][0].getLen());	
						}
						
						int outputSize=0;
						BLayer[] outputLayers=new BLayer[oldInputLayers.length];
						for(int outputLayerInd=0; outputLayerInd<outputLayers.length; outputLayerInd++)
						{
							outputLayers[outputLayerInd]=null;
							outputLayers[outputLayerInd]
									=new FullyConnectedBLayer(new Sigmoid(), 
											new BLayer[]{network.layers[layerIndex][netLayerIndex]}, 
											network.layers[layerIndex][netLayerIndex]
													.getInputLayers()[outputLayerInd].getOutputSize());
							outputSize+=outputLayers[outputLayerInd].getOutputSize();
						}
						
						network.layers[layerIndex][netLayerIndex].setOutputLayers(outputLayers);
						
						BLayer[] oldPrevInputLayers=network.layers[layerIndex-1][0].getInputLayers();
						network.layers[layerIndex-1][0].setInputLayers(inputLayers);
						
						SplitNetwork encoderNetwork=new SplitFeedForwardNetwork(
								new BLayer[][]{
									inputLayers,
									new BLayer[]{oldInputLayers[0]},
									new BLayer[]{network.layers[layerIndex][netLayerIndex]}, 
									outputLayers});
						
						MPBackPropGradientDescent bpgd=new MPBackPropGradientDescent(1000, unsupervisedEpochs, learningRate); 
						bpgd.setRegularization(new L2Regularization(outputSize, learningRate, 0.1));
						bpgd
						.trainNetwork(encoderNetwork, 
								savedInputs, 
								layerInputs,
								costFunction);
						network.layers[layerIndex-1][0].setInputLayers(oldPrevInputLayers);
						savedInputs=null;
					}
					network.layers[layerIndex][netLayerIndex].outputLayers=new ArrayList<>();
					network.layers[layerIndex][netLayerIndex].setInputLayers(inputLayers);
					if(!(layerIndex==1 && network.layers[layerIndex][netLayerIndex] instanceof ConvolutionBLayerSparseVector))
					{
						int outputSize=0;
						BLayer[] outputLayers=new BLayer[network.layers[layerIndex][netLayerIndex].getInputLayers().length];
						for(int outputLayerInd=0; outputLayerInd<outputLayers.length; outputLayerInd++)
						{
							outputLayers[outputLayerInd]=null;
							outputLayers[outputLayerInd]
									=new FullyConnectedBLayer(new Sigmoid(), 
											new BLayer[]{network.layers[layerIndex][netLayerIndex]}, 
											network.layers[layerIndex][netLayerIndex]
													.getInputLayers()[outputLayerInd].getOutputSize());
							outputSize+=outputLayers[outputLayerInd].getOutputSize();
						}
						
						network.layers[layerIndex][netLayerIndex].setOutputLayers(outputLayers);
						
						SplitNetwork encoderNetwork=new SplitFeedForwardNetwork(
								new BLayer[][]{
									inputLayers, 
									new BLayer[]{network.layers[layerIndex][netLayerIndex]}, 
									outputLayers});
						MPBackPropGradientDescent bpgd=new MPBackPropGradientDescent(1000, unsupervisedEpochs, learningRate); 
						bpgd.setRegularization(new L2Regularization(outputSize, learningRate, 0.1));
						bpgd
						.trainNetwork(encoderNetwork, 
								layerInputs, 
								layerInputs,
								costFunction);
					}
					else
					{
						savedInputs=layerInputs;
					}
				}
				else
				{
					SplitNetwork encoderNetwork=new SplitFeedForwardNetwork(
							new BLayer[][]{
								inputLayers, 
								new BLayer[]{network.layers[layerIndex][netLayerIndex]}});
					
					MPBackPropGradientDescent bpgd=new MPBackPropGradientDescent(1000, unsupervisedEpochs, learningRate); 
					bpgd.setRegularization(new L2Regularization(network.layers[layerIndex][netLayerIndex].getOutputSize(), learningRate, 0.1));
					bpgd
					.trainNetwork(encoderNetwork, 
							layerInputs, 
							desiredOutputs,
							costFunction);
				}
				
				network.layers[layerIndex][netLayerIndex].setInputLayers(oldInputLayers);
				network.layers[layerIndex][netLayerIndex].setOutputLayers(oldOutputLayers);
				
				layerOutputs.put(network.layers[layerIndex][netLayerIndex], currentLayerOutputs);
			}
		}
		return network;
	}

}
