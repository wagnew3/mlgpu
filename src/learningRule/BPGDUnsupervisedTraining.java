package learningRule;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.Sigmoid;
import checkers.CheckersF1Error;
import costFunctions.CostFunction;
import costFunctions.EuclideanDistanceCostFunction;
import evaluationFunctions.EvaluationFunction;
import layer.BInputLayer;
import layer.BLayer;
import layer.ConvolutionBLayerSparseVector;
import layer.FullyConnectedBLayer;
import layer.SparseArrayRealVector;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import nDimensionalMatrices.SparseFMatrix;
import network.SplitFeedForwardNetwork;
import network.SplitNetwork;
import regularization.L2Regularization;
import validation.NoValidation;
import validation.Validator;

public class BPGDUnsupervisedTraining extends MPBackPropGradientDescent
{
	
	public MPLearningRule backprop;

	public BPGDUnsupervisedTraining(MPLearningRule backprop) 
	{
		super(-1, -1, -1f);
		this.backprop=backprop;
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
						
						backprop.trainNetwork(encoderNetwork, 
								savedInputs, 
								layerInputs,
								costFunction,
								new NoValidation(null, null));
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
						
						backprop.trainNetwork(encoderNetwork, 
								layerInputs, 
								layerInputs,
								costFunction,
								new NoValidation(null, null));
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
					backprop.trainNetwork(encoderNetwork, 
							layerInputs, 
							desiredOutputs,
							costFunction,
							new NoValidation(null, null));
				}
				
				network.layers[layerIndex][netLayerIndex].setInputLayers(oldInputLayers);
				network.layers[layerIndex][netLayerIndex].setOutputLayers(oldOutputLayers);
				
				layerOutputs.clear();
				layerOutputs.put(network.layers[layerIndex][netLayerIndex], currentLayerOutputs);
			}
		}
		return network;
	}
	
	@Override
	public void trainNetwork(SplitNetwork network, Matrix[][] inputs,
			Matrix[][] desiredOutputs, CostFunction costFunction, Validator validator) 
	{	
		backprop.trainNetwork(network, 
				inputs, 
				desiredOutputs,
				costFunction,
				validator);
	}

}
