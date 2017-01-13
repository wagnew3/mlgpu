package learningRule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import costFunctions.CostFunction;
import layer.BLayer;
import layer.ConvolutionBLayerSparseVector;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import network.SplitNetwork;
import regularization.Regularization;

public class MPBackPropGradientDescentNestrovDS extends MPLearningRule 
{
	
	protected int batchSize;
	protected int epochs;
	protected float learningRate;
	protected Regularization regularization=null;
	
	protected Matrix[][] deltas;
	protected Matrix[][][] weights;
	
	protected Matrix[] costFunctionDerivatives;
	protected HashMap<BLayer, Matrix> outputsDerivatives;
	HashMap<BLayer, Matrix>[] outputs;
	protected boolean deltasWeightsPDsinitialized;
	
	protected Matrix[][] totalBiasPDs;
	protected Matrix[][][] totalWeightPDs;
	
	protected float momentumFactor;
	protected Matrix[][] deltasMomentum;
	protected Matrix[][][] weightsMomentum;
	
	int currentEpochG;
	int currentSampleG;
	int currentBatchG;
	
	public MPBackPropGradientDescentNestrovDS(int batchSize, int epochs, float learningRate, float momentumFactor)
	{
		this.batchSize=batchSize;
		this.epochs=epochs;
		this.learningRate=learningRate;
		this.momentumFactor=momentumFactor;
	}
	
	public void setRegularization(Regularization regularization)
	{
		this.regularization=regularization;
	}

	@Override
	public void trainNetwork(SplitNetwork network, Matrix[][] inputs,
			Matrix[][] desiredOutputs, CostFunction costFunction) 
	{		
		totalBiasPDs=new Matrix[network.getLayers().length][];
		deltasMomentum=new Matrix[network.getLayers().length][];
		totalWeightPDs=new Matrix[network.getLayers().length][][];
		weightsMomentum=new Matrix[network.getLayers().length][][];
		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
		{
			totalBiasPDs[layerInd]=new Matrix[network.getLayers()[layerInd].length];
			deltasMomentum[layerInd]=new Matrix[network.getLayers()[layerInd].length];
			totalWeightPDs[layerInd]=new Matrix[network.getLayers()[layerInd].length][];
			weightsMomentum[layerInd]=new Matrix[network.getLayers()[layerInd].length][];
			for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
			{
				totalBiasPDs[layerInd][netInd]=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getBiases().getRows()][1]);
				deltasMomentum[layerInd][netInd]=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getBiases().getRows()][1]);
				totalWeightPDs[layerInd][netInd]=new Matrix[network.getLayers()[layerInd][netInd].getInputLayers().length];
				weightsMomentum[layerInd][netInd]=new Matrix[network.getLayers()[layerInd][netInd].getInputLayers().length];
				for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
				{
					totalWeightPDs[layerInd][netInd][inputNetInd]
							=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getWeights()[0].getRows()]
									[network.getLayers()[layerInd][netInd].getWeights()[0].getCols()]);
					weightsMomentum[layerInd][netInd][inputNetInd]
							=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getWeights()[0].getRows()]
									[network.getLayers()[layerInd][netInd].getWeights()[0].getCols()]);
				}
			}
		}
		
		deltasWeightsPDsinitialized=false;
		List<WeightedData> sortedInputs;
		long time;
		for(int currentEpoch=0; currentEpoch<epochs; currentEpoch++)
		{
			sortedInputs=new ArrayList<>();
			for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
			{
				sortedInputs.add(new WeightedData(new Matrix[][]{inputs[sampleInd], desiredOutputs[sampleInd]}, costFunction.getCost(inputs[sampleInd], network.getOutput(inputs[sampleInd]), desiredOutputs[sampleInd])));
			}
			double[] inputsCDF=new double[sortedInputs.size()];
			Collections.shuffle(sortedInputs);		
			Collections.sort(sortedInputs);
			double sum=0.0;
			for(int sortedInd=0; sortedInd<inputsCDF.length; sortedInd++)
			{
				sum+=sortedInputs.get(sortedInd).weight;
			}
			
			double cdf=0.0;
			for(int sortedInd=0; sortedInd<inputsCDF.length; sortedInd++)
			{
				cdf+=sortedInputs.get(sortedInd).weight;
				inputsCDF[sortedInd]=cdf/sum;
			}
			
			
			for(int sampleInd=0; sampleInd<sortedInputs.size(); sampleInd+=batchSize)
			{
				//Add momentum componant in Nestrov momentum
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
					{
						network.getLayers()[layerInd][netInd].updateBiases(deltasMomentum[layerInd][netInd], -momentumFactor);
						network.getLayers()[layerInd][netInd].updateWeights(weightsMomentum[layerInd][netInd], -momentumFactor);		
					}
				}
				
				Matrix[][] batchInputs=new Matrix[Math.min(batchSize, sortedInputs.size()-sampleInd)][];
				Matrix[][] batchDesiredOutputs=new Matrix[Math.min(batchSize, sortedInputs.size()-sampleInd)][];
				for(int batchInd=0; batchInd<Math.min(batchSize, sortedInputs.size()-sampleInd); batchInd++)
				{
					double random=Math.random();
					int ind=Arrays.binarySearch(inputsCDF, random);
					if(ind<0)
					{
						ind=-(ind+1);
					}
					batchInputs[batchInd]=sortedInputs.get(ind).data[0];
					batchDesiredOutputs[batchInd]=sortedInputs.get(ind).data[1];
					/*
					batchInputs[batchInd]=sortedInputs.get(batchInd+sampleInd).data[0];
					batchDesiredOutputs[batchInd]=sortedInputs.get(batchInd+sampleInd).data[1];
					*/
				}
				
				Object[][] pdss=new Object[Math.min(batchSize, sortedInputs.size()-sampleInd)][];
				for(int batchInd=0; batchInd<Math.min(batchSize, sortedInputs.size()-sampleInd); batchInd++)
				{
					currentEpochG=currentEpoch;
					currentSampleG=sampleInd;
					currentBatchG=batchInd;
					
					Object[] pds=backprop(network, batchInputs[batchInd], batchDesiredOutputs[batchInd], costFunction);
					
					pdss[batchInd]=pds;
					Matrix[][] biasPDs=(Matrix[][])pds[0];
					Matrix[][][] weightPDs=(Matrix[][][])pds[1];
					
					for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
					{
						for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
						{
							if(batchInd==0)
							{
								totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].scal(0, 1, totalBiasPDs[layerInd][netInd]);
							}
							totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omadScale(biasPDs[layerInd][netInd], (float)(1.0/Math.min(batchSize, sortedInputs.size()-sampleInd)));
							for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
							{
								if(batchInd==0)
								{
									totalWeightPDs[layerInd][netInd][inputNetInd]=totalWeightPDs[layerInd][netInd][inputNetInd].scal(0, 1, totalWeightPDs[layerInd][netInd][inputNetInd]);
								}
								totalWeightPDs[layerInd][netInd][inputNetInd]
										=totalWeightPDs[layerInd][netInd][inputNetInd]
												.omadScale(weightPDs[layerInd][netInd][inputNetInd], 
														(float)(1.0/Math.min(batchSize, sortedInputs.size()-sampleInd)));
							}
						}
					}
				}
				
				if(regularization!=null)
				{
					//network=regularization.regularize(network);
				}
				
				int u=0;
				
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
					{
						network.getLayers()[layerInd][netInd].updateBiases(totalBiasPDs[layerInd][netInd], learningRate);
						network.getLayers()[layerInd][netInd].updateWeights(totalWeightPDs[layerInd][netInd], learningRate);		
					}
				}
				
				//update momentum
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
					{
						deltasMomentum[layerInd][netInd].omscal(momentumFactor).omsubScale(totalBiasPDs[layerInd][netInd], learningRate);
						for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
						{
							weightsMomentum[layerInd][netInd][inputNetInd].omscal(momentumFactor).omsubScale(totalWeightPDs[layerInd][netInd][inputNetInd], learningRate);
						}
					}
				}
			}
			
			System.out.println("Epoch: "+currentEpoch);
			
			
			
			if((currentEpoch+1)%5==0)
			{
				double totalError=0.0;
				Matrix[] networkOutput=null;
				for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
				{
					networkOutput=network.getOutput(inputs[sampleInd]);
					//networkOutput=new Matrix[]{((ConvolutionBLayerSparseVector)network.getLayers()[1][0]).getOutputPart(inputs[sampleInd], null, new FDMatrix(network.getLayers()[1][0].getOutputSize(), 1))};
					totalError+=costFunction.getCost(inputs[sampleInd], networkOutput, desiredOutputs[sampleInd]);
				}
				totalError/=inputs.length;
				totalError/=networkOutput[0].getLen();
				System.out.println("totalError: "+totalError);
			}
			
			/*
			double relativeError=0.0;
			for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
			{
				ArrayRealVector networkOutput=network.getOutput(inputs[sampleInd]);
				relativeError+=costFunction.getCost(inputs[sampleInd], networkOutput, desiredOutputs[sampleInd])/networkOutput.getL1Norm();
			}
			relativeError/=inputs.length;
			System.out.println("Relative Error: "+relativeError);
			*/
			
			if(network.layers.length>3)
			{
				//MNISTNumbers.evalNumberNet(network);
			}
			//System.out.println(time);
			//MNISTNumbers.evalNumberNet(network);
			//double output=network.getOutput(inputs[0]).getEntry(0);
			int u=0;
		}
	}
	
	//0=deltas, 1=weightPDs
	protected Object[] backprop(SplitNetwork network, Matrix[] inputs, Matrix[] desiredOutputs, CostFunction costFunction)
	{	
		if(network.layers[1][0].biases.get(0, 0)!=0.0f)
		{
			int u=0;
		}
		if(!deltasWeightsPDsinitialized)
		{
			initializeOutputsOutputsDerivatives(network);
		}
		
		outputs=network.getOutputs(inputs, outputs);
		
		if(network.layers[1][0].biases.get(0, 0)!=0.0f)
		{
			int u=0;
		}
		for(int layerIndex=1; layerIndex<network.getLayers().length; layerIndex++)
		{
			for(int netIndex=0; netIndex<network.getLayers()[layerIndex].length; netIndex++)
			{
				Matrix[] nextLayerInputs=new Matrix[network.getLayers()[layerIndex][netIndex].getInputLayers().length];
				for(int inputInd=0; inputInd<nextLayerInputs.length; inputInd++)
				{
					nextLayerInputs[inputInd]=outputs[layerIndex-1].get(network.getLayers()[layerIndex][netIndex].getInputLayers()[inputInd]);
				}
				
				outputsDerivatives.put(network.getLayers()[layerIndex][netIndex], 
						network.getLayers()[layerIndex][netIndex].getOutputDerivatives(nextLayerInputs, outputsDerivatives.get(network.getLayers()[layerIndex][netIndex])));
			}
		}
		
		if(network.layers[1][0].biases.get(0, 0)!=0.0f)
		{
			int u=0;
		}

		Matrix[] outputsArray=new Matrix[network.getLayers()[network.getLayers().length-1].length];
		for(int netIndex=0; netIndex<outputsArray.length; netIndex++)
		{
			outputsArray[netIndex]=outputs[network.getLayers().length-1].get(network.getLayers()[network.getLayers().length-1][netIndex]);
		}
		
		if(!deltasWeightsPDsinitialized)
		{
			initializeDeltasWeightsPDsCostFunctionDerivatives(network, outputs);
			deltasWeightsPDsinitialized=true;
		}
		
		if(network.layers[1][0].biases.get(0, 0)!=0.0f)
		{
			int u=0;
		}
		
		costFunctionDerivatives=costFunction.getCostDerivative(inputs, outputsArray, desiredOutputs,
				costFunctionDerivatives);
		for(int netIndex=0; netIndex<costFunctionDerivatives.length; netIndex++)
		{
			deltas[deltas.length-1][netIndex]=costFunctionDerivatives[netIndex].oebemult(outputsDerivatives.get(network.getLayers()[network.getLayers().length-1][netIndex]));
			for(int ind=0; ind<deltas[deltas.length-1][netIndex].getLen(); ind++)
			{
				if(deltas[deltas.length-1][netIndex].get(ind, 0)>100)
				{
					int u=0;
				}
			}
			//costFunctionDerivatives[netIndex].clear();
			//outputsDerivatives.get(network.getLayers()[network.getLayers().length-1][netIndex]).clear();
			
			for(int inputInd=0; inputInd<weights[weights.length-1][netIndex].length; inputInd++)
			{
				weights[weights.length-1][netIndex][inputInd]=network.getLayers()[weights.length-1][netIndex]
						.getWeightPDs(outputs[weights.length-2]
								.get(network.getLayers()[network.getLayers().length-1][netIndex].getInputLayers()[inputInd]),
								deltas[weights.length-1][netIndex],
								weights[weights.length-1][netIndex][inputInd]);
				
				/*
				outputs[weights.length-2]
						.get(network.getLayers()[network.getLayers().length-1]
								[netIndex].getInputLayers()[inputInd]).clear();
				*/
			}
		}
		
		if(network.layers[1][0].biases.get(0, 0)!=0.0f)
		{
			int u=0;
		}

		for(int layerIndex=deltas.length-2; layerIndex>0; layerIndex--)
		{
			for(int netIndex=0; netIndex<network.getLayers()[layerIndex].length; netIndex++)
			{
				Matrix[] outputsWeights=new Matrix[network.getLayers()[layerIndex][netIndex].getOutputLayers().length];
				Matrix[] outputsDeltas=new Matrix[network.getLayers()[layerIndex][netIndex].getOutputLayers().length];
				for(int outputInd=0; outputInd<network.getLayers()[layerIndex][netIndex].getOutputLayers().length; outputInd++)
				{
					for(int inputInd=0; inputInd<network.getLayers()[layerIndex][netIndex]
							.getOutputLayers()[outputInd].getInputLayers().length; inputInd++)
					{
						if(network.getLayers()[layerIndex][netIndex].equals(network.getLayers()[layerIndex][netIndex]
							.getOutputLayers()[outputInd].getInputLayers()[inputInd]))
						{
							outputsWeights[outputInd]=network.getLayers()[layerIndex][netIndex]
									.getOutputLayers()[outputInd].getWeights()[inputInd];
							outputsDeltas[outputInd]=deltas[layerIndex+1][netIndexOf(network, network.getLayers()[layerIndex][netIndex]
									.getOutputLayers()[outputInd], layerIndex+1)];
						}
					}
				}
				
				deltas[layerIndex][netIndex]=network.getLayers()[layerIndex][netIndex].getDeltas(outputsWeights, outputsDeltas, outputsDerivatives.get(network.getLayers()[layerIndex][netIndex]), deltas[layerIndex][netIndex]);
				for(int inputInd=0; inputInd<network.getLayers()[layerIndex][netIndex].getInputLayers().length; inputInd++)
				{
					weights[layerIndex][netIndex][inputInd]
							=network.getLayers()[layerIndex][netIndex].getWeightPDs(outputs[layerIndex-1]
									.get(network.getLayers()[layerIndex][netIndex].getInputLayers()[inputInd]), 
									deltas[layerIndex][netIndex],
									weights[layerIndex][netIndex][inputInd]);
				}
				
			}
		}
		
		if(network.layers[1][0].biases.get(0, 0)!=0.0f)
		{
			int u=0;
		}
		
		return new Object[]{deltas, weights};
	}
	
	protected void initializeDeltasWeightsPDsCostFunctionDerivatives(SplitNetwork network, HashMap<BLayer, Matrix>[] outputs)
	{
		deltas=new Matrix[network.getLayers().length][];
		weights=new Matrix[network.getLayers().length][][];
		costFunctionDerivatives=new Matrix[network.getLayers()[network.getLayers().length-1].length];
		deltas[deltas.length-1]=new Matrix[network.getLayers()[network.getLayers().length-1].length];
		weights[weights.length-1]=new Matrix[network.getLayers()[network.getLayers().length-1].length][];
		
		for(int netIndex=0; netIndex<network.getLayers()[network.getLayers().length-1].length; netIndex++)
		{
			weights[weights.length-1][netIndex]=new Matrix[network.getLayers()[network.getLayers().length-1][netIndex].getInputLayers().length];
			costFunctionDerivatives[netIndex]=new FDMatrix(network.getLayers()[network.getLayers().length-1][netIndex].getOutputSize(), 1);
			for(int inputInd=0; inputInd<weights[weights.length-1][netIndex].length; inputInd++)
			{
				int rows=network.getLayers()[network.getLayers().length-1][netIndex].getOutputSize();
				int cols=outputs[weights.length-2].get(network.getLayers()[network.getLayers().length-1][netIndex].getInputLayers()[inputInd]).getRows();
				weights[weights.length-1][netIndex][inputInd]=new FDMatrix(rows, cols);
			}
		}

		for(int layerIndex=deltas.length-2; layerIndex>0; layerIndex--)
		{
			deltas[layerIndex]=new Matrix[network.getLayers()[layerIndex].length];
			weights[layerIndex]=new Matrix[network.getLayers()[layerIndex].length][];
			for(int netIndex=0; netIndex<network.getLayers()[layerIndex].length; netIndex++)
			{
				deltas[layerIndex][netIndex]=new FDMatrix(network.getLayers()[layerIndex][netIndex].getOutputSize(), 1);
				weights[layerIndex][netIndex]=new Matrix[network.getLayers()[layerIndex][netIndex].getInputLayers().length];
				for(int inputInd=0; inputInd<network.getLayers()[layerIndex][netIndex].getInputLayers().length; inputInd++)
				{
					int rows=0;
					int cols=0;
					if(network.getLayers()[layerIndex][netIndex] instanceof ConvolutionBLayerSparseVector)
					{
						rows=network.getLayers()[layerIndex][netIndex].getOutputSize();
						cols=1;
					}
					else
					{
						rows=network.getLayers()[layerIndex][netIndex].getOutputSize();
						cols=outputs[layerIndex-1].get(network.getLayers()[layerIndex][netIndex].getInputLayers()[inputInd]).getRows();
					}
					weights[layerIndex][netIndex][inputInd]=new FDMatrix(rows, cols);
				}
			}
		}
	}
	
	protected void initializeOutputsOutputsDerivatives(SplitNetwork network)
	{
		outputs=new HashMap[network.getLayers().length];
		outputsDerivatives=new HashMap<>();
		
		outputs[0]=new HashMap<>();
		for(int netIndex=0; netIndex<network.getLayers()[0].length; netIndex++)
		{
			outputs[0].put(network.getLayers()[0][netIndex], new FDMatrix(network.getLayers()[0][netIndex].getOutputSize(), 1));
		}
		for(int layerIndex=1; layerIndex<network.getLayers().length; layerIndex++)
		{
			outputs[layerIndex]=new HashMap<>();
			for(int netIndex=0; netIndex<network.getLayers()[layerIndex].length; netIndex++)
			{
				outputs[layerIndex].put(network.getLayers()[layerIndex][netIndex], new FDMatrix(network.getLayers()[layerIndex][netIndex].getOutputSize(), 1));
				outputsDerivatives.put(network.getLayers()[layerIndex][netIndex], 
						new FDMatrix(network.getLayers()[layerIndex][netIndex].getOutputSize(), 1));
			}
		}
	}
	
	protected int netIndexOf(SplitNetwork network, BLayer layer, int layerInd)
	{
		for(int ind=0; ind<network.getLayers()[layerInd].length; ind++)
		{
			if(layer.equals(network.getLayers()[layerInd][ind]))
			{
				return ind;
			}
		}
		return -1;
	}
}

class WeightedData implements Comparable<WeightedData>
{
	
	Matrix[][] data;
	float weight;
	
	public WeightedData(Matrix[][] data, float weight)
	{
		this.data=data;
		this.weight=weight;
	}

	@Override
	public int compareTo(WeightedData o)
	{
		return (int)Math.signum(o.weight-weight);
	}
	
}
