package learningRule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import jcuda.CudaException;
import costFunctions.CostFunction;
import layer.BLayer;
import layer.ConvolutionBLayerSparseVector;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import network.SplitNetwork;
import regularization.Regularization;
import test.MNISTNumbers;
import validation.Validator;

public class RProp extends MPLearningRule
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
	
	protected Matrix[][] prevTotalBiasPDs;
	protected Matrix[][][] prevTotalWeightPDs;
	
	protected Matrix[][] rPropBiasDeltaChanges;
	protected Matrix[][][] rPropWeightDeltaChanges;
	
	protected Matrix[][] rPropBiasDeltas;
	protected Matrix[][][] rPropWeightDeltas;
	
	int currentEpochG;
	int currentSampleG;
	int currentBatchG;
	
	float np=1.2f;
	float nm=0.5f;
	
	float maxDelta=50.0f;
	float minDelta=0.000001f;
	
	public RProp(int batchSize, int epochs, float learningRate)
	{
		this.batchSize=batchSize;
		this.epochs=epochs;
		this.learningRate=learningRate;
	}
	
	public void setRegularization(Regularization regularization)
	{
		this.regularization=regularization;
	}

	@Override
	public void trainNetwork(SplitNetwork network, Matrix[][] inputs,
			Matrix[][] desiredOutputs, CostFunction costFunction, Validator validator) 
	{		
		totalBiasPDs=new Matrix[network.getLayers().length][];
		totalWeightPDs=new Matrix[network.getLayers().length][][];
		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
		{
			totalBiasPDs[layerInd]=new Matrix[network.getLayers()[layerInd].length];
			totalWeightPDs[layerInd]=new Matrix[network.getLayers()[layerInd].length][];
			for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
			{
				totalBiasPDs[layerInd][netInd]=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getBiases().getRows()][1]);
				totalWeightPDs[layerInd][netInd]=new Matrix[network.getLayers()[layerInd][netInd].getInputLayers().length];
				for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
				{
					totalWeightPDs[layerInd][netInd][inputNetInd]
							=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getRows()]
									[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getCols()]);
				}
			}
		}
		
		prevTotalBiasPDs=new Matrix[network.getLayers().length][];
		prevTotalWeightPDs=new Matrix[network.getLayers().length][][];
		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
		{
			prevTotalBiasPDs[layerInd]=new Matrix[network.getLayers()[layerInd].length];
			prevTotalWeightPDs[layerInd]=new Matrix[network.getLayers()[layerInd].length][];
			for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
			{
				prevTotalBiasPDs[layerInd][netInd]=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getBiases().getRows()][1]);
				prevTotalWeightPDs[layerInd][netInd]=new Matrix[network.getLayers()[layerInd][netInd].getInputLayers().length];
				for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
				{
					prevTotalWeightPDs[layerInd][netInd][inputNetInd]
							=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getRows()]
									[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getCols()]);
				}
			}
		}
		
		rPropBiasDeltas=new Matrix[network.getLayers().length][];
		rPropWeightDeltas=new Matrix[network.getLayers().length][][];
		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
		{
			rPropBiasDeltas[layerInd]=new Matrix[network.getLayers()[layerInd].length];
			rPropWeightDeltas[layerInd]=new Matrix[network.getLayers()[layerInd].length][];
			for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
			{
				rPropBiasDeltas[layerInd][netInd]=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getBiases().getRows()][1]);
				rPropWeightDeltas[layerInd][netInd]=new Matrix[network.getLayers()[layerInd][netInd].getInputLayers().length];
				for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
				{
					rPropWeightDeltas[layerInd][netInd][inputNetInd]
							=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getRows()]
									[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getCols()]);
				}
			}
		}
		
		rPropBiasDeltaChanges=new Matrix[network.getLayers().length][];
		rPropWeightDeltaChanges=new Matrix[network.getLayers().length][][];
		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
		{
			rPropBiasDeltaChanges[layerInd]=new Matrix[network.getLayers()[layerInd].length];
			rPropWeightDeltaChanges[layerInd]=new Matrix[network.getLayers()[layerInd].length][];
			for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
			{
				rPropBiasDeltaChanges[layerInd][netInd]=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getBiases().getRows()][1]);			
				for(int row=0; row<rPropBiasDeltaChanges[layerInd][netInd].getRows(); row++)
				{
					for(int col=0; col<rPropBiasDeltaChanges[layerInd][netInd].getCols(); col++)
					{
						rPropBiasDeltaChanges[layerInd][netInd].set(row, col, 0.1f);
					}
				}
				rPropWeightDeltaChanges[layerInd][netInd]=new Matrix[network.getLayers()[layerInd][netInd].getInputLayers().length];
				for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
				{
					rPropWeightDeltaChanges[layerInd][netInd][inputNetInd]
							=new FDMatrix(new float[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getRows()]
									[network.getLayers()[layerInd][netInd].getWeights()[inputNetInd].getCols()]);
					for(int row=0; row<rPropWeightDeltaChanges[layerInd][netInd][inputNetInd].getRows(); row++)
					{
						for(int col=0; col<rPropWeightDeltaChanges[layerInd][netInd][inputNetInd].getCols(); col++)
						{
							rPropWeightDeltaChanges[layerInd][netInd][inputNetInd].set(row, col, 0.1f);
						}
					}
				}
			}
		}
		
		deltasWeightsPDsinitialized=false;
		List<Matrix[][]> randomizedInputs=new ArrayList<Matrix[][]>();
		for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
		{
			randomizedInputs.add(new Matrix[][]{inputs[sampleInd], desiredOutputs[sampleInd]});
		}
		long time;
		for(int currentEpoch=0; currentEpoch<epochs; currentEpoch++)
		{
		    time=System.nanoTime();
			Collections.shuffle(randomizedInputs);
			//System.out.println("unrandomized inputs");
						
			for(int sampleInd=0; sampleInd<randomizedInputs.size(); sampleInd+=batchSize)
			{
				Matrix[][] batchInputs=new Matrix[Math.min(batchSize, randomizedInputs.size()-sampleInd)][];
				Matrix[][] batchDesiredOutputs=new Matrix[Math.min(batchSize, randomizedInputs.size()-sampleInd)][];
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedInputs.size()-sampleInd); batchInd++)
				{
					batchInputs[batchInd]=randomizedInputs.get(batchInd+sampleInd)[0];
					batchDesiredOutputs[batchInd]=randomizedInputs.get(batchInd+sampleInd)[1];
				}
				
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedInputs.size()-sampleInd); batchInd++)
				{
					currentEpochG=currentEpoch;
					currentSampleG=sampleInd;
					currentBatchG=batchInd;
					
					Object[] pds=backprop(network, batchInputs[batchInd], batchDesiredOutputs[batchInd], costFunction);
					
					Matrix[][] biasPDs=(Matrix[][])pds[0];
					Matrix[][][] weightPDs=(Matrix[][][])pds[1];
					for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
					{
						for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
						{
							if(batchInd==0)
							{
								totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omscal(0);
							}
							totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omadScale(biasPDs[layerInd][netInd], (float)(1.0/Math.min(batchSize, randomizedInputs.size()-sampleInd)));
							for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
							{
								if(batchInd==0)
								{
									totalWeightPDs[layerInd][netInd][inputNetInd]=totalWeightPDs[layerInd][netInd][inputNetInd].omscal(0);
								}
								totalWeightPDs[layerInd][netInd][inputNetInd]
										=totalWeightPDs[layerInd][netInd][inputNetInd]
												.omadScale(weightPDs[layerInd][netInd][inputNetInd], 
														(float)(1.0/Math.min(batchSize, randomizedInputs.size()-sampleInd)));
							}
						}
					}
				}
				
				int u=0;
				
				if(regularization!=null)
				{
					//network=regularization.regularize(network);
				}
				
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
					{
						rProp(prevTotalBiasPDs[layerInd][netInd], totalBiasPDs[layerInd][netInd],
								rPropBiasDeltaChanges[layerInd][netInd], rPropBiasDeltas[layerInd][netInd]);
						network.getLayers()[layerInd][netInd].updateBiases(rPropBiasDeltas[layerInd][netInd], 1.0f);
						prevTotalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].copyTo(prevTotalBiasPDs[layerInd][netInd]);
						for(int inputLayerInd=0; 
								inputLayerInd<network.getLayers()[layerInd][netInd].getInputLayers().length;
								inputLayerInd++)
						{
							rProp(prevTotalWeightPDs[layerInd][netInd][inputLayerInd], totalWeightPDs[layerInd][netInd][inputLayerInd],
									rPropWeightDeltaChanges[layerInd][netInd][inputLayerInd], rPropWeightDeltas[layerInd][netInd][inputLayerInd]);
							prevTotalWeightPDs[layerInd][netInd][inputLayerInd]
									=totalWeightPDs[layerInd][netInd][inputLayerInd].copyTo(rPropWeightDeltas[layerInd][netInd][inputLayerInd]);
						}
						network.getLayers()[layerInd][netInd].updateWeights(totalWeightPDs[layerInd][netInd], 1.0f);
					}
				}
			}
			
			System.out.println("Epoch: "+currentEpoch);
			
			
			
			if((currentEpoch+1)%5==0)
			{
				double totalError=0.0;
				Matrix[] networkOutput=null;
				int outLen=0;
				for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
				{
					networkOutput=network.getOutput(inputs[sampleInd]);
					//networkOutput=new Matrix[]{((ConvolutionBLayerSparseVector)network.getLayers()[1][0]).getOutputPart(inputs[sampleInd], null, new FDMatrix(network.getLayers()[1][0].getOutputSize(), 1))};
					try
					{
						totalError+=costFunction.getCost(inputs[sampleInd], networkOutput, desiredOutputs[sampleInd]);
					}
					catch(CudaException e)
					{
						e.printStackTrace();
					}
					outLen=networkOutput[0].getLen();
					for(Matrix mat: networkOutput)
					{
						mat.clear();
					}
				}
				totalError/=inputs.length;
				totalError/=networkOutput[0].getLen();
				System.out.println("average error: "+totalError);
				
				double validationAccuracy=validator.validate(network);
				System.out.println("Validation accuracy: "+validationAccuracy);
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
			
			if(true)
			{
				//MNISTNumbers.evalNumberNet(network);
			}
			time=System.nanoTime()-time;
			//System.out.println(time);
			//MNISTNumbers.evalNumberNet(network);
			//double output=network.getOutput(inputs[0]).getEntry(0);
			int u=0;
		}
	}
	
	protected Matrix rProp(Matrix prevDers, Matrix curDers, Matrix deltasChanges, Matrix deltas)
	{
		for(int devRow=0; devRow<prevDers.getRows(); devRow++)
		{
			for(int devCol=0; devCol<prevDers.getCols(); devCol++)
			{
				float devProd=Math.signum(prevDers.get(devRow, devCol))*Math.signum(curDers.get(devRow, devCol));
				if(devProd>0)
				{
					deltasChanges.set(devRow, devCol, Math.min(deltasChanges.get(devRow, devCol)*np, maxDelta));
					deltas.set(devRow, devCol, Math.signum(curDers.get(devRow, devCol))*deltasChanges.get(devRow, devCol));
				}
				else if(devProd<0)
				{
					deltasChanges.set(devRow, devCol, Math.max(deltasChanges.get(devRow, devCol)*nm, minDelta));
					deltas.set(devRow, devCol, -1.0f*deltas.get(devRow, devCol));
					curDers.set(devRow, devCol, 0.0f);
				}
				else
				{
					deltas.set(devRow, devCol, Math.signum(curDers.get(devRow, devCol))*deltasChanges.get(devRow, devCol));
				}
			}
		}
		return deltas;
	}
	
	//0=deltas, 1=weightPDs
	protected Object[] backprop(SplitNetwork network, Matrix[] inputs, Matrix[] desiredOutputs, CostFunction costFunction)
	{	
		if(!deltasWeightsPDsinitialized)
		{
			initializeOutputsOutputsDerivatives(network);
		}
		
		outputs=network.getOutputs(inputs, outputs);
		
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
		
		costFunctionDerivatives=costFunction.getCostDerivative(inputs, outputsArray, desiredOutputs,
				costFunctionDerivatives);
		for(int netIndex=0; netIndex<costFunctionDerivatives.length; netIndex++)
		{
			deltas[deltas.length-1][netIndex]=costFunctionDerivatives[netIndex].oebemult(outputsDerivatives.get(network.getLayers()[network.getLayers().length-1][netIndex]));
			
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