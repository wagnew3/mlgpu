package learningRule;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import costFunctions.CostFunction;
import layer.BLayer;
import layer.ConvolutionBLayerSparseVector;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import network.SplitNetwork;

public class BackpropThread extends Thread
{
	
	static volatile List<Matrix[][]> traingSamplesToBackprop=new ArrayList<>();
	
	protected Matrix[][] deltas;
	protected Matrix[][][] weights;
	protected Matrix[] costFunctionDerivatives;
	protected HashMap<BLayer, Matrix> outputsDerivatives;
	HashMap<BLayer, Matrix>[] outputs;
	public boolean deltasWeightsPDsinitialized;
	
	public volatile boolean run;
	
	protected Matrix[][] totalBiasPDs;
	protected Matrix[][][] totalWeightPDs;
	
	SplitNetwork network;
	CostFunction costFunction;
	
	static volatile int batchSize;
	
	public BackpropThread(SplitNetwork network, CostFunction costFunction)
	{
		run=true;
		this.network=network;
		this.costFunction=costFunction;
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
		start();
	}
	
	@Override
	public void run()
	{
		while(run)
		{
			Matrix[][] sample=getNextSample();
			if(sample!=null)
			{
				Object[] pds=backprop(network, sample[0], sample[1], costFunction);
				Matrix[][] biasPDs=(Matrix[][])pds[0];
				Matrix[][][] weightPDs=(Matrix[][][])pds[1];
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
					{
						totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omad(biasPDs[layerInd][netInd]);
						for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
						{
							totalWeightPDs[layerInd][netInd][inputNetInd]
									=totalWeightPDs[layerInd][netInd][inputNetInd]
											.omad(weightPDs[layerInd][netInd][inputNetInd]);
						}
					}
				}
			}
			else
			{
				try
				{
					Thread.sleep(1);
				} 
				catch (InterruptedException e1) 
				{
					e1.printStackTrace();
				}
			}
		}
	}
	
	public static boolean isEmpty()
	{
		synchronized(traingSamplesToBackprop)
		{
			return traingSamplesToBackprop.isEmpty();
		}
	}
	
	public static void setBatchSize(int newBatchSize)
	{
		batchSize=newBatchSize;
	}
	
	public volatile boolean finished=true;
	
	public static Matrix[][] getNextSample()
	{
		synchronized(traingSamplesToBackprop)
		{
			if(!traingSamplesToBackprop.isEmpty())
			{
				return traingSamplesToBackprop.remove(0);
			}
			else
			{
				return null;
			}
		}
	}
	
	public static void addSamples(List<Matrix[][]> samples)
	{
		synchronized(traingSamplesToBackprop)
		{
			traingSamplesToBackprop.addAll(samples);
		}	
	}
	
	public void resetMats()
	{
		finished=false;
		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
		{
			for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
			{
				totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omscal(0);
				for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
				{
					totalWeightPDs[layerInd][netInd][inputNetInd]=totalWeightPDs[layerInd][netInd][inputNetInd].omscal(0);
				}
			}
		}
		deltasWeightsPDsinitialized=false;
	}
	
	public Matrix[][] getTotalBiasPDs()
	{
		return totalBiasPDs;
	}
	
	public Matrix[][][] getTotalWeightPDs()
	{
		return totalWeightPDs;
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
		
		for(int layerIndex=deltas.length-2; layerIndex>0; layerIndex--)
		{
			for(int netIndex=0; netIndex<network.getLayers()[layerIndex].length; netIndex++)
			{
				for(int deltaInd=0; deltaInd<deltas[layerIndex][netIndex].getLen(); deltaInd++)
				{
					if(!Float.isFinite(deltas[layerIndex][netIndex].get(deltaInd, 0)))
					{
						int y=0;
					}
				}
				for(int inputInd=0; inputInd<network.getLayers()[layerIndex][netIndex].getInputLayers().length; inputInd++)
				{
					for(int deltaInd=0; deltaInd<weights[layerIndex][netIndex][inputInd].getLen(); deltaInd++)
					{
						if(!Float.isFinite(weights[layerIndex][netIndex][inputInd].get(deltaInd, 0)))
						{
							int y=0;
						}
					}
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
