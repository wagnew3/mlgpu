package learningRule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import jcuda.CudaException;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import network.SplitNetwork;
import test.MNISTNumbers;
import validation.Validator;
import costFunctions.CostFunction;

public class RPropMultithreaded extends RProp
{

	int numberThreads=6;
	BackpropThread[] bpThreads;
	
	float initialDeltaValue=0.1f;
	
	public RPropMultithreaded(int batchSize, int epochs, float learningRate)
	{
		super(batchSize, epochs, learningRate);
	}
	
	@Override
	public void trainNetwork(SplitNetwork network, Matrix[][] inputs,
			Matrix[][] desiredOutputs, CostFunction costFunction, Validator validator) 
	{
		bpThreads=new BackpropThread[numberThreads];
		for(int tInd=0; tInd<numberThreads; tInd++)
		{
			bpThreads[tInd]=new BackpropThread(network, costFunction);
		}
		
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
						rPropBiasDeltaChanges[layerInd][netInd].set(row, col, initialDeltaValue);
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
							rPropWeightDeltaChanges[layerInd][netInd][inputNetInd].set(row, col, initialDeltaValue);
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
				List<Matrix[][]> batchSamples=new ArrayList<>();
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedInputs.size()-sampleInd); batchInd++)
				{
					batchSamples.add(new Matrix[][]{randomizedInputs.get(batchInd+sampleInd)[0],
							randomizedInputs.get(batchInd+sampleInd)[1]});
				}
				BackpropThread.setBatchSize(batchSamples.size());
				BackpropThread.addSamples(batchSamples);
				
				for(int tInd=0; tInd<numberThreads; tInd++)
				{
					bpThreads[tInd].resetMats();
				}
				
				int maxSleepCount=1000;
				int sleepCount=0;
				while(!BackpropThread.isEmpty() && sleepCount<maxSleepCount)
				{
					try
					{
						Thread.sleep(1);
					} 
					catch (InterruptedException e) 
					{
						e.printStackTrace();
					}
					//sleepCount++;
				}
				if(sleepCount>=maxSleepCount)
				{
					for(int bpThreadInd=0; bpThreadInd<bpThreads.length; bpThreadInd++)
					{
						bpThreads[bpThreadInd].run=false;
					}
					for(int tInd=0; tInd<numberThreads; tInd++)
					{
						bpThreads[tInd]=new BackpropThread(network, costFunction);
					}
					continue;
				}
				
				for(int bpThreadInd=0; bpThreadInd<bpThreads.length; bpThreadInd++)
				{
					currentEpochG=currentEpoch;
					currentSampleG=sampleInd;
					currentBatchG=bpThreadInd;
				
					Matrix[][] biasPDs=bpThreads[bpThreadInd].totalBiasPDs;
					Matrix[][][] weightPDs=bpThreads[bpThreadInd].totalWeightPDs;
					for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
					{
						for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
						{
							if(bpThreadInd==0)
							{
								totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omscal(0);
							}
							totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omad(biasPDs[layerInd][netInd]);
							for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
							{
								if(bpThreadInd==0)
								{
									totalWeightPDs[layerInd][netInd][inputNetInd]=totalWeightPDs[layerInd][netInd][inputNetInd].omscal(0);
								}
								totalWeightPDs[layerInd][netInd][inputNetInd]
										=totalWeightPDs[layerInd][netInd][inputNetInd]
												.omad(weightPDs[layerInd][netInd][inputNetInd]);
							}
						}
					}
					
					for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
					{
						for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
						{
							totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].omscal((float)(1.0/batchSize));
							for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
							{
								totalWeightPDs[layerInd][netInd][inputNetInd]
										=totalWeightPDs[layerInd][netInd][inputNetInd].omscal((float)(1.0/batchSize));
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
			
			if((currentEpoch+1)%10==0)
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
		for(int bpThreadInd=0; bpThreadInd<bpThreads.length; bpThreadInd++)
		{
			bpThreads[bpThreadInd].run=false;
		}
		int u=0;
	}

}
