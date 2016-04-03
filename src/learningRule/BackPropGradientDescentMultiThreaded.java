package learningRule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import costFunctions.CostFunction;
import network.Network;
import test.MNISTNumbers;

public class BackPropGradientDescentMultiThreaded extends BackPropGradientDescent
{

    protected int numberWorkers;
    protected double relativeError;
    protected double maxRelativeError;
    
    private ArrayRealVector[] totalBiasPDs;
    private BlockRealMatrix[] totalWeightPDs;
    
    public BackPropGradientDescentMultiThreaded(int batchSize, int maxEpochs, 
    		double learningRate, int numberWorkers, double maxRelativeError) 
    {
		super(batchSize, maxEpochs, learningRate);
		this.numberWorkers=numberWorkers;
		this.maxRelativeError=maxRelativeError;
    }
    
    	@Override
	public void trainNetwork(Network network, ArrayRealVector[] inputs,
			ArrayRealVector[] desiredOutputs, CostFunction costFunction) 
	{
		List<ArrayRealVector[]> randomizedInputs=new ArrayList<ArrayRealVector[]>();
		for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
		{
			randomizedInputs.add(new ArrayRealVector[]{inputs[sampleInd], desiredOutputs[sampleInd]});
		}
		
		/*
		long time;
		for(int currentEpoch=0; currentEpoch<epochs; currentEpoch++)
		{
		    time=System.nanoTime();
			Collections.shuffle(randomizedInputs);
			
			for(int sampleInd=0; sampleInd<randomizedInputs.size(); sampleInd+=batchSize)
			{
				ArrayRealVector[] batchInputs=new ArrayRealVector[Math.min(batchSize, randomizedInputs.size()-sampleInd)];
				ArrayRealVector[] batchDesiredOutputs=new ArrayRealVector[Math.min(batchSize, randomizedInputs.size()-sampleInd)];
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedInputs.size()-sampleInd); batchInd++)
				{
					batchInputs[batchInd]=randomizedInputs.get(batchInd+sampleInd)[0];
					batchDesiredOutputs[batchInd]=randomizedInputs.get(batchInd+sampleInd)[1];
				}
				
				ArrayRealVector[] totalBiasPDs=new ArrayRealVector[network.getLayers().length];
				BlockRealMatrix[] totalWeightPDs=new BlockRealMatrix[network.getLayers().length];
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					totalBiasPDs[layerInd]=new ArrayRealVector(network.getLayers()[layerInd].getBiases().getDimension());
					totalWeightPDs[layerInd]=new BlockRealMatrix(network.getLayers()[layerInd].getWeights().getRowDimension(),
							network.getLayers()[layerInd].getWeights().getColumnDimension());
				}
				
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedInputs.size()-sampleInd); batchInd++)
				{
					Object[] pds=backprop(network, batchInputs[batchInd], batchDesiredOutputs[batchInd], costFunction);
					ArrayRealVector[] biasPDs=(ArrayRealVector[])pds[0];
					RealMatrix[] weightPDs=(RealMatrix[])pds[1];
					for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
					{
						try
						{
						totalBiasPDs[layerInd]=totalBiasPDs[layerInd].add(biasPDs[layerInd]);
						}
						catch(Exception e)
						{
							e.printStackTrace();
						}
						totalWeightPDs[layerInd]=totalWeightPDs[layerInd].add(weightPDs[layerInd]);
					}
				}
				
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					network.getLayers()[layerInd].updateBiases(totalBiasPDs[layerInd], learningRate);
					network.getLayers()[layerInd].updateWeights(totalWeightPDs[layerInd], learningRate);
				}
				//MNISTNumbers.evalNumberNet(network);
			}
			
			time=System.nanoTime()-time;
			//System.out.println(time);
			//MNISTNumbers.evalNumberNet(network);
			//double output=network.getOutput(inputs[0]).getEntry(0);
			//int u=0;
		}
		*/
		
		relativeError=Double.MAX_VALUE;
		
		BackPropWorker[] workers=new BackPropWorker[numberWorkers];
		for(int workerInd=0; workerInd<workers.length; workerInd++)
		{
			List<ArrayRealVector[]> inputOutputs=new ArrayList<>();
			for(int listInd=0; listInd<randomizedInputs.size(); listInd++)
			{
				inputOutputs.add(new ArrayRealVector[]{randomizedInputs.get(listInd)[0].copy(), randomizedInputs.get(listInd)[1].copy()});
			}
			Collections.shuffle(inputOutputs);
			workers[workerInd]=new BackPropWorker(inputOutputs, 
					costFunction, batchSize,
				    this, network);
		}
		
		//updateWorkerNetworks(workers, network);
		
		int numberEpochs=0;
		while(relativeError>maxRelativeError)
		{
			/*
			startEpoch(workers);
			
			long time=System.nanoTime();
			
			while(!allFinishedEpoch(workers))
			{
				totalBiasPDs=new ArrayRealVector[network.getLayers().length];
				totalWeightPDs=new BlockRealMatrix[network.getLayers().length];
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					totalBiasPDs[layerInd]=new ArrayRealVector(network.getLayers()[layerInd].getBiases().getDimension());
					totalWeightPDs[layerInd]=new BlockRealMatrix(network.getLayers()[layerInd].getWeights().getRowDimension(),
							network.getLayers()[layerInd].getWeights().getColumnDimension());
				}
				
				updateWorkerNetworks(workers, network);
				startBatch(workers);
				while(!allFinishedBatch(workers)
						&& !allFinishedEpoch(workers))
				{
					
					try 
					{
						Thread.sleep(1);
					} 
					catch (InterruptedException e)
					{
						e.printStackTrace();
					}
					
				}
				
				if(allFinishedEpoch(workers))
				{
					break;
				}
				
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					network.getLayers()[layerInd].updateBiases(totalBiasPDs[layerInd], learningRate);
					network.getLayers()[layerInd].updateWeights(totalWeightPDs[layerInd], learningRate);
				}
			}
			
			time=System.nanoTime()-time;
			System.out.println("Batch Time "+time);
			*/
			
			try 
			{
				Thread.sleep(5000);
			} 
			catch (InterruptedException e) 
			{
				e.printStackTrace();
			}
			
	    	relativeError=0;
	    	for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
    		{
				ArrayRealVector networkOutput=network.getOutput(inputs[sampleInd]);
				relativeError+=costFunction.getCost(networkOutput, desiredOutputs[sampleInd])/desiredOutputs[sampleInd].getL1Norm();
			}
			relativeError/=inputs.length;
			
			MNISTNumbers.evalNumberNet(network);
			System.out.println("Relative Error: "+relativeError);
			
			numberEpochs++;
			if(numberEpochs==epochs)
			{
				relativeError=0;
				break;
			}
		}
	}
    	
	protected synchronized void addToBiasChanges(ArrayRealVector[] workerBiasPDs)
	{
		for(int layerInd=1; layerInd<totalBiasPDs.length; layerInd++)
	    {
			totalBiasPDs[layerInd]=totalBiasPDs[layerInd].add(workerBiasPDs[layerInd]);
	    }
	}
	
	protected synchronized void addToWeightChanges(BlockRealMatrix[] workerWeightPDs)
	{
	    for(int layerInd=1; layerInd<workerWeightPDs.length; layerInd++)
	    {
	    	totalWeightPDs[layerInd]=totalWeightPDs[layerInd].add(workerWeightPDs[layerInd]);
	    }
	}
	
	protected synchronized void updateWeightsBiases(Network network, ArrayRealVector[] workerBiasPDs, BlockRealMatrix[] workerWeightPDs)
	{
		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
		{
			network.getLayers()[layerInd].updateBiases(workerBiasPDs[layerInd], learningRate);
			network.getLayers()[layerInd].updateWeights(workerWeightPDs[layerInd], learningRate);
		}
	}
	
	protected boolean allFinishedBatch(BackPropWorker[] workers)
	{
		for(BackPropWorker worker: workers)
		{
			if(!worker.finishedBatch)
			{
				return false;
			}
		}
		return true;
	}
	
	protected void startBatch(BackPropWorker[] workers)
	{
		for(BackPropWorker worker: workers)
		{
			worker.finishedBatch=false;
		}
	}
	
	protected boolean allFinishedEpoch(BackPropWorker[] workers)
	{
		for(BackPropWorker worker: workers)
		{
			if(!worker.finishedEpoch)
			{
				return false;
			}
		}
		return true;
	}
	
	protected void startEpoch(BackPropWorker[] workers)
	{
		for(BackPropWorker worker: workers)
		{
			worker.finishedEpoch=false;
		}
	}
	
	protected void updateWorkerNetworks(BackPropWorker[] workers, Network network)
	{
		for(BackPropWorker worker: workers)
		{
			worker.setNetwork(network.clone());
		}
	}

}

class BackPropWorker implements Runnable
{
    
    List<ArrayRealVector[]> inputsOutputs;
    CostFunction costFunction;
    int batchSize;
    BackPropGradientDescentMultiThreaded backProp;
    Network centralNetwork;
    
    volatile boolean finishedBatch;
    volatile boolean finishedEpoch;
    
    public BackPropWorker(List<ArrayRealVector[]> inputsOutputs, CostFunction costFunction, int batchSize,
	    BackPropGradientDescentMultiThreaded backProp, Network centralNetwork)
    {
		this.inputsOutputs=inputsOutputs;
		this.costFunction=costFunction;
		this.batchSize=batchSize;
		this.backProp=backProp;
		this.centralNetwork=centralNetwork;
		finishedBatch=true;
		finishedEpoch=true;
		new Thread(this).start();
    }
    
    @Override
    public void run() 
    {
    	/*
    	while(finishedBatch)
    	{
    		try 
    		{
				Thread.sleep(1);
			} 
    		catch (InterruptedException e) 
    		{
				e.printStackTrace();
			}
    	}
    	*/
		while(backProp.relativeError>backProp.maxRelativeError)
		{
        	Collections.shuffle(inputsOutputs);
        	long time=System.nanoTime();
        	for(int sampleInd=0; sampleInd<inputsOutputs.size(); sampleInd+=batchSize)
        	{
        		Network network=centralNetwork.clone();
        		
        		
        		ArrayRealVector[] batchInputs=new ArrayRealVector[Math.min(batchSize, inputsOutputs.size()-sampleInd)];
        		ArrayRealVector[] batchDesiredOutputs=new ArrayRealVector[Math.min(batchSize, inputsOutputs.size()-sampleInd)];
        		
        		
        		/*
        		for(int batchInd=0; batchInd<batchSize; batchInd++)
        		{
        			batchInputs[batchInd]=inputsOutputs.get((int)(Math.random()*inputsOutputs.size()))[0];
        			batchDesiredOutputs[batchInd]=inputsOutputs.get((int)(Math.random()*inputsOutputs.size()))[1];
        		}
        		*/
        		
        		
        		for(int batchInd=0; batchInd<Math.min(batchSize, inputsOutputs.size()-sampleInd); batchInd++)
        		{
        			batchInputs[batchInd]=inputsOutputs.get(batchInd+sampleInd)[0];
        			batchDesiredOutputs[batchInd]=inputsOutputs.get(batchInd+sampleInd)[1];
        		}
        		
        		/*
        		int randomStart=(int)(Math.random()*inputsOutputs.size());
        		for(int batchInd=0; batchInd<batchSize; batchInd++)
        		{
        			batchInputs[batchInd]=inputsOutputs.get((randomStart+batchInd)%inputsOutputs.size())[0];
        			batchDesiredOutputs[batchInd]=inputsOutputs.get((randomStart+batchInd)%inputsOutputs.size())[1];
        		}
        		*/
        		
        		ArrayRealVector[] totalBiasPDs=new ArrayRealVector[network.getLayers().length];
        		BlockRealMatrix[] totalWeightPDs=new BlockRealMatrix[network.getLayers().length];
        		for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
        		{
        			totalBiasPDs[layerInd]=new ArrayRealVector(network.getLayers()[layerInd].getBiases().getDimension());
        			totalWeightPDs[layerInd]=new BlockRealMatrix(network.getLayers()[layerInd].getWeights().getRowDimension(),
        					network.getLayers()[layerInd].getWeights().getColumnDimension());
        		}
        		
        		for(int batchInd=0; batchInd<Math.min(batchSize, inputsOutputs.size()-sampleInd); batchInd++)
        		{
        			Object[] pds=backProp.backprop(network, batchInputs[batchInd], batchDesiredOutputs[batchInd], costFunction);
        			ArrayRealVector[] biasPDs=(ArrayRealVector[])pds[0];
        			RealMatrix[] weightPDs=(RealMatrix[])pds[1];
        			for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
        			{
        				try
        				{
        				totalBiasPDs[layerInd]=totalBiasPDs[layerInd].add(biasPDs[layerInd]);
        				}
        				catch(Exception e)
        				{
        					e.printStackTrace();
        				}
        				totalWeightPDs[layerInd]=totalWeightPDs[layerInd].add(weightPDs[layerInd]);
        			}
        		}
        		
        		/*
        		backProp.addToBiasChanges(totalBiasPDs);
        		backProp.addToWeightChanges(totalWeightPDs);
        		*/
        		
        		backProp.updateWeightsBiases(centralNetwork, totalBiasPDs, totalWeightPDs);
        		
        		/*
        		finishedBatch=true;
        		while(finishedBatch)
		    	{
		    		try 
		    		{
						Thread.sleep(1);
					} 
		    		catch (InterruptedException e) 
		    		{
						e.printStackTrace();
					}
		    	}
		    	*/
        	}
        	time=System.nanoTime()-time;
        	System.out.println("Thread epoch time: "+time);
        	
        	/*
        	finishedEpoch=true;
        	while(finishedEpoch)
        	{
        		try 
	    		{
					Thread.sleep(1);
				} 
	    		catch (InterruptedException e) 
	    		{
					e.printStackTrace();
				}
        	}
        	*/
		}
		int u=0;
    }
    
}
