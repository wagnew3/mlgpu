package learningRule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import costFunctions.CostFunction;
import costFunctions.PreprocessCostFunction;
import evaluationFunctions.EvaluationFunction;
import network.Network;

public class BackPropGradientDescentStopBelowPPEval extends BackPropGradientDescent
{

	protected double maxEval;
	protected EvaluationFunction evaluationFunction;
	
	public BackPropGradientDescentStopBelowPPEval(int batchSize, int epochs, double learningRate, 
			double maxEval, EvaluationFunction evaluationFunction)
	{
		super(batchSize, epochs, learningRate);
		this.maxEval=maxEval;
		this.evaluationFunction=evaluationFunction;
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
		
		double eval=Double.MAX_VALUE;
		
		for(int currentEpoch=0; currentEpoch<epochs && eval>maxEval; currentEpoch++)
		{
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
				
				((PreprocessCostFunction)costFunction).preprocessDerivatives(batchInputs, batchDesiredOutputs, network);
				
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
			
			((PreprocessCostFunction)costFunction).preprocessCosts(inputs, desiredOutputs, network);
			
			/*
			double cost=0.0;
			for(int sampleInd=0; sampleInd<randomizedInputs.size(); sampleInd++)
			{
				cost+=costFunction
						.getCost(randomizedInputs.get(sampleInd)[0], 
								network.getOutput(randomizedInputs.get(sampleInd)[0]), 
								randomizedInputs.get(sampleInd)[1]);
			}
			cost/=randomizedInputs.size();
			*/
			
			System.out.println("Cost: "+((PreprocessCostFunction)costFunction).totalCost());
		}
		
		
	}
	
}
