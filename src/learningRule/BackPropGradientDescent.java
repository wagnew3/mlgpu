package learningRule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import costFunctions.CostFunction;
import network.Network;
//import test.MNISTNumbers;
import test.MNISTNumbers;

public class BackPropGradientDescent extends LearningRule
{
	
	protected int batchSize;
	protected int epochs;
	protected double learningRate;
	
	public BackPropGradientDescent(int batchSize, int epochs, double learningRate)
	{
		this.batchSize=batchSize;
		this.epochs=epochs;
		this.learningRate=learningRate;
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
				
			}
			System.out.println("******************* Epoch: "+currentEpoch);
			//MNISTNumbers.evalNumberNet(network);
			time=System.nanoTime()-time;
			//System.out.println(time);
			//MNISTNumbers.evalNumberNet(network);
			//double output=network.getOutput(inputs[0]).getEntry(0);
			//int u=0;
		}
		
		double totalError=0.0;
		for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
		{
			ArrayRealVector networkOutput=network.getOutput(inputs[sampleInd]);
			totalError+=costFunction.getCost(inputs[sampleInd], networkOutput, desiredOutputs[sampleInd]);
		}
		totalError/=inputs.length;
		System.out.println("Relative Error: "+totalError);
		
		double relativeError=0.0;
		for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
		{
			ArrayRealVector networkOutput=network.getOutput(inputs[sampleInd]);
			relativeError+=costFunction.getCost(inputs[sampleInd], networkOutput, desiredOutputs[sampleInd])/networkOutput.getL1Norm();
		}
		relativeError/=inputs.length;
		System.out.println("Relative Error: "+relativeError);
	}
	
	//0=deltas, 1=weightPDs
	protected Object[] backprop(Network network, ArrayRealVector input, ArrayRealVector desiredOutput, CostFunction costFunction)
	{
		ArrayRealVector[] outputs=new ArrayRealVector[network.getLayers().length];
		ArrayRealVector[] outputsDerivative=new ArrayRealVector[network.getLayers().length];
		outputs[0]=input;
		for(int layerIndex=1; layerIndex<network.getLayers().length; layerIndex++)
		{
			outputs[layerIndex]=network.getLayers()[layerIndex].getOutput(outputs[layerIndex-1]);
			outputsDerivative[layerIndex]=network.getLayers()[layerIndex].getOutputDerivatives(outputs[layerIndex-1]);
		}
		
		ArrayRealVector[] deltas=new ArrayRealVector[network.getLayers().length];
		RealMatrix[] weights=new RealMatrix[network.getLayers().length];
		
		ArrayRealVector costFunctionDerivative=costFunction.getCostDerivative(input, outputs[outputs.length-1], desiredOutput);
		
		deltas[deltas.length-1]=costFunctionDerivative.ebeMultiply(outputsDerivative[outputsDerivative.length-1]);
		weights[weights.length-1]=network.getLayers()[weights.length-1].getWeightPDs(outputs[weights.length-2], deltas[weights.length-1]);
		
		for(int layerIndex=deltas.length-2; layerIndex>0; layerIndex--)
		{
			deltas[layerIndex]=network.getLayers()[layerIndex]
					.getDeltas(network.getLayers()[layerIndex+1].getWeights(),
							deltas[layerIndex+1], outputsDerivative[layerIndex]);
			
			weights[layerIndex]=network.getLayers()[layerIndex].getWeightPDs(outputs[layerIndex-1], deltas[layerIndex]);
		}
		
		return new Object[]{deltas, weights};
	}

}
