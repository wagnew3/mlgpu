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
import layer.BLayer;
import network.SplitNetwork;
import test.MNISTNumbers;

public class MPBackPropGradientDescentEvalFunc extends MPBackPropGradientDescent
{

	protected EvaluationFunction evalFunc;
	
	public MPBackPropGradientDescentEvalFunc(int batchSize, int epochs,
			double learningRate, EvaluationFunction evalFunc) 
	{
		super(batchSize, epochs, learningRate);
		this.evalFunc=evalFunc;
	}
	
	@Override
	public void trainNetwork(SplitNetwork network, ArrayRealVector[] inputs,
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
				if(costFunction instanceof PreprocessCostFunction)
				{
					((PreprocessCostFunction)costFunction).preprocessDerivatives(inputs, desiredOutputs, network);
				}
				ArrayRealVector[] batchInputs=new ArrayRealVector[Math.min(batchSize, randomizedInputs.size()-sampleInd)];
				ArrayRealVector[] batchDesiredOutputs=new ArrayRealVector[Math.min(batchSize, randomizedInputs.size()-sampleInd)];
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedInputs.size()-sampleInd); batchInd++)
				{
					batchInputs[batchInd]=randomizedInputs.get(batchInd+sampleInd)[0];
					batchDesiredOutputs[batchInd]=randomizedInputs.get(batchInd+sampleInd)[1];
				}
				
				ArrayRealVector[][] totalBiasPDs=new ArrayRealVector[network.getLayers().length][];
				BlockRealMatrix[][][] totalWeightPDs=new BlockRealMatrix[network.getLayers().length][][];
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					totalBiasPDs[layerInd]=new ArrayRealVector[network.getLayers()[layerInd].length];
					totalWeightPDs[layerInd]=new BlockRealMatrix[network.getLayers()[layerInd].length][];
					for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
					{
						totalBiasPDs[layerInd][netInd]=new ArrayRealVector(network.getLayers()[layerInd][netInd].getBiases().getDimension());
						totalWeightPDs[layerInd][netInd]=new BlockRealMatrix[network.getLayers()[layerInd][netInd].getInputLayers().length];
						for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
						{
							totalWeightPDs[layerInd][netInd][inputNetInd]
									=new BlockRealMatrix(network.getLayers()[layerInd][netInd].getWeights()[0].getRowDimension()
											, network.getLayers()[layerInd][netInd].getWeights()[0].getColumnDimension());
						}
					}
				}
				
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedInputs.size()-sampleInd); batchInd++)
				{
					Object[] pds=backprop(network, batchInputs[batchInd], batchDesiredOutputs[batchInd], costFunction);
					ArrayRealVector[][] biasPDs=(ArrayRealVector[][])pds[0];
					RealMatrix[][][] weightPDs=(RealMatrix[][][])pds[1];
					for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
					{
						for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
						{
							totalBiasPDs[layerInd][netInd]=totalBiasPDs[layerInd][netInd].add(biasPDs[layerInd][netInd]);
							for(int inputNetInd=0; inputNetInd<network.getLayers()[layerInd][netInd].getInputLayers().length; inputNetInd++)
							{
								totalWeightPDs[layerInd][netInd][inputNetInd]=totalWeightPDs[layerInd][netInd][inputNetInd].add(weightPDs[layerInd][netInd][inputNetInd]);
							}
						}
					}
				}
				
				if(regularization!=null)
				{
					network=regularization.regularize(network);
				}
				
				for(int layerInd=1; layerInd<network.getLayers().length; layerInd++)
				{
					for(int netInd=0; netInd<network.getLayers()[layerInd].length; netInd++)
					{
						network.getLayers()[layerInd][netInd].updateBiases(totalBiasPDs[layerInd][netInd], learningRate);
						network.getLayers()[layerInd][netInd].updateWeights(totalWeightPDs[layerInd][netInd], learningRate);
					}
				}
			}
			System.out.println("******************* Epoch: "+currentEpoch);
			
			if(costFunction instanceof PreprocessCostFunction)
			{
				((PreprocessCostFunction)costFunction).preprocessCosts(inputs, desiredOutputs, network);
				System.out.println("Total Cost:"+((PreprocessCostFunction)costFunction).totalCost());
			}
			
			double totalError=0.0;
			for(int sampleInd=0; sampleInd<inputs.length; sampleInd++)
			{
				ArrayRealVector networkOutput=network.getOutput(inputs[sampleInd]);
				totalError+=costFunction.getCost(inputs[sampleInd], networkOutput, desiredOutputs[sampleInd]);
			}
			totalError/=inputs.length;
			System.out.println("totalError: "+totalError);
			
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
			
			if(currentEpoch%100==0
					&& currentEpoch>0)
			{
				//double eval=evalFunc.getEval(network);
				
				//System.out.println("Eval: "+eval);
			}
			
			if(network.layers.length>3)
			{
				//MNISTNumbers.evalNumberNet(network);
			}
			time=System.nanoTime()-time;
			//System.out.println(time);
			//MNISTNumbers.evalNumberNet(network);
			//double output=network.getOutput(inputs[0]).getEntry(0);
			//int u=0;
		}
	}

}
