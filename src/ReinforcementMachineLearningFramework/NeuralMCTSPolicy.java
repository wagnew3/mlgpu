package ReinforcementMachineLearningFramework;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.L2Pooling;
import activationFunctions.RectifiedLinearActivationFunction;
import activationFunctions.Sigmoid;
import activationFunctions.TanH;
import checkers.CheckersF1CostFunction;
import checkers.CheckersF1Error;
import checkers.CheckersState;
import costFunctions.CrossEntropy;
import costFunctions.EuclideanDistanceCostFunction;
import layer.BInputLayer;
import layer.BLayer;
import layer.ConvolutionBLayer;
import layer.ConvolutionLayer;
import layer.FullyConnectedBLayer;
import layer.FullyConnectedLayer;
import layer.InputLayer;
import layer.Layer;
import layer.PoolingLayer;
import learningRule.BPGDUnsupervisedTraining;
import learningRule.BackPropGradientDescent;
import learningRule.BackPropGradientDescentStopBelowEval;
import learningRule.BackPropGradientDescentStopBelowPPEval;
import learningRule.MPBackPropGradientDescent;
import learningRule.MPBackPropGradientDescentEvalFunc;
import network.FeedForwardNetwork;
import network.Network;
import network.SplitFeedForwardNetwork;
import network.SplitNetwork;
import regularization.L2Regularization;
import boardGame.BoardEnvironment;

public class NeuralMCTSPolicy extends NeuralStatePolicy
{
	
	public HashMap<State, double[]> stateInfo; //0=value, 1=timesVisted
	protected double totalNumberVisits;
	protected LimitedActionsEnvironment environment;
	
	public NeuralMCTSPolicy(int stateLength, RewardFunction rewardFunction, LimitedActionsEnvironment environment)
	{
		super(stateLength, rewardFunction);
		stateInfo=new HashMap<>();
		totalNumberVisits=0.000001;
		this.environment=environment;
	}

	@Override
	public Action getAction(State state, LimitedActionsEnvironment environment) 
	{
		StateAction[] possibleStateActions=environment.getAllPossibleStateActions(state);
		if(possibleStateActions==null)
		{
			return null;
		}
		double bestValue=Double.NEGATIVE_INFINITY;
		StateAction bestStateAction=null;
		for(StateAction possibleStateAction: possibleStateActions)
		{
			Double value;
			double[] result=stateInfo.get(state);
			if(result==null)
			{
				value=0.0;
			}
			else
			{
				value=result[0];
			}
			//Double value=getStateValue(possibleStateAction.state);
			value+=rewardFunction.getReward(state, possibleStateAction.state, 
					possibleStateAction.action);
			
			double timesVisted=getTimesVisted(possibleStateAction.state);
			value=Math.pow(totalNumberVisits, 0.9)*(value/timesVisted)+Math.sqrt(0.005*Math.log(totalNumberVisits)/timesVisted);
			
			if(value>bestValue)
			{
				bestValue=value;
				bestStateAction=possibleStateAction;
			}
		}
		if(bestStateAction==null)
		{
			StateAction stateAction=environment.getRandomStateAction(state);
			if(stateAction!=null)
			{
				incrementTimesVisted(stateAction.state);
				return stateAction.action;
			}
			else
			{
				//((BoardEnvironment)environment).displayBoard();
				return null;
			}	
		}
		else
		{
			incrementTimesVisted(bestStateAction.state);
			return bestStateAction.action;
		}
	}
	
	protected void incrementTimesVisted(State state)
	{
		Object val=stateInfo.get(state);
		if(val==null)
		{
			stateInfo.put(state, new double[]{0.0, 1.0});
		}
		else
		{
			((double[])val)[1]++;
		}
		totalNumberVisits++;
	}
	
	protected double getTimesVisted(State state)
	{
		return stateInfo.getOrDefault(state, new double[]{0.0, 1.0})[1];
	}
	
	@Override
	public void setStateValue(State state, double value) 
	{
		Object val=stateInfo.get(state);
		if(val==null)
		{
			stateInfo.put(state, new double[]{value, 0.0});
		}
		else
		{
			((double[])val)[0]=value;
		}
	}

	@Override
	public double getStateValue(State state) 
	{
		double[] result=stateInfo.get(state);
		if(result==null)
		{
			return scaleFilterOutputs.unScaleData(network.getOutput(scaleFilterInputs.scaleData(new ArrayRealVector(state.getNNValue())))).getEntry(0);
		}
		else
		{
			return result[0];
		}
	}
	
	public double getStateValueNet(State state) 
	{
		return scaleFilterOutputs.unScaleData(network.getOutput(scaleFilterInputs.scaleData(new ArrayRealVector(state.getNNValue())))).getEntry(0);
	}
	
	public Double getStateValueNoNet(State state) 
	{
		double[] result=stateInfo.get(state);
		if(result==null)
		{
			return null;
		}
		else
		{
			return result[0];
		}
	}
	
	public Set<State> getStates()
	{
		return stateInfo.keySet();
	}
	
	public void trim()
	{
		Set<Entry<State, double[]>> stateInformation=stateInfo.entrySet();
		List<Entry<State, double[]>> toRemove=new ArrayList<>();
		for(Entry<State, double[]> stateEntry: stateInformation)
		{
			if(stateEntry.getValue()[1]<=25)
			{
				toRemove.add(stateEntry);
			}
		}
		stateInformation.removeAll(toRemove);
	}
	
	public synchronized void trainNN()
	{
		/*
		InputLayer inputLayer=new InputLayer(null, stateLength);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, stateLength);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, stateLength/2);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 1);
		network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
		*/
		
		/*
		InputLayer inputLayer=new InputLayer(null, stateLength);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, 2*stateLength);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, 1);
		network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, outputLayer});
		*/
		
		/*
		InputLayer inputLayer=new InputLayer(null, 6*6);
		ConvolutionLayer convLayer=new ConvolutionLayer(new Sigmoid(), new int[]{6, 6}, 1, 3);
		//PoolingLayer poolingLayer=new PoolingLayer(new L2Pooling(), new int[]{4, 4}, 2, 1);
		FullyConnectedLayer hiddenLayer=new FullyConnectedLayer(new Sigmoid(), convLayer, 30);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer, 10);
		//FullyConnectedLayer hiddenLayer=new FullyConnectedLayer(new Sigmoid(), poolingLayer, 100);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 1);
		Network network=new FeedForwardNetwork(new Layer[]{inputLayer, convLayer, hiddenLayer, hiddenLayer2, outputLayer});
		*/
		
		/*
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		FullyConnectedBLayer hiddenLayer1a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, stateLength);
		FullyConnectedBLayer hiddenLayer1b=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, stateLength);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1a, hiddenLayer1b}, stateLength/2);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{hiddenLayer1a, hiddenLayer1b}, new BLayer[]{hiddenLayer2a}, new BLayer[]{outputLayer}});
		*/
		
		/*
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		ConvolutionBLayer convLayer=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{6, 6}, 1, 3);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{convLayer}, stateLength/2);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{convLayer}, new BLayer[]{hiddenLayer2a}, new BLayer[]{outputLayer}});
		*/
		
		/*
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		ConvolutionBLayer convLayer1a=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{6, 6}, 1, 3);
		ConvolutionBLayer convLayer1b=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{6, 6}, 1, 4);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{convLayer1a, convLayer1b}, stateLength/2);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{convLayer1a, convLayer1b}, new BLayer[]{hiddenLayer2a}, new BLayer[]{outputLayer}});
		*/
		
		/*
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		FullyConnectedBLayer hiddenLayer1a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, stateLength);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1a}, stateLength/2);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{hiddenLayer1a}, new BLayer[]{hiddenLayer2a}, new BLayer[]{outputLayer}});
		*/
		
		/*
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		ConvolutionBLayer convLayer1a=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{6, 6}, 1, 3);
		ConvolutionBLayer convLayer1b=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{6, 6}, 1, 4);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{convLayer1a}, stateLength);
		FullyConnectedBLayer hiddenLayer2b=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{convLayer1b}, stateLength);
		FullyConnectedBLayer hiddenLayer3a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a, hiddenLayer2b}, stateLength/2);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer3a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{convLayer1a, convLayer1b}, new BLayer[]{hiddenLayer2a, hiddenLayer2b}, new BLayer[]{hiddenLayer3a}, new BLayer[]{outputLayer}});
		*/
		
		
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		BInputLayer inputLayer2=new BInputLayer(null, null, 4);
		ConvolutionBLayer convLayer1a=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{8, 8}, 1, 3);
		ConvolutionBLayer convLayer1b=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{8, 8}, 1, 4);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{convLayer1a}, stateLength/4);
		FullyConnectedBLayer hiddenLayer2b=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{convLayer1b}, stateLength/4);
		FullyConnectedBLayer hiddenLayer2c=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer2}, 1);
		FullyConnectedBLayer hiddenLayer3a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2c, hiddenLayer2a, hiddenLayer2b}, stateLength/8);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer3a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1, inputLayer2}, new BLayer[]{convLayer1a, convLayer1b}, new BLayer[]{hiddenLayer2c, hiddenLayer2a, hiddenLayer2b}, new BLayer[]{hiddenLayer3a}, new BLayer[]{outputLayer}});
		
		
		/*
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		ConvolutionBLayer convLayer1a=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{6, 6}, 1, 3);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{convLayer1a}, stateLength);
		FullyConnectedBLayer hiddenLayer3a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a}, stateLength/4);
		FullyConnectedBLayer hiddenLayer4a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer3a}, stateLength/16);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer4a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{convLayer1a}, new BLayer[]{hiddenLayer2a}, new BLayer[]{hiddenLayer3a}, new BLayer[]{hiddenLayer4a}, new BLayer[]{outputLayer}});
		*/
		
		Set<Entry<State, double[]>> stateInformation=stateInfo.entrySet();
		
		List<ArrayRealVector> states=new ArrayList<>();
		List<ArrayRealVector> stateValues=new ArrayList<>();
		
		int quota=15000;
		int[] numberAdded=new int[12];
		int above=500;
		boolean allQuotasMet=false;
		
		while(above>=0 && !allQuotasMet)
		{
			for(Entry<State, double[]> stateEntry: stateInformation)
			{
				if(stateEntry.getValue()[1]>above
						&& numberAdded[((CheckersState)stateEntry.getKey()).numberPieces()-1]<quota)
				{
					numberAdded[((CheckersState)stateEntry.getKey()).numberPieces()-1]++;
					states.add(new ArrayRealVector(stateEntry.getKey().getNNValue()));
					stateValues.add(new ArrayRealVector(new double[]{stateEntry.getValue()[0]}));
				}
			}
			
			allQuotasMet=true;
			for(int quotaInd=0; quotaInd<numberAdded.length; quotaInd++)
			{
				if(numberAdded[quotaInd]<quota)
				{
					allQuotasMet=false;
					break;
				}
			}
			above--;
		}
		
		System.out.println("Training on "+states.size()+" samples");

		/*
		new BackPropGradientDescentStopBelowPPEval(100, 30, 0.1, 0.1, 
				new CheckersF1Error(environment, this)).trainNetwork(network, 
				scaleFilterInputs.scaleData(states.toArray(new ArrayRealVector[0]), true), 
				scaleFilterOutputs.scaleData(stateValues.toArray(new ArrayRealVector[0]), true),
				new CheckersF1CostFunction(environment, this, stateInfo.keySet().iterator().next()));
				*/
		
		/*
		new BackPropGradientDescent(100000, 20000, 0.01).trainNetwork(network, 
				scaleFilterInputs.scaleData(states.toArray(new ArrayRealVector[0]), true), 
				scaleFilterOutputs.scaleData(stateValues.toArray(new ArrayRealVector[0]), true),
				new EuclideanDistanceCostFunction());
				*/
		
		/*
		double lambda=0.01;
		MPBackPropGradientDescentEvalFunc bpgd=new MPBackPropGradientDescentEvalFunc(100000, 50, lambda, new CheckersF1Error(environment, this));
		bpgd.setRegularization(new L2Regularization(outputLayer.getOutputSize(), lambda, 0.1));
		bpgd
		.trainNetwork(network, 
				scaleFilterInputs.scaleData(states.toArray(new ArrayRealVector[0]), true), 
				scaleFilterOutputs.scaleData(stateValues.toArray(new ArrayRealVector[0]), true),
				new EuclideanDistanceCostFunction());
		*/
		
		/*
		new BackPropGradientDescentStopBelowEval(500, 300, 0.01, -0.95, 
				new CheckersF1Error(environment, this)).trainNetwork(network, 
				scaleFilterInputs.scaleData(states.toArray(new ArrayRealVector[0]), true), 
				scaleFilterOutputs.scaleData(stateValues.toArray(new ArrayRealVector[0]), true),
				new EuclideanDistanceCostFunction());
				*/
		
		double lambda=0.01;
		BPGDUnsupervisedTraining bpgd=new BPGDUnsupervisedTraining(100, 30, lambda, null);
		//bpgd.setRegularization(new L2Regularization(outputLayer.getOutputSize(), lambda, 0.1));
		
		ArrayRealVector[] inputs=scaleFilterInputs.scaleData(states.toArray(new ArrayRealVector[0]), true);
		ArrayRealVector[] outputs=scaleFilterOutputs.scaleData(stateValues.toArray(new ArrayRealVector[0]), true);
		
		
		for(int i=0; i<1; i++)
		{
			bpgd.unsupervisedTrain(network, inputs,
					outputs, new EuclideanDistanceCostFunction());
			
			bpgd.trainNetwork(network, inputs,
					outputs, new EuclideanDistanceCostFunction());
		}
	}

}
