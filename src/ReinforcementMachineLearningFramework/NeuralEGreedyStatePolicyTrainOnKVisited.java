package ReinforcementMachineLearningFramework;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.Sigmoid;
import costFunctions.CrossEntropy;
import costFunctions.EuclideanDistanceCostFunction;
import filters.ScaleFilter;
import layer.FullyConnectedLayer;
import layer.InputLayer;
import layer.Layer;
import learningRule.BackPropGradientDescent;
import network.FeedForwardNetwork;
import network.Network;

public class NeuralEGreedyStatePolicyTrainOnKVisited extends NeuralEGreedyStatePolicy
{

    protected int vistedThreshold;
    protected Hashtable<State, Integer> timesVisted;
    
    protected int stateLength;
    
	
	public NeuralEGreedyStatePolicyTrainOnKVisited(double exploreChance, RewardFunction rewardFunction, int stateLength, int vistedThreshold) 
	{
		super(exploreChance, rewardFunction, null);
        this.vistedThreshold=vistedThreshold;
        timesVisted=new Hashtable<>();
        this.stateLength=stateLength;
        
        InputLayer inputLayer=new InputLayer(null, stateLength);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, stateLength);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, stateLength/2);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 1);
		network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
	}
	
	@Override
	public Action getAction(State state, LimitedActionsEnvironment environment) 
	{
		if(Math.random()<exploreChance)
		{
			StateAction stateAction=environment.getRandomStateAction(state);
			if(stateAction!=null)
			{
				incrementTimesVisted(stateAction.state);
				return stateAction.action;
			}
			else
			{
				return null;
			}
		}
		else
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
				/*
				double value
				=network.getOutput(new ArrayRealVector(possibleStateAction.state.getValue())).getEntry(0)
						+rewardFunction.getReward(state, possibleStateAction.state, 
								possibleStateAction.action);
				*/
				Double value=values.get(possibleStateAction.state);
				if(value==null)
				{
					value=network.getOutput(scaleFilter.scaleData(new ArrayRealVector(possibleStateAction.state.getValue()))).getEntry(0);
					//value=network.getOutput(new ArrayRealVector(possibleStateAction.state.getValue())).getEntry(0);
				}
				value+=rewardFunction.getReward(state, possibleStateAction.state, 
						possibleStateAction.action);
				/*
				double value=getStateValue(possibleStateAction.state)
						+rewardFunction.getReward(state, possibleStateAction.state, 
								possibleStateAction.action);
				*/
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
					return null;
				}	
			}
			else
			{
				incrementTimesVisted(bestStateAction.state);
				return bestStateAction.action;
			}
		}
	}
	
	public synchronized void trainNN()
	{
	    InputLayer inputLayer=new InputLayer(null, stateLength);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, stateLength);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, stateLength/2);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 1);
		network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
		
		Set<Entry<State, Double>> stateInformation=getStateValues();
		List<ArrayRealVector> states=new ArrayList<>();
		List<ArrayRealVector> stateValues=new ArrayList<>();
		
		for(Entry<State, Double> stateEntry: stateInformation)
		{
			if(getTimesVisted(stateEntry.getKey())>=vistedThreshold)
			{
				states.add(new ArrayRealVector(stateEntry.getKey().getValue()));
				stateValues.add(new ArrayRealVector(new double[]{stateEntry.getValue()+0.5}));
			}
		}
		
		System.out.println("Training on "+states.size()+" samples");

		new BackPropGradientDescent(100, 120, 0.1).trainNetwork(network, 
				scaleFilter.scaleData(states.toArray(new ArrayRealVector[0]), true), 
				scaleFilter.scaleData(stateValues.toArray(new ArrayRealVector[0]), false),
				new EuclideanDistanceCostFunction());
		
		/*
		backprop.trainNetwork(network, 
				states.toArray(new ArrayRealVector[0]), 
				stateValues.toArray(new ArrayRealVector[0]),
				new EuclideanDistanceCostFunction());
		*/
	}
	
	protected void incrementTimesVisted(State state)
	{
		timesVisted.put(state, getTimesVisted(state)+1);
	}
	
	protected int getTimesVisted(State state)
	{
		return timesVisted.getOrDefault(state, 0);
	}
	
	@Override
	public void setStateValue(State state, double value) 
	{
		timesVisted.put(state, new double[]{value, getTimesVisted(state)});
	}

	@Override
	public double getStateValue(State state) 
	{
		Double result=timesVisted.get(state)[0];
		if(result==null)
		{
			return network.getOutput(scaleFilter.scaleData(new ArrayRealVector(state.getValue()))).getEntry(0);
		}
		else
		{
			return result;
		}
	}

}
