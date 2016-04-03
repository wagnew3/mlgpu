package ReinforcementMachineLearningFramework;

import java.util.List;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.Sigmoid;
import costFunctions.CrossEntropy;
import costFunctions.EuclideanDistanceCostFunction;
import layer.FullyConnectedLayer;
import layer.InputLayer;
import layer.Layer;
import learningRule.BackPropGradientDescent;
import network.FeedForwardNetwork;
import network.Network;

public class NeuralEGreedyStatePolicy extends NeuralStatePolicy
{
	
	protected int stateLength;
    protected BackPropGradientDescent backprop;
    protected double exploreChance;
    protected RewardFunction rewardFunction;
    
	
	public NeuralEGreedyStatePolicy(double exploreChance, RewardFunction rewardFunction, Network network) 
	{
		this.exploreChance=exploreChance;
		this.rewardFunction=rewardFunction;
		this.network=network;
	}
	
	@Override
	public Action getAction(State state, LimitedActionsEnvironment environment) 
	{
		if(Math.random()<exploreChance)
		{
			Action action=environment.getRandomStateAction(state).action;
			return action;
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
					value=network.getOutput(new ArrayRealVector(possibleStateAction.state.getValue())).getEntry(0);
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
				Action action=environment.getRandomStateAction(state).action;
				return action;
			}
			else
			{
				return bestStateAction.action;
			}
		}
	}
	
	public Action getBestAction(State state, int k, LimitedActionsEnvironment environment) 
	{
		StateAction[] possibleStateActions=environment.getAllPossibleStateActions(state);
		double bestValue=Double.MIN_VALUE;
		StateAction bestStateAction=null;
		for(StateAction possibleStateAction: possibleStateActions)
		{
			Double value=values.get(possibleStateAction.state);
			if(value==null)
			{
				value=network.getOutput(new ArrayRealVector(possibleStateAction.state.getValue())).getEntry(0);
			}
			value+=rewardFunction.getReward(state, possibleStateAction.state, 
					possibleStateAction.action);
			
			if(value>bestValue)
			{
				bestValue=value;
				bestStateAction=possibleStateAction;
			}
		}
		if(bestStateAction==null)
		{
			Action action=environment.getRandomStateAction(state).action;
			return action;
		}
		else
		{
			return bestStateAction.action;
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
		ArrayRealVector[] states=new ArrayRealVector[stateInformation.size()];
		ArrayRealVector[] stateValues=new ArrayRealVector[stateInformation.size()];
		
		int index=0;
		for(Entry<State, Double> stateEntry: stateInformation)
		{
			states[index]=new ArrayRealVector(stateEntry.getKey().getValue());
			stateValues[index]=new ArrayRealVector(new double[]{stateEntry.getValue()+0.5});
			index++;
		}
		
		//states=scaleFilter.scaleData(states);
		//values=scaleFilter.scaleData(values);
		
		backprop.trainNetwork(network, states, stateValues, new EuclideanDistanceCostFunction());
	}

}
