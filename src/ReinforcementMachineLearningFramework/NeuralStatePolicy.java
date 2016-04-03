package ReinforcementMachineLearningFramework;

import java.util.Map.Entry;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.Sigmoid;
import costFunctions.EuclideanDistanceCostFunction;
import filters.ScaleFilter;
import layer.BInputLayer;
import layer.BLayer;
import layer.FullyConnectedBLayer;
import layer.FullyConnectedLayer;
import layer.InputLayer;
import layer.Layer;
import learningRule.BackPropGradientDescent;
import learningRule.MPBackPropGradientDescent;
import network.FeedForwardNetwork;
import network.Network;
import network.SplitFeedForwardNetwork;
import network.SplitNetwork;

public abstract class NeuralStatePolicy extends StatePolicy
{

	public SplitNetwork network;
	public HashMap<State, Double> values;
	public RewardFunction rewardFunction;
	public ScaleFilter scaleFilterInputs;
	public ScaleFilter scaleFilterOutputs;
	protected int stateLength;
	
	public NeuralStatePolicy(int stateLength, RewardFunction rewardFunction)
	{
		scaleFilterInputs=new ScaleFilter();
		scaleFilterOutputs=new ScaleFilter();
		values=new HashMap<>();
		this.rewardFunction=rewardFunction;
		this.stateLength=stateLength;
		
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		FullyConnectedBLayer hiddenLayer1a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, stateLength);
		FullyConnectedBLayer hiddenLayer1b=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, stateLength);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1a, hiddenLayer1b}, stateLength/2);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{hiddenLayer1a, hiddenLayer1b}, new BLayer[]{hiddenLayer2a}, new BLayer[]{outputLayer}});
	}
	
	@Override
	public abstract Action getAction(State state, LimitedActionsEnvironment environment);

	@Override
	public void setStateValue(State state, double value) 
	{
		values.put(state, value);
	}

	@Override
	public double getStateValue(State state) 
	{
		Double result=values.get(state);
		if(result==null)
		{
			return scaleFilterOutputs.unScaleData(network.getOutput(scaleFilterInputs.scaleData(new ArrayRealVector(state.getValue())))).getEntry(0);
		}
		else
		{
			return result;
		}
	}
	
	public Double getStateValueNoNet(State state) 
	{
		return values.get(state);
	}

	@Override
	public Set<Entry<State, Double>> getStateValues() 
	{
		return values.entrySet();
	}
	
	public Set<State> getStates()
	{
		return values.keySet();
	}
	
	public synchronized void trainNN()
	{
		BInputLayer inputLayer1=new BInputLayer(null, null, stateLength);
		FullyConnectedBLayer hiddenLayer1a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, stateLength);
		FullyConnectedBLayer hiddenLayer1b=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, stateLength);
		FullyConnectedBLayer hiddenLayer2a=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1a, hiddenLayer1b}, stateLength/2);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2a}, 1);
		network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{hiddenLayer1a, hiddenLayer1b}, new BLayer[]{hiddenLayer2a}, new BLayer[]{outputLayer}});
		
		Set<Entry<State, Double>> stateInformation=getStateValues();
		List<ArrayRealVector> states=new ArrayList<>();
		List<ArrayRealVector> stateValues=new ArrayList<>();
		
		for(Entry<State, Double> stateEntry: stateInformation)
		{
			states.add(new ArrayRealVector(stateEntry.getKey().getValue()));
			stateValues.add(new ArrayRealVector(new double[]{stateEntry.getValue()}));
		}
		
		System.out.println("Training on "+states.size()+" samples");

		new MPBackPropGradientDescent(100, 120, 0.1).trainNetwork(network, 
				scaleFilterInputs.scaleData(states.toArray(new ArrayRealVector[0]), true), 
				scaleFilterOutputs.scaleData(stateValues.toArray(new ArrayRealVector[0]), true),
				new EuclideanDistanceCostFunction());
	}
	
	public Action getBestAction(State state, LimitedActionsEnvironment environment) 
	{
		StateAction[] possibleStateActions=environment.getAllPossibleStateActions(state);
		double bestValue=Double.MIN_VALUE;
		StateAction bestStateAction=null;
		for(StateAction possibleStateAction: possibleStateActions)
		{
			Double value=getStateValue(possibleStateAction.state);
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

}
