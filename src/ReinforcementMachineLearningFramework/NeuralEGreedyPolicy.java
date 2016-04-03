package ReinforcementMachineLearningFramework;

import java.util.List;
import java.util.Set;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.RectifiedLinearActivationFunction;
import activationFunctions.Sigmoid;
import costFunctions.CrossEntropy;
import costFunctions.EuclideanDistanceCostFunction;
import filters.ScaleFilter;
import learning.BackPropGradientDescent;
import network.FeedForwardNetwork;
import network.Network;
import tempDiffLambdaRMLTesuaro.StateAction;

import java.util.Map.Entry;

public class NeuralEGreedyPolicy extends EGreedyPolicy
{

	protected int stateLength;
	protected int actionLength;
	protected Network network;
	protected Action createAction;
	protected ScaleFilter scaleFilter;
	
	public NeuralEGreedyPolicy(double exploreChance, int stateLength, int actionLength, Action createAction) 
	{
		super(exploreChance);
		this.stateLength=stateLength;
		this.actionLength=actionLength;
		this.createAction=createAction;
		network=new FeedForwardNetwork(new Sigmoid(), new int[]{stateLength, stateLength, 1});
		scaleFilter=new ScaleFilter();
	}
	
	@Override
	public Action getKthBestAction(State state, int k, Environment environment) 
	{
		Object[] stateInfo=stateActions.get(state);
		if(stateInfo==null 
				|| Math.random()<e 
				|| ((List<ActionListElement>)stateInfo[0]).size()<=k
				/*|| ((List<ActionListElement>)stateInfo[0]).get(k).getValue()<=0*/)
		{
			Action action=environment.getRandomAction(state);
			ActionListElement actionListElement=new ActionListElement(action, 0.0);
			putActionListElement(state, actionListElement);
			return action;
		}
		//return ((List<ActionListElement>)stateInfo[0]).get(k).getAction();
		return getNNBestAction(state, environment);
	}
	
	protected Action getNNBestAction(State state, Environment environment)
	{
		StateAction[] allPossibleStateActions=((LimitedActionsEnvironment)environment).getAllPossibleStateActions(state);
		StateAction bestStateAction=null;
		double bestValue=Double.NEGATIVE_INFINITY;
		Object[] stateInfo=stateActions.get(state);
		
		for(StateAction possibleStateAction: allPossibleStateActions)
		{
			double value=network.getOutput(new ArrayRealVector(possibleStateAction.getState().getValue())).getEntry(0);
			
			if(value>bestValue)
			{
				bestStateAction=possibleStateAction;
				bestValue=value;
			}
		}

		return bestStateAction.getAction();
	}
	
	public void trainNN()
	{
		Set<Entry<State, Object[]>> stateInformation=getStateActionValues();
		ArrayRealVector[] states=new ArrayRealVector[stateInformation.size()];
		ArrayRealVector[] values=new ArrayRealVector[stateInformation.size()];
		
		int index=0;
		for(Entry<State, Object[]> stateEntry: stateInformation)
		{
			states[index]=new ArrayRealVector(stateEntry.getKey().getValue());
			values[index]=new ArrayRealVector(new double[]{((List<ActionListElement>)stateEntry.getValue()[0]).get(0).value+0.5});
			index++;
		}
		
		//states=scaleFilter.scaleData(states);
		//values=scaleFilter.scaleData(values);
		
		new BackPropGradientDescent(10, 60, 0.1).trainNetwork(network, states, values, new CrossEntropy());
	}
	

}
