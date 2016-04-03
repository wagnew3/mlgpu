package ReinforcementMachineLearningFramework;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;
import java.util.Map.Entry;

public class EGreedyStatePolicy  extends StatePolicy
{
	
	double e;
	HashMap<State, Double> values;
	RewardFunction rewardFunction;
	
	public EGreedyStatePolicy(double exploreChance, RewardFunction rewardFunction)
	{
		e=exploreChance;
		values=new HashMap<>();
		this.rewardFunction=rewardFunction;
	}

	@Override
	public Action getAction(State state, LimitedActionsEnvironment environment) 
	{
		if(Math.random()<e)
		{
			Action action=environment.getRandomAction(state);
			return action;
		}
		else
		{
			StateAction[] possibleStateActions=environment.getAllPossibleStateActions(state);
			double bestValue=Double.MIN_VALUE;
			StateAction bestStateAction=null;
			for(StateAction possibleStateAction: possibleStateActions)
			{
				if(getStateValue(possibleStateAction.state)
						+rewardFunction.getReward(state, possibleStateAction.state, 
								possibleStateAction.action)>bestValue)
				{
					bestValue=getStateValue(possibleStateAction.state)+rewardFunction.getReward(state, possibleStateAction.state, 
							possibleStateAction.action);
					bestStateAction=possibleStateAction;
				}
			}
			if(bestStateAction==null)
			{
				Action action=environment.getRandomAction(state);
				return action;
			}
			else
			{
				return bestStateAction.action;
			}
		}
	}

	@Override
	public void setStateValue(State state, double value) 
	{
		values.put(state, value);
	}

	@Override
	public double getStateValue(State state) 
	{
		Double value=values.get(state);
		if(value==null)
		{
			return 0.0;
		}
		else
		{
			if(Double.isNaN(value))
			{
				int t=0;
			}
			return value;
		}
	}

	@Override
	public Set<Entry<State, Double>> getStateValues() 
	{
		return values.entrySet();
	}
	

}
