package monteCarloState;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StatePolicy;
import hc.HCState;

public class MonteCarloState extends ReinforcementLearner
{

	LimitedActionsEnvironment enviroment;
	StatePolicy policy;
	State startState;
	int numberEpisodes;
	int exploreState;
	Hashtable<State, Double> totalReturns;
	
	public MonteCarloState(LimitedActionsEnvironment enviroment, 
			StatePolicy policy,
			State startState,
			int numberEpisodes)
	{
		this.enviroment=enviroment;
		this.policy=policy;
		this.startState=startState;
		this.numberEpisodes=numberEpisodes;
		totalReturns=new Hashtable<>();
	}
	
	@Override
	public void learn() 
	{
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			List<State> encounteredStates=new ArrayList<>();
			List<Double> rewards=new ArrayList<>();
			
			enviroment.setStartState(startState);
			State currentState=enviroment.getCurrentState();
			Action bestAction=policy.getAction(currentState, enviroment);
			
			int actions=0;
			
			double reward=0.0;
			
			while(!enviroment.getCurrentState().isEndState())
			{
				actions++;
				reward=enviroment.takeAction(bestAction, 0);
				
				encounteredStates.add(currentState);
				rewards.add(reward);
				
				State newCurrentState=enviroment.getCurrentState();
				if(((HCState)newCurrentState).getValue()[0]==0.0)
				{
					int u=0;
				}
				Action newBestAction=policy.getAction(newCurrentState, enviroment);
				currentState=newCurrentState;
				bestAction=newBestAction;
			}
			
			Hashtable<State, State> updatedStates=new Hashtable<>();
			for(int stateInd=0; stateInd<encounteredStates.size(); stateInd++)
			{
				if(updatedStates.get(encounteredStates.get(stateInd))==null)
				{
					double returnValue=0.0;
					for(int returnInd=stateInd; returnInd<encounteredStates.size(); returnInd++)
					{
						returnValue+=rewards.get(returnInd);
					}
					double oldStateValue=getStateValue(encounteredStates.get(stateInd));
					double oldNumberUpdates=getTotalReturns(encounteredStates.get(stateInd));
					double newStateValue=(oldStateValue*oldNumberUpdates+returnValue)/(oldNumberUpdates+1.0);
					setStateValue(encounteredStates.get(stateInd), newStateValue);
					incTotalReturns(encounteredStates.get(stateInd));
					
				}
				updatedStates.put(encounteredStates.get(stateInd), encounteredStates.get(stateInd));
			}
			
			if(actions>1)
			{
				int o=0;
			}
			
			System.out.println("Took "+actions+" actions Episode "+numberEpisodesCompleted);
		}
	}
	
	public int run(State initialRunState)
	{
		enviroment.setStartState(initialRunState);
		State currentState=enviroment.getCurrentState();
		Action bestAction=policy.getAction(currentState, enviroment);
		
		int actions=0;
		
		while(!enviroment.getCurrentState().isEndState())
		{
			actions++;
			enviroment.takeAction(bestAction, 0);
			State newCurrentState=enviroment.getCurrentState();
			Action newBestAction=policy.getAction(newCurrentState, enviroment);
			currentState=newCurrentState;
			bestAction=newBestAction;
		}
		return actions;
		//System.out.println("Took "+actions+" actions");
	}
	
	private double getStateValue(State state)
	{
		return policy.getStateValue(state);
	}
	
	private void setStateValue(State state, double value)
	{
		policy.setStateValue(state, value);
	}
	
	private double getTotalReturns(State state)
	{
		return totalReturns.getOrDefault(state, 0.0);
	}
	
	private void incTotalReturns(State state)
	{
		policy.setStateValue(state, getTotalReturns(state)+1.0);
	}
	
}
