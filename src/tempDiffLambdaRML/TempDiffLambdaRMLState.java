package tempDiffLambdaRML;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionListElement;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicyTesuaro;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;
import ReinforcementMachineLearningFramework.StatePolicy;
import hc.HCState;

public class TempDiffLambdaRMLState extends ReinforcementLearner
{

	LimitedActionsEnvironment enviroment;
	StatePolicy policy;
	State startState;
	int numberEpisodes;
	double stepSize; //a
	double discountRate; //y
	double traceDecay; //lambda
	HashMap<State, Double> eligibilityTraces;
	int exploreState;
	
	public TempDiffLambdaRMLState(LimitedActionsEnvironment enviroment, 
			StatePolicy policy,
			State startState,
			int numberEpisodes,
			double stepSize,
			double discountRate,
			double traceDecay)
	{
		this.enviroment=enviroment;
		this.policy=policy;
		this.startState=startState;
		this.numberEpisodes=numberEpisodes;
		this.stepSize=stepSize;
		this.discountRate=discountRate;
		this.traceDecay=traceDecay;
	}
	
	@Override
	public void learn() 
	{
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			enviroment.setStartState(startState);
			State currentState=enviroment.getCurrentState();
			Action bestAction=policy.getAction(currentState, enviroment);
			
			int actions=0;
			
			eligibilityTraces=new HashMap<>(); //[0]=StateAction, [1]=ActionListElement
			double reward=0.0;
			
			while(!enviroment.getCurrentState().isEndState())
			{
				actions++;
				reward=enviroment.takeAction(bestAction, 0);
				State newCurrentState=enviroment.getCurrentState();
				if(((HCState)newCurrentState).getValue()[0]==0.0)
				{
					int u=0;
				}
				Action newBestAction=policy.getAction(newCurrentState, enviroment);
				//double diff=reward+discountRate*getStateValue(newCurrentState)-getStateValue(currentState);
				setStateEligibility(currentState, getStateEligibility(currentState)+1);
				
				List<State> toRemove=new ArrayList<>();
				for(State state: eligibilityTraces.keySet())
				{
					double diff=reward+discountRate*getStateValue(newCurrentState)-getStateValue(state);
					if(diff>0 && getStateValue(state)>0.5)
					{
						int u=0;
					}
					policy.setStateValue(state, policy.getStateValue(state)
							+stepSize*diff*getStateEligibility(state));
					setStateEligibility(state, discountRate*traceDecay*getStateEligibility(state));
					if(getStateEligibility(state)<0.001)
					{
						toRemove.add(state);
					}
				}
				
				for(State state: toRemove)
				{
					eligibilityTraces.remove(state);
				}
				
				//((NeuralEGreedyPolicy)policy).trainNN(encounteredStateActions, traceDecay, stepSize);
				
				currentState=newCurrentState;
				bestAction=newBestAction;
			}
			
			/*
			setStateEligibility(currentState, getStateEligibility(currentState)+1);
			
			List<State> toRemove=new ArrayList<>();
			for(State state: eligibilityTraces.keySet())
			{
				double diff=reward-getStateValue(state);
				policy.setStateValue(state, policy.getStateValue(state)
						+stepSize*diff*getStateEligibility(state));
				setStateEligibility(state, discountRate*traceDecay*getStateEligibility(state));
				if(getStateEligibility(state)<0.001)
				{
					toRemove.add(state);
				}
			}
			*/
			
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
	
	private double getStateEligibility(State state)
	{
		Double value=eligibilityTraces.get(state);
		if(value==null)
		{
			return 0.0;
		}
		else
		{
			return value;
		}
	}
	
	private void setStateEligibility(State state, double eligibility)
	{
		eligibilityTraces.put(state, eligibility);
	}

}
