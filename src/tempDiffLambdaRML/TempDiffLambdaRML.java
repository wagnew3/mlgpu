package tempDiffLambdaRML;

import java.util.Hashtable;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionListElement;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicyTesuaro;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;

public class TempDiffLambdaRML extends ReinforcementLearner
{

	Environment enviroment;
	ActionPolicy policy;
	State startState;
	int numberEpisodes;
	double stepSize; //a
	double discountRate; //y
	double traceDecay; //lambda
	Hashtable<StateAction, StateAction> stateActions;
	
	public TempDiffLambdaRML(Environment enviroment, 
			ActionPolicy policy,
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
		stateActions=new Hashtable<>();
	}
	
	@Override
	public void learn() 
	{
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			enviroment.setStartState(startState);
			State currentState=enviroment.getCurrentState();
			Action bestAction=policy.getKthBestAction(currentState, 0, enviroment);
			
			int actions=0;
			
			Hashtable<StateAction, Object[]> encounteredStateActions=new Hashtable<>(); //[0]=StateAction, [1]=ActionListElement
			
			while(!enviroment.getCurrentState().isEndState())
			{
				actions++;
				double reward=enviroment.takeAction(bestAction, this);
				State newCurrentState=enviroment.getCurrentState();
				Action newBestAction=policy.getKthBestAction(newCurrentState, 0, enviroment);
				double diff=reward+discountRate*getActionStateValue(newCurrentState, newBestAction)
							-getActionStateValue(currentState, bestAction);
				setActionStateEligibility(currentState, bestAction,
						getActionStateEligibility(currentState, bestAction)+1);
				
				StateAction updateStateAction=stateActions.get(new StateAction(currentState, bestAction));
				encounteredStateActions.put(updateStateAction, new Object[]{updateStateAction, policy.getActionListElement(currentState, bestAction)});
				
				for(Object[] stateActionInfo: encounteredStateActions.values())
				{
					/*
					policy.setStateActionValue(((StateAction)stateActionInfo[0]).state, ((StateAction)stateActionInfo[0]).action, ((ActionListElement)stateActionInfo[1]).value
							+stepSize*diff*((StateAction)stateActionInfo[0]).eligibility);
					*/
					((StateAction)stateActionInfo[0]).eligibility=discountRate*traceDecay*((StateAction)stateActionInfo[0]).eligibility;
					if(((StateAction)stateActionInfo[0]).eligibility<0.001)
					{
						((StateAction)stateActionInfo[0]).eligibility=0;
						encounteredStateActions.remove(stateActionInfo);
					}
				}
				
				//((NeuralEGreedyPolicy)policy).trainNN(encounteredStateActions, traceDecay, stepSize);
				
				setPreviousStateAction(newCurrentState, newBestAction, stateActions.get(new StateAction(currentState, bestAction)));
				currentState=newCurrentState;
				bestAction=newBestAction;
			}
			
			for(Object[] stateActionInfo: encounteredStateActions.values())
			{
				((StateAction)stateActionInfo[0]).eligibility=0.0;
			}
			
			System.out.println("Took "+actions+" actions Episode "+numberEpisodesCompleted);
		}
	}
	
	private double getActionStateValue(State state, Action action)
	{
		return policy.getStateActionValue(state, action);
	}
	
	private double getActionStateEligibility(State state, Action action)
	{
		StateAction stateAction=new StateAction(state, action);
		if(stateActions.get(stateAction)!=null)
		{
			return stateActions.get(stateAction).getEligibility();
		}
		else
		{
			return 0.0;
		}
	}
	
	private void setActionStateValue(State state, Action action, double value)
	{
		policy.setStateActionValue(state, action, value);
	}
	
	private void setActionStateEligibility(State state, Action action, double eligibility)
	{
		StateAction stateAction=new StateAction(state, action);
		if(stateActions.get(stateAction)!=null)
		{
			stateActions.get(stateAction).setEligibility(eligibility);
		}
		else
		{
			stateAction.setEligibility(eligibility);
			stateActions.put(stateAction, stateAction);
		}
	}
	
	private void setPreviousStateAction(State state, Action action, StateAction preiousStateAction)
	{
		StateAction stateAction=new StateAction(state, action);
		if(stateActions.get(stateAction)!=null)
		{
			stateActions.get(stateAction).setPreviousStateAction(preiousStateAction);
		}
		else
		{
			stateAction.setPreviousStateAction(preiousStateAction);
			stateActions.put(stateAction, stateAction);
		}
	}

}
