package tempDiffLambdaRMLTesuaro;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicyTesuaro;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.State;
import hc.HCState;

public class TempDiffLambdaRMLNNTesuaro extends ReinforcementLearner
{

	Environment enviroment;
	NeuralEGreedyPolicyTesuaro policy;
	State startState;
	int numberEpisodes;
	double stepSize; //a
	double discountRate; //y
	double traceDecay; //lambda
	Hashtable<StateAction, StateAction> stateActions;
	
	public TempDiffLambdaRMLNNTesuaro(Environment enviroment, 
			NeuralEGreedyPolicyTesuaro policy,
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
		double totalRewardAllEpisodes=0.0;
		boolean repeatActions=false;
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			enviroment.setStartState(startState);
			State currentState=enviroment.getCurrentState();
			Action bestAction=policy.getKthBestAction(currentState, 0, enviroment);
			
			List<StateAction> encounteredStateActions=new ArrayList<>(); //[0]=StateAction, [1]=ActionListElement
			//encounteredStateActions.add(new StateAction(currentState, bestAction));
			
			if(bestAction.getValue()[0]==1.0)
			{
				int u=0;
			}
			
			int actions=0;
			
			double totalRewardThisEpisode=0.0;
			
			
			while(!enviroment.getCurrentState().isEndState())
			{
				actions++;
				double reward=enviroment.takeAction(bestAction);
				totalRewardThisEpisode+=reward;
				State newCurrentState=enviroment.getCurrentState();
				Action newBestAction=policy.getKthBestAction(newCurrentState, 0, enviroment);
				double diff=reward+discountRate*getActionStateValue(newCurrentState, newBestAction)
							-getActionStateValue(currentState, bestAction);
				setActionStateEligibility(currentState, bestAction,
						getActionStateEligibility(currentState, bestAction)+1);
				
				
				
				
				StateAction updateStateAction=stateActions.get(new StateAction(currentState, bestAction));
				encounteredStateActions.add(updateStateAction);
				if(encounteredStateActions.size()>100)
				{
					encounteredStateActions.remove(0);
				}
				
				policy.trainNN(encounteredStateActions, diff);
				
				setPreviousStateAction(newCurrentState, newBestAction, stateActions.get(new StateAction(currentState, bestAction)));
				currentState=newCurrentState;
				bestAction=newBestAction;
			}
			//totalRewardThisEpisode
			
			System.out.println("Took "+actions+" actions Episode "+numberEpisodesCompleted);
			
			HCState state=(HCState)startState;
			StateAction[] allPossibleActions=((LimitedActionsEnvironment)enviroment).getAllPossibleStateActions(state);
			for(StateAction possibleStateAction: allPossibleActions)
			{
				double value=((NeuralEGreedyPolicyTesuaro)policy).network.getOutput(new ArrayRealVector(possibleStateAction.getState().getValue())).getEntry(0);
				System.out.println(value);
			}
			
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
