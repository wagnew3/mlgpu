package tempDiffLambdaRML;

import java.io.File;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionListElement;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicyTesuaro;
import ReinforcementMachineLearningFramework.NeuralEGreedyStatePolicy;
import ReinforcementMachineLearningFramework.NeuralStatePolicy;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;
import ReinforcementMachineLearningFramework.StatePolicy;
import boardGame.BoardEnvironment;
import hc.HCState;
import network.Network;

public class TempDiffLambdaEMLStateNNGame extends ReinforcementLearner
{

	protected static final File saveLocationA=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/neuralNetA");
	protected static final File saveLocationB=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/neuralNetB");
	
	BoardEnvironment enviroment;
	NeuralStatePolicy[] policies;
	State startState;
	int numberEpisodes;
	double stepSize; //a
	double discountRate; //y
	double traceDecay; //lambda
	volatile boolean[] noActions;
	volatile boolean[] finished;
	RewardFunction rewardFunction;
	volatile int state; //0=thread0's turn, 1=thread1's turn, 2=thread0 finished, 3=thread0 finished, 4=thread0 and thread1 finished
	//0=finish turn
	//1=finish game
	
	
	public TempDiffLambdaEMLStateNNGame(BoardEnvironment enviroment, 
			NeuralStatePolicy[] policies,
			State startState,
			int numberEpisodes,
			double stepSize,
			double discountRate,
			double traceDecay,
			RewardFunction rewardFunction)
	{
		this.enviroment=enviroment;
		this.policies=policies;
		this.startState=startState;
		this.numberEpisodes=numberEpisodes;
		this.stepSize=stepSize;
		this.discountRate=discountRate;
		this.traceDecay=traceDecay;
		noActions=new boolean[2];
		finished=new boolean[2];
		this.rewardFunction=rewardFunction;
	}
	
	public void learn(int numberPlayers) 
	{
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			enviroment.setStartState(startState);
			
			/*
			if(numberEpisodesCompleted>numberEpisodes-1000)
			{
				enviroment.displayBoard();
				int u=0;
			}
			*/
			
			boolean noAction=false;
			int actions=0;
			
			Hashtable<State, Double> eligibilityTraces[]=new Hashtable[numberPlayers];
			for(int tracesIndex=0; tracesIndex<eligibilityTraces.length; tracesIndex++)
			{
				eligibilityTraces[tracesIndex]=new Hashtable<State, Double>();
			}
			
			StateAction[] previousStateActions=new StateAction[numberPlayers];
			for(int stateActionIndex=0; stateActionIndex<previousStateActions.length; stateActionIndex++)
			{
				previousStateActions[stateActionIndex]=new StateAction(enviroment.getCurrentState(), null);
			}
			
			int finishedPlayerIndex=-1;
			while(!enviroment.getCurrentState().isEndState() && !noAction)
			{
				actions++;
				for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
				{
					previousStateActions[playerIndex]=makeMove(playerIndex, eligibilityTraces[playerIndex], previousStateActions[playerIndex]);
					if(previousStateActions[playerIndex]==null)
					{
						noAction=true;
						break;
					}
					else if(enviroment.getCurrentState().isEndState())
					{
						finishedPlayerIndex=playerIndex;
						break;
					}
					
					/*
					if(numberEpisodesCompleted>numberEpisodes-1000)
					{
						enviroment.displayBoard();
						int u=0;
					}
					*/
					
				}
				if(actions>2000)
				{
					noAction=true;
					break;
				}
			}
			
			if(!noAction)
			{
				for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
				{
					finalUpdate((playerIndex+finishedPlayerIndex)%numberPlayers, 
							eligibilityTraces[(playerIndex+finishedPlayerIndex)%numberPlayers], 
							previousStateActions[(playerIndex+finishedPlayerIndex)%numberPlayers]);
				}
			}
			
			if(numberEpisodesCompleted%200000==0 && numberEpisodesCompleted>0)
			{
				for(int playerInd=0; playerInd<numberPlayers; playerInd++)
				{
					policies[playerInd].trainNN();
					if(playerInd==0)
					{
						Network.saveNetwork(saveLocationA, policies[playerInd].network);
						//policies[playerInd].network=Network.loadNetwork(saveLocationA);
					}
					else
					{
						Network.saveNetwork(saveLocationB, policies[playerInd].network);
						//policies[playerInd].network=Network.loadNetwork(saveLocationB);
					}
				}
			}
			
			System.out.println("Took "+actions+" actions Episode "+numberEpisodesCompleted);
		}
	}
	
	 //-1: update at end
	public StateAction makeMove(int playerNumber, Hashtable<State, Double> eligibilityTraces, StateAction oldStateAction)
	{
		double reward=rewardFunction.getReward(oldStateAction.state, enviroment.getCurrentState(), oldStateAction.action);
		Action newBestAction=policies[playerNumber].getAction(enviroment.getCurrentState(), enviroment);
		if(newBestAction==null)
		{
			return null;
		}	
		enviroment.takeAction(newBestAction, playerNumber);
		State newCurrentState=enviroment.getCurrentState();
		
		if(reward!=0.0)
		{
			int u=0;
		}
		
		setStateEligibility(oldStateAction.state, getStateEligibility(oldStateAction.state, eligibilityTraces)+1, eligibilityTraces);
		
		List<State> toRemove=new ArrayList<>();
		for(State state: eligibilityTraces.keySet())
		{
			double diff=reward+discountRate*getStateValue(newCurrentState, playerNumber)-getStateValue(state, playerNumber);
			policies[playerNumber].setStateValue(state, policies[playerNumber].getStateValue(state)
					+stepSize*diff*getStateEligibility(state, eligibilityTraces));
			setStateEligibility(state, discountRate*traceDecay*getStateEligibility(state, eligibilityTraces), eligibilityTraces);
			if(getStateEligibility(state, eligibilityTraces)<0.001)
			{
				toRemove.add(state);
			}
		}
		for(State state: toRemove)
		{
			eligibilityTraces.remove(state);
		}
		
		return new StateAction(newCurrentState, newBestAction);
	}
	
	public void finalUpdate(int playerNumber, Hashtable<State, Double> eligibilityTraces, StateAction oldStateAction)
	{
		double reward=rewardFunction.getReward(oldStateAction.state, enviroment.getCurrentState(), oldStateAction.action);
		State newCurrentState=enviroment.getCurrentState();
		
		if(reward!=0.0)
		{
			int u=0;
		}
		
		setStateEligibility(oldStateAction.state, getStateEligibility(oldStateAction.state, eligibilityTraces)+1, eligibilityTraces);
		
		List<State> toRemove=new ArrayList<>();
		for(State state: eligibilityTraces.keySet())
		{
			double diff=reward+discountRate*getStateValue(newCurrentState, playerNumber)-getStateValue(state, playerNumber);
			policies[playerNumber].setStateValue(state, policies[playerNumber].getStateValue(state)
					+stepSize*diff*getStateEligibility(state, eligibilityTraces));
			setStateEligibility(state, discountRate*traceDecay*getStateEligibility(state, eligibilityTraces), eligibilityTraces);
		}
	}
	
	
	public int run(State initialRunState, int playerNumber)
	{
		enviroment.setStartState(initialRunState);
		State currentState=enviroment.getCurrentState();
		Action bestAction=policies[playerNumber].getAction(currentState, enviroment);
		
		int actions=0;
		
		while(!enviroment.getCurrentState().isEndState())
		{
			actions++;
			enviroment.takeAction(bestAction, playerNumber);
			State newCurrentState=enviroment.getCurrentState();
			Action newBestAction=policies[playerNumber].getBestAction(newCurrentState, enviroment);
			currentState=newCurrentState;
			bestAction=newBestAction;
		}
		return actions;
		//System.out.println("Took "+actions+" actions");
	}
	
	public void playAgainstHuman(int numberPlayers, int humanNumber) 
	{
		enviroment.setStartState(startState);
			
		enviroment.displayBoard();
		
		boolean noAction=false;
		int actions=0;
		
		Hashtable<State, Double> eligibilityTraces[]=new Hashtable[numberPlayers];
		for(int tracesIndex=0; tracesIndex<eligibilityTraces.length; tracesIndex++)
		{
			eligibilityTraces[tracesIndex]=new Hashtable<State, Double>();
		}
		
		StateAction[] previousStateActions=new StateAction[numberPlayers];
		for(int stateActionIndex=0; stateActionIndex<previousStateActions.length; stateActionIndex++)
		{
			previousStateActions[stateActionIndex]=new StateAction(enviroment.getCurrentState(), null);
		}
		
		int finishedPlayerIndex=-1;
		while(!enviroment.getCurrentState().isEndState() && !noAction)
		{
			actions++;
			for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
			{
				if(playerIndex==humanNumber)
				{
					enviroment.humanMove(playerIndex);
				}
				else
				{
					makeMoveNoLearning(playerIndex);
				}
				if(previousStateActions[playerIndex]==null)
				{
					noAction=true;
					break;
				}
				else if(enviroment.getCurrentState().isEndState())
				{
					finishedPlayerIndex=playerIndex;
					break;
				}
				enviroment.displayBoard();
				
			}
		}
		enviroment.displayBoard();
			
	}
	
	public void makeMoveNoLearning(int playerNumber)
	{
		Action newBestAction=policies[playerNumber].getBestAction(enviroment.getCurrentState(), enviroment);
		if(newBestAction==null)
		{
			return;
		}	
		enviroment.takeAction(newBestAction, playerNumber);	
	}
	
	private double getStateValue(State state, int playerNumber)
	{
		return policies[playerNumber].getStateValue(state);
	}
	
	private void setStateValue(State state, double value, int playerNumber)
	{
		policies[playerNumber].setStateValue(state, value);
	}
	
	private double getStateEligibility(State state, Hashtable<State, Double> eligibilityTraces)
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
	
	private void setStateEligibility(State state, double eligibility, Hashtable<State, Double> eligibilityTraces)
	{
		eligibilityTraces.put(state, eligibility);
	}

	@Override
	public void learn() 
	{
		System.out.println("Calling unimplemented TDLNN learn method!");
	}

}
