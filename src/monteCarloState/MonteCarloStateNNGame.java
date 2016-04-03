package monteCarloState;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.NeuralMCTSPolicy;
import ReinforcementMachineLearningFramework.NeuralStatePolicy;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;
import ReinforcementMachineLearningFramework.StatePolicy;
import boardGame.BoardEnvironment;
import checkers.CheckersReward;
import network.Network;
import network.SplitNetwork;

public class MonteCarloStateNNGame extends MonteCarloState
{
	
	protected static final File netSaveLocationA=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/neuralNetACheckers8x8");
	protected static final File netSaveLocationB=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/neuralNetBCheckers8x8");
	
	protected static final File policySaveLocationA=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/policyACheckers8x8");
	protected static final File policySaveLocationB=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/policyBCheckers8x8");
	
	protected static final int searchesPerState=5;
	
	NeuralStatePolicy[] policies;
	volatile boolean[] noActions;
	volatile boolean[] finished;
	RewardFunction rewardFunction;
	volatile int state; //0=thread0's turn, 1=thread1's turn, 2=thread0 finished, 3=thread0 finished, 4=thread0 and thread1 finished
	//0=finish turn
	//1=finish game
	
	
	public MonteCarloStateNNGame(BoardEnvironment enviroment, 
			NeuralStatePolicy[] policies,
			State startState,
			int numberEpisodes,
			RewardFunction rewardFunction)
	{
		super(enviroment, null, startState, numberEpisodes);
		this.policies=policies;
		noActions=new boolean[2];
		finished=new boolean[2];
		this.rewardFunction=rewardFunction;
	}
	
	/*
	public void learn(int numberPlayers) 
	{
		Hashtable<State, Double> totalReturns[]=new Hashtable[numberPlayers];
		for(int tracesIndex=0; tracesIndex<totalReturns.length; tracesIndex++)
		{
			totalReturns[tracesIndex]=new Hashtable<State, Double>();
		}
		
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			enviroment.setStartState(startState);
			
			boolean noAction=false;
			int actions=0;
			
			List<State> encounteredStates[]=new ArrayList[numberPlayers];
			for(int tracesIndex=0; tracesIndex<encounteredStates.length; tracesIndex++)
			{
				encounteredStates[tracesIndex]=new ArrayList<>();
			}
			
			List<Double> rewards[]=new ArrayList[numberPlayers];
			for(int tracesIndex=0; tracesIndex<rewards.length; tracesIndex++)
			{
				rewards[tracesIndex]=new ArrayList<>();
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
					previousStateActions[playerIndex]=makeMove(playerIndex,
							encounteredStates[playerIndex], rewards[playerIndex],
							previousStateActions[playerIndex]);
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
							encounteredStates[(playerIndex+finishedPlayerIndex)%numberPlayers], 
							rewards[(playerIndex+finishedPlayerIndex)%numberPlayers],
							previousStateActions[(playerIndex+finishedPlayerIndex)%numberPlayers]);
				}
			}
			
			for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
			{
				Hashtable<State, State> updatedStates=new Hashtable<>();
				for(int stateInd=0; stateInd<encounteredStates[playerIndex].size(); stateInd++)
				{
					if(updatedStates.get(encounteredStates[playerIndex].get(stateInd))==null)
					{
						double returnValue=0.0;
						for(int returnInd=stateInd; returnInd<encounteredStates[playerIndex].size(); returnInd++)
						{
							returnValue+=rewards[playerIndex].get(returnInd);
						}
						double oldStateValue=getStateValue(encounteredStates[playerIndex].get(stateInd), playerIndex);
						double oldNumberUpdates=getTotalReturns(encounteredStates[playerIndex].get(stateInd), totalReturns[playerIndex]);
						double newStateValue=(oldStateValue*oldNumberUpdates+returnValue)/(oldNumberUpdates+1.0);
						setStateValue(encounteredStates[playerIndex].get(stateInd), newStateValue, playerIndex);
						incTotalReturns(encounteredStates[playerIndex].get(stateInd), totalReturns[playerIndex]);
					}
					updatedStates.put(encounteredStates[playerIndex].get(stateInd), encounteredStates[playerIndex].get(stateInd));
				}
			}
			
			if(numberEpisodesCompleted%500000==0 && numberEpisodesCompleted>0)
			{
				for(int playerInd=0; playerInd<numberPlayers; playerInd++)
				{
					((NeuralMCTSPolicy)policies[playerInd]).trim();
				}
			}
			
			if(numberEpisodesCompleted%(numberEpisodes-1)==0 && numberEpisodesCompleted>0)
			{
				for(int playerInd=0; playerInd<numberPlayers; playerInd++)
				{
					policies[playerInd].trainNN();
					if(playerInd==0)
					{
						SplitNetwork.saveNetwork(netSaveLocationA, policies[playerInd].network);
						StatePolicy.savePolicy(policySaveLocationA, policies[playerInd]);
					}
					else
					{
						SplitNetwork.saveNetwork(netSaveLocationB, policies[playerInd].network);
						StatePolicy.savePolicy(policySaveLocationB, policies[playerInd]);
					}
				}
			}
			
			if(numberEpisodesCompleted%1000==0)
			{
				System.out.println("Took "+actions+" actions Episode "+numberEpisodesCompleted);
			}
		}
	}
	*/

	public void learn(int numberPlayers) 
	{
		Hashtable<State, Double>[] totalReturns=new Hashtable[numberPlayers];
		for(int tracesIndex=0; tracesIndex<totalReturns.length; tracesIndex++)
		{
			totalReturns[tracesIndex]=new Hashtable<State, Double>();
		}
		
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			enviroment.setStartState(startState);
			
			boolean noAction=false;
			int actions=0;
			
			List<State> encounteredStates[]=new ArrayList[numberPlayers];
			State[] oldEnvStates=new State[2];
			for(int tracesIndex=0; tracesIndex<encounteredStates.length; tracesIndex++)
			{
				encounteredStates[tracesIndex]=new ArrayList<>();
				oldEnvStates[tracesIndex]=enviroment.getCurrentState();
			}
			
			List<Double> rewards[]=new ArrayList[numberPlayers];
			for(int tracesIndex=0; tracesIndex<rewards.length; tracesIndex++)
			{
				rewards[tracesIndex]=new ArrayList<>();
			}
			
			StateAction[] previousStateActions=new StateAction[numberPlayers];
			for(int stateActionIndex=0; stateActionIndex<previousStateActions.length; stateActionIndex++)
			{
				previousStateActions[stateActionIndex]=new StateAction(enviroment.getCurrentState(), null);
			}
			
			int finishedPlayerIndex=-1;
			while(!enviroment.getCurrentState().isEndState() && !noAction)
			{
				//((BoardEnvironment)enviroment).displayBoard();
				actions++;
				for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
				{
					State currentState=enviroment.getCurrentState();
					
					for(int numberSearches=0; numberSearches<searchesPerState; numberSearches++)
					{
						learnFromState(numberPlayers, currentState, playerIndex);
						enviroment.setStartState(currentState);
						((BoardEnvironment)enviroment).turn=playerIndex;
					}
					
					State oldState=enviroment.getCurrentState();
					previousStateActions[playerIndex]=makeMove(playerIndex,
							encounteredStates[playerIndex], rewards[playerIndex],
							previousStateActions[playerIndex], oldEnvStates[playerIndex]);
					oldEnvStates[playerIndex]=oldState;
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
					
					
					
				}
				if(actions>2000)
				{
					noAction=true;
					break;
				}
			}
			
			//((BoardEnvironment)enviroment).displayBoard();
			
			if(!noAction)
			{
				for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
				{
					finalUpdate((playerIndex+finishedPlayerIndex)%numberPlayers, 
							encounteredStates[(playerIndex+finishedPlayerIndex)%numberPlayers], 
							rewards[(playerIndex+finishedPlayerIndex)%numberPlayers],
							previousStateActions[(playerIndex+finishedPlayerIndex)%numberPlayers]);
				}
			}
			
			for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
			{
				Hashtable<State, State> updatedStates=new Hashtable<>();
				for(int stateInd=0; stateInd<encounteredStates[playerIndex].size(); stateInd++)
				{
					if(updatedStates.get(encounteredStates[playerIndex].get(stateInd))==null)
					{
						double returnValue=0.0;
						for(int returnInd=stateInd; returnInd<encounteredStates[playerIndex].size(); returnInd++)
						{
							returnValue+=rewards[playerIndex].get(returnInd);
						}
						double oldStateValue=getStateValue(encounteredStates[playerIndex].get(stateInd), playerIndex);
						double oldNumberUpdates=getTotalReturns(encounteredStates[playerIndex].get(stateInd), totalReturns[playerIndex]);
						double newStateValue=(oldStateValue*oldNumberUpdates+returnValue)/(oldNumberUpdates+1.0);
						setStateValue(encounteredStates[playerIndex].get(stateInd), newStateValue, playerIndex);
						incTotalReturns(encounteredStates[playerIndex].get(stateInd), totalReturns[playerIndex]);
					}
					updatedStates.put(encounteredStates[playerIndex].get(stateInd), encounteredStates[playerIndex].get(stateInd));
				}
			}
			
			if(numberEpisodesCompleted%500000==0 && numberEpisodesCompleted>0)
			{
				for(int playerInd=0; playerInd<numberPlayers; playerInd++)
				{
					((NeuralMCTSPolicy)policies[playerInd]).trim();
				}
			}
			
			if(numberEpisodesCompleted%(numberEpisodes-1)==0 && numberEpisodesCompleted>0)
			{
				
				for(int playerInd=0; playerInd<numberPlayers; playerInd++)
				{
					if(playerInd==0)
					{
						StatePolicy.savePolicy(policySaveLocationA, policies[playerInd]);
					}
					else
					{
						StatePolicy.savePolicy(policySaveLocationB, policies[playerInd]);
					}
				}
				for(int playerInd=0; playerInd<numberPlayers; playerInd++)
				{
					if(playerInd==0)
					{
						//policies[playerInd].network=SplitNetwork.loadNetwork(netSaveLocationA);
						policies[playerInd].trainNN();
						//SplitNetwork.saveNetwork(netSaveLocationA, policies[playerInd].network);
					}
					else
					{
						//policies[playerInd].network=SplitNetwork.loadNetwork(netSaveLocationB);
						policies[playerInd].trainNN();
						SplitNetwork.saveNetwork(netSaveLocationB, policies[playerInd].network);
					}
				}
			}
			
			if(numberEpisodesCompleted%1000==0)
			{
				System.out.println("Took "+actions+" actions Episode "+numberEpisodesCompleted);
			}
		}
	}
	
	protected void learnFromState(int numberPlayers, State startLearnState, int turn)
	{
		Hashtable<State, Double>[] totalReturns=new Hashtable[numberPlayers];
		for(int tracesIndex=0; tracesIndex<totalReturns.length; tracesIndex++)
		{
			totalReturns[tracesIndex]=new Hashtable<State, Double>();
		}
		
		boolean noAction=false;
		int actions=0;
		
		List<State> encounteredStates[]=new ArrayList[numberPlayers];
		State[] oldEnvStates=new State[2];
		for(int tracesIndex=0; tracesIndex<encounteredStates.length; tracesIndex++)
		{
			encounteredStates[tracesIndex]=new ArrayList<>();
			oldEnvStates[tracesIndex]=enviroment.getCurrentState();
		}
		
		List<Double> rewards[]=new ArrayList[numberPlayers];
		for(int tracesIndex=0; tracesIndex<rewards.length; tracesIndex++)
		{
			rewards[tracesIndex]=new ArrayList<>();
		}
		
		StateAction[] previousStateActions=new StateAction[numberPlayers];
		for(int stateActionIndex=0; stateActionIndex<previousStateActions.length; stateActionIndex++)
		{
			previousStateActions[stateActionIndex]=new StateAction(enviroment.getCurrentState(), null);
		}
		
		int finishedPlayerIndex=-1;
		while(!enviroment.getCurrentState().isEndState() && !noAction)
		{
			//((BoardEnvironment)enviroment).displayBoard();
			actions++;
			for(int playerIndex=turn; playerIndex-turn<numberPlayers; playerIndex++)
			{
				State oldState=enviroment.getCurrentState();
				previousStateActions[playerIndex%numberPlayers]=makeMove(playerIndex%numberPlayers,
						encounteredStates[playerIndex%numberPlayers], rewards[playerIndex%numberPlayers],
						previousStateActions[playerIndex%numberPlayers], oldEnvStates[playerIndex%numberPlayers]);
				oldEnvStates[playerIndex%numberPlayers]=oldState;
				if(previousStateActions[playerIndex%numberPlayers]==null)
				{
					noAction=true;
					break;
				}
				else if(enviroment.getCurrentState().isEndState())
				{
					finishedPlayerIndex=playerIndex%numberPlayers;
					break;
				}
			}
			if(actions>2000)
			{
				noAction=true;
				break;
			}
		}

		//((BoardEnvironment)enviroment).displayBoard();
		if(!noAction && actions>1)
		{
			for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
			{
				finalUpdate((playerIndex+finishedPlayerIndex)%numberPlayers, 
						encounteredStates[(playerIndex+finishedPlayerIndex)%numberPlayers], 
						rewards[(playerIndex+finishedPlayerIndex)%numberPlayers],
						previousStateActions[(playerIndex+finishedPlayerIndex)%numberPlayers]);
			}
		}
		
		for(int playerIndex=0; playerIndex<numberPlayers; playerIndex++)
		{
			Hashtable<State, State> updatedStates=new Hashtable<>();
			for(int stateInd=0; stateInd<encounteredStates[playerIndex].size(); stateInd++)
			{
				if(updatedStates.get(encounteredStates[playerIndex].get(stateInd))==null)
				{
					double returnValue=0.0;
					for(int returnInd=stateInd; returnInd<encounteredStates[playerIndex].size(); returnInd++)
					{
						returnValue+=rewards[playerIndex].get(returnInd);
					}
					double oldStateValue=getStateValue(encounteredStates[playerIndex].get(stateInd), playerIndex);
					double oldNumberUpdates=getTotalReturns(encounteredStates[playerIndex].get(stateInd), totalReturns[playerIndex]);
					double newStateValue=(oldStateValue*oldNumberUpdates+returnValue)/(oldNumberUpdates+1.0);
					setStateValue(encounteredStates[playerIndex].get(stateInd), newStateValue, playerIndex);
					incTotalReturns(encounteredStates[playerIndex].get(stateInd), totalReturns[playerIndex]);
				}
				updatedStates.put(encounteredStates[playerIndex].get(stateInd), encounteredStates[playerIndex].get(stateInd));
			}
		}
	}
	
	 //-1: update at end
	public StateAction makeMove(int playerNumber, List<State> encounteredStates,
			List<Double> rewards, StateAction oldStateAction, State oldState)
	{
		double reward=rewardFunction.getReward(oldState, enviroment.getCurrentState(), oldStateAction.action);
		if(reward!=0 && Math.abs(reward)<0.25)
		{
			int o=0;
		}
		Action newBestAction=policies[playerNumber].getAction(enviroment.getCurrentState(), enviroment);
		if(newBestAction==null)
		{
			return null;
		}	
		enviroment.takeAction(newBestAction, playerNumber);
		
		encounteredStates.add(oldStateAction.state);
		rewards.add(reward);
		
		State newCurrentState=enviroment.getCurrentState();
		return new StateAction(newCurrentState, newBestAction);
	}
	
	public void finalUpdate(int playerNumber, List<State> encounteredStates,
			List<Double> rewards, StateAction oldStateAction)
	{
		double reward=rewardFunction.getReward(oldStateAction.state, enviroment.getCurrentState(), oldStateAction.action);
		encounteredStates.add(oldStateAction.state);
		rewards.add(reward);
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
			
		((BoardEnvironment)enviroment).displayBoard();
		
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
					((BoardEnvironment)enviroment).humanMove(playerIndex);
				}
				else
				{
					makeMoveNoLearning(playerIndex, 3);
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
				((BoardEnvironment)enviroment).displayBoard();
				
			}
		}
		((BoardEnvironment)enviroment).displayBoard();
			
	}
	
	public void makeMoveNoLearning(int playerNumber, int searchDepth)
	{
		//Action newBestAction=policies[playerNumber].getBestAction(enviroment.getCurrentState(), enviroment);
		Action newBestAction=(Action)findWorstCaseStateValue(playerNumber, searchDepth, (LimitedActionsEnvironment)enviroment.clone())[1];
		if(newBestAction==null)
		{
			return;
		}	
		enviroment.takeAction(newBestAction, playerNumber);	
	}
	
	public Object[] findWorstCaseStateValue(int playerNumber, int searchDepth, LimitedActionsEnvironment possibleEnv)
	{
		int numberActionsToSearchOn=50;
		double discount=0.00001;
		double bestReward=Double.NEGATIVE_INFINITY;
		double tempReward;
		Action bestAction=null;
		List<StateAction> topStateActions=getTopNActions(playerNumber, numberActionsToSearchOn, possibleEnv);
		for(StateAction stateAction: topStateActions)
		{
			LimitedActionsEnvironment newEnv=(LimitedActionsEnvironment)possibleEnv.clone();
			tempReward=newEnv.takeAction(stateAction.action, playerNumber);
			tempReward+=((NeuralMCTSPolicy)policies[playerNumber]).getStateValueNet(newEnv.getCurrentState());
			if(!newEnv.getCurrentState().isEndState() && searchDepth>0)
			{
				double futureReward=Double.MAX_VALUE;
				List<StateAction> topStateActionsOtherPlayer=getTopNActions((playerNumber+1)%2, numberActionsToSearchOn, newEnv);
				for(StateAction otherPlayerStateAction: topStateActionsOtherPlayer)
				{
					LimitedActionsEnvironment newOPEnv=(LimitedActionsEnvironment)newEnv.clone();
					newOPEnv.takeAction(otherPlayerStateAction.action, (playerNumber+1)%2);
					double possibleMinReward=(double)findWorstCaseStateValue(playerNumber, searchDepth-1, newOPEnv)[0];
					if(possibleMinReward<futureReward)
					{
						futureReward=possibleMinReward;
					}
				}
				tempReward=futureReward;
			}
			if(tempReward>bestReward)
			{
				bestReward=tempReward;
				bestAction=stateAction.action;
			}
		}
		
		return new Object[]{bestReward-discount, bestAction};
	}
	
	//[0]=List<StateAction>, [1]=List<Double> rewards
	protected List<StateAction> getTopNActions(int playerNumber, int n, LimitedActionsEnvironment possibleEnv)
	{
		List<StateAction> bestStateActions=new ArrayList<>();
		List<Double> highestValues=new ArrayList<>();
		StateAction[] possibleActions=possibleEnv.getAllPossibleStateActions(possibleEnv.getCurrentState());
		for(StateAction stateAction: possibleActions)
		{
			LimitedActionsEnvironment newEnv=(LimitedActionsEnvironment)possibleEnv.clone();
			double tempReward=newEnv.takeAction(stateAction.action, playerNumber);
			tempReward+=((NeuralMCTSPolicy)policies[playerNumber]).getStateValueNet(newEnv.getCurrentState());
			
			int place=Collections.binarySearch(highestValues, tempReward);
			if(place<0)
			{
				place=-(place+1);
			}
			if(place<n)
			{
				highestValues.add(place, tempReward);
				bestStateActions.add(place, stateAction);
				
				if(highestValues.size()>n)
				{
					highestValues.remove(highestValues.size()-1);
					bestStateActions.remove(bestStateActions.size()-1);
				}
			}
		}
		return bestStateActions;
	}
	
	private double getStateValue(State state, int playerNumber)
	{
		return policies[playerNumber].getStateValue(state);
	}
	
	private void setStateValue(State state, double value, int playerNumber)
	{
		policies[playerNumber].setStateValue(state, value);
	}
	
	private double getTotalReturns(State state, Hashtable<State, Double> totalReturns)
	{
		Double value=totalReturns.get(state);
		if(value==null)
		{
			return 0.0;
		}
		else
		{
			return value;
		}
	}
	
	private void incTotalReturns(State state, Hashtable<State, Double> totalReturns)
	{
		totalReturns.put(state, getTotalReturns(state, totalReturns)+1.0);
	}

	@Override
	public void learn() 
	{
		System.out.println("Calling unimplemented TDLNN learn method!");
	}

}
