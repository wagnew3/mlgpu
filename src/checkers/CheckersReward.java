package checkers;

import java.util.Arrays;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class CheckersReward extends RewardFunction
{
	 //0=black, 1=red
	public double getReward(State previousState, State currentState, Action action) 
	{
		if(((CheckersState)currentState).isEndState())
		{
			if(Arrays.deepEquals(((CheckersState)currentState).getBoardState(), ((CheckersAction)action).boardChanges)
					|| (action!=null && Arrays.deepEquals(((CheckersState)currentState).getBoardState(), ((CheckersAction)action).boardChanges))) //made winning move
			{
				return 0.25;
			}
			else
			{
				return -0.25;
			}
		}
		else
		{
			if(action!=null)
			{
				int playerNumber=getPlayerNumber(previousState, action);
				if(playerNumber!=-1)
				{
					double valThisOld=numberPieces(playerNumber, previousState);
					double valOtherOld=numberPieces((playerNumber+1)%2, previousState);
					
					double valThisNew=numberPieces(playerNumber, currentState);
					double valOtherNew=numberPieces((playerNumber+1)%2, currentState);
					
					if(0.01*((valThisNew-valThisOld)+(valOtherOld-valOtherNew))!=0.0)
					{
						int u=0;
					}
					return 0.01*((valThisNew-valThisOld)+(valOtherOld-valOtherNew));
				}
				else
				{
					return 0.0;
				}
			}
			else
			{
				return 0.0;
			}
		}
	}
	
	protected int getPlayerNumber(State state, Action action)
	{
		CheckersState cState=(CheckersState)state;
		CheckersAction cAction=(CheckersAction)action;
		
		for(int rowInd=0; rowInd<cState.getBoardState().length; rowInd++)
		{
			for(int colInd=0; colInd<cState.getBoardState()[rowInd].length; colInd++)
			{
				if(cState.getBoardState()[rowInd][colInd]==0
						&& cAction.boardChanges[rowInd][colInd]!=0)
				{
					if(cAction.boardChanges[rowInd][colInd]<0)
					{
						return 0;
					}
					else
					{
						return 1;
					}
				}
			}
		}
		return -1;
	}
	
	protected double numberPieces(int playerNumber, State state)
	{
		double numberPieces=0.0;
		CheckersState cState=(CheckersState)state;
		for(int rowInd=0; rowInd<cState.getBoardState().length; rowInd++)
		{
			for(int colInd=0; colInd<cState.getBoardState()[rowInd].length; colInd++)
			{
				if(playerNumber==0) //black
				{
					if(cState.getBoardState()[rowInd][colInd]==-2)
					{
						numberPieces+=1.5;
					}
					else if(cState.getBoardState()[rowInd][colInd]==-1)
					{
						numberPieces+=1.0;
					}
				}
				else //red
				{
					if(cState.getBoardState()[rowInd][colInd]==2)
					{
						numberPieces+=1.5;
					}
					else if(cState.getBoardState()[rowInd][colInd]==1)
					{
						numberPieces+=1.0;
					}
				}
			}
		}
		return numberPieces;
	}

}
