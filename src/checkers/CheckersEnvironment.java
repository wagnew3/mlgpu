package checkers;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;
import boardGame.BoardEnvironment;

public class CheckersEnvironment extends BoardEnvironment
{

	protected CheckersReward rewardFunction;
	boolean lastPlayer=false; //black always starts
	
	public CheckersEnvironment(int size, CheckersReward rewardFunction) 
	{
		super(2);
		this.rewardFunction=rewardFunction;
	}

	@Override
	public double takePlayerAction(Action action) 
	{
		CheckersAction checkersAction=(CheckersAction)action;
		double reward=0.0;
		reward=rewardFunction.getReward(currentState, 
				new CheckersState(checkersAction.boardChanges), checkersAction);
		currentState=new CheckersState(checkersAction.boardChanges);
		return reward;
	}
	
	@Override
	public StateAction[] getAllPossibleStateActions(State currentState) 
	{
		CheckersState checkersState=(CheckersState)currentState;
		checkersState.color=turn==1;
		List<CheckersAction> allPossibleActions=new ArrayList<>();
		
		for(int row=0; row<checkersState.getBoardState().length; row++)
		{
			for(int col=0; col<checkersState.getBoardState()[row].length; col++)
			{
				List<CheckersAction> possibleActions=getCheckersActions(checkersState, new int[]{row, col}, false, checkersState.color);
				for(CheckersAction checkerAction: possibleActions)
				{
					for(int colInd=0; colInd<checkerAction.boardChanges[0].length; colInd++) //king
					{
						if(checkerAction.boardChanges[0][colInd]==1)
						{
							checkerAction.boardChanges[0][colInd]=2;
						}
					}
					for(int colInd=0; colInd<checkerAction.boardChanges[checkerAction.boardChanges.length-1].length; colInd++)
					{
						if(checkerAction.boardChanges[checkerAction.boardChanges.length-1][colInd]==-1)
						{
							checkerAction.boardChanges[checkerAction.boardChanges.length-1][colInd]=-2;
						}
					}
				}
				allPossibleActions.addAll(possibleActions);
			}
		}
		
		StateAction[] allPossibleStateActions=new StateAction[allPossibleActions.size()];
		for(int stateActionInd=0; stateActionInd<allPossibleStateActions.length; stateActionInd++)
		{
			allPossibleStateActions[stateActionInd]=
					new StateAction(new CheckersState(allPossibleActions.get(stateActionInd).boardChanges),
							allPossibleActions.get(stateActionInd));
		}
		if(allPossibleStateActions.length==0)
		{
			int u=0;
			for(int row=0; row<checkersState.getBoardState().length; row++)
			{
				for(int col=0; col<checkersState.getBoardState()[row].length; col++)
				{
					List<CheckersAction> possibleActions=getCheckersActions(checkersState, new int[]{row, col}, false, checkersState.color);
				}
			}
		}
		return allPossibleStateActions;
	}
	 //true=red (+), false=black (-)
	private List<CheckersAction> getCheckersActions(CheckersState state, int[] position, boolean jumped, boolean color)
	{
		if(color && state.getBoardState()[position[0]][position[1]]<=0)
		{
			return new ArrayList<CheckersAction>();
		}
		else if(!color && state.getBoardState()[position[0]][position[1]]>=0)
		{
			return new ArrayList<CheckersAction>();
		}
		
		List<CheckersAction> possibleActions=new ArrayList<>();
		
		for(int rowOffset=-2; rowOffset<=2; rowOffset++)
		{
			for(int colOffset=-2; colOffset<=2; colOffset++)
			{
				if(state.getBoardState()[position[0]][position[1]]==-2)
				{
					int u=0;
				}
				if(!jumped
						&& (color || (state.getBoardState()[position[0]][position[1]]==-2 || rowOffset>0))
						&& (!color || (state.getBoardState()[position[0]][position[1]]==2 || rowOffset<0))
						&& Math.abs(rowOffset)==1 && Math.abs(colOffset)==1
						&& position[0]+rowOffset>=0
						&& state.getBoardState().length>position[0]+rowOffset
						&& position[1]+colOffset>=0
						&& state.getBoardState()[0].length>position[1]+colOffset
						&& state.getBoardState()[position[0]+rowOffset][position[1]+colOffset]==0)
				{
				    	byte[][] move=new byte[][]{new byte[]{(byte)(position[0]+rowOffset), (byte)(position[1]+colOffset)}};
					
				    	byte[][] newState=new byte[state.getBoardState().length][state.getBoardState()[0].length];
					for(int rowInd=0; rowInd<newState.length; rowInd++)
					{
						System.arraycopy(state.getBoardState()[rowInd], 0, newState[rowInd], 0, state.getBoardState()[rowInd].length);
					}
					newState[position[0]+rowOffset][position[1]+colOffset]=newState[position[0]][position[1]];
					newState[position[0]][position[1]]=0;
					
					possibleActions.add(new CheckersAction(newState));
				}
				if((color || (state.getBoardState()[position[0]][position[1]]==-2 || rowOffset>0)) //moving in right direction
						&& (!color || (state.getBoardState()[position[0]][position[1]]==2 || rowOffset<0))
						&& Math.abs(rowOffset)==2 //not out of board bounds
						&& Math.abs(colOffset)==2
						&& position[0]+rowOffset>=0
						&& state.getBoardState().length>position[0]+rowOffset
						&& position[1]+colOffset>=0
						&& state.getBoardState()[0].length>position[1]+colOffset
						&& ((color && state.getBoardState()[position[0]+rowOffset/2][position[1]+colOffset/2]<0) //jumping piece of opposite color
								|| (!color && state.getBoardState()[position[0]+rowOffset/2][position[1]+colOffset/2]>0))
						&& state.getBoardState()[position[0]+rowOffset][position[1]+colOffset]==0) //landing in empty square
				{
				    	byte[][] action=new byte[][]{new byte[]{(byte)(position[0]+rowOffset), (byte)(position[1]+colOffset)}};

				    	byte[][] newState=new byte[state.getBoardState().length][state.getBoardState()[0].length];
					for(int rowInd=0; rowInd<newState.length; rowInd++)
					{
						System.arraycopy(state.getBoardState()[rowInd], 0, newState[rowInd], 0, state.getBoardState()[rowInd].length);
					}
					newState[position[0]+rowOffset/2][position[1]+colOffset/2]=0;
					newState[position[0]+rowOffset][position[1]+colOffset]=newState[position[0]][position[1]];
					
					newState[position[0]][position[1]]=0;
					
					List<CheckersAction> recursedActions=
							getCheckersActions(new CheckersState(newState), 
									new int[]{position[0]+rowOffset, position[1]+colOffset}, true, color);
					possibleActions.addAll(recursedActions);
					possibleActions.add(new CheckersAction(newState));
				}
			}
		}
		
		return possibleActions;
		
	}

	@Override
	public StateAction getRandomStateAction(State state) 
	{
		StateAction[] allPossiblStateActions=getAllPossibleStateActions(state);
		if(allPossiblStateActions.length==0)
		{
			return null;
		}
		return allPossiblStateActions[(int)(allPossiblStateActions.length*Math.random())];
	}

	@Override
	public void humanMove(int playerNumber) //0=blue (negative), 1=red (positive)
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		System.out.print("Enter move x,y>x',y':");
		
		String input=null;
		try 
		{
			input = br.readLine();
		} 
		catch (IOException e1) 
		{
			e1.printStackTrace();
		}
		
		int firstCommaInd=input.indexOf(",");
		int gtInd=input.indexOf(">");
		int secondCommaInd=input.lastIndexOf(",");
		
		if(firstCommaInd==-1 || gtInd==-1 || secondCommaInd==-1)
		{
			System.out.println("Incorrect format!");
			humanMove(playerNumber); 
		}
		
		int x=-1;
		int y=-1;
		int newX=-1;
		int newY=-1;
		
		try
		{
			x=Integer.parseInt(input.substring(0, firstCommaInd));
			y=Integer.parseInt(input.substring(firstCommaInd+1, gtInd));
			newX=Integer.parseInt(input.substring(gtInd+1, secondCommaInd));
			newY=Integer.parseInt(input.substring(secondCommaInd+1));
		}
		catch(NumberFormatException e)
		{
			System.out.println("Incorrect format!");
			humanMove(playerNumber); 
		}
		
		try
		{
			if(!((playerNumber==0 || currentState.getBoardState()[x][y]>0)
					&& (playerNumber==1 || currentState.getBoardState()[x][y]<0)))
			{
				System.out.println("Not moving correct piece!");
				humanMove(playerNumber);
			}
		}
		catch(ArrayIndexOutOfBoundsException e)
		{
			System.out.println("Coordinates not in board!");
			humanMove(playerNumber);
		}
		
		System.out.println("Moving "+currentState.getBoardState()[x][y]+" at "+x+" "+y+" to "+newX+" "+newY);
		
		byte[][] action=new byte[currentState.getBoardState().length][currentState.getBoardState()[0].length];
		for(int rowInd=0; rowInd<action.length; rowInd++)
		{
			System.arraycopy(currentState.getBoardState()[rowInd], 0, action[rowInd], 0, action[rowInd].length);
		}
		
		action[newX][newY]=currentState.getBoardState()[x][y];
		action[x][y]=0;
		
		if(Math.abs(newX-x)==2)
		{
			action[(newX+x)/2][(newY+y)/2]=0;
		}
		
		
		if(newX==action.length-1 && playerNumber==0)
		{
			action[newX][newY]=-2;
		}
		
		if(newX==0 && playerNumber==1)
		{
			action[newX][newY]=2;
		}
				
		takeAction(new CheckersAction(action), playerNumber);
	}

	@Override
	public Environment clone() 
	{
		CheckersEnvironment newEnv=new CheckersEnvironment(-1, rewardFunction);
		newEnv.setStartState(getCurrentState());
		newEnv.turn=turn;
		newEnv.lastPlayer=lastPlayer;
		return newEnv;
	}

}
