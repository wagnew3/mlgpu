package offPolicyMonteCarlo;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;

public class CarRaceEnvironment extends Environment
{

	public int[][] track;
	private int maxXSpeed;
	private int maxYSpeed;
	
	private CarRaceState currentState;
	private CarRaceState startState;
	
	public CarRaceEnvironment(int[][] track, int maxXSpeed, int maxYSpeed)
	{
		this.track=track;
		this.maxXSpeed=maxXSpeed;
		this.maxYSpeed=maxYSpeed;
	}
	
	@Override
	public State getCurrentState() 
	{
		return currentState;
	}

	@Override
	public int step(ActionPolicy policy) 
	{
		CarRaceAction action=(CarRaceAction)policy.getAction(currentState);
		CarRaceState nextState=new CarRaceState(currentState.x+currentState.xVel+action.xAccel,
						currentState.y+currentState.yVel+action.yAccel,
						(int)Math.round(Math.signum(currentState.xVel+action.xAccel)*Math.min(Math.abs(currentState.xVel+action.xAccel), maxXSpeed)),
						(int)Math.round(Math.signum(currentState.yVel+action.yAccel)*Math.min(Math.abs(currentState.yVel+action.yAccel), maxYSpeed)),
						currentState,
						action);
		if(!isOnTrack(nextState))
		{
			CarRaceState resetState=new CarRaceState(currentState.x,
					currentState.y,
					0,
					0,
					nextState,
					null);
			
			/*
			int offTrackX=resetState.x;
			int offTrackY=resetState.y;
			int moveAmount=0;
			boolean onTrack=false;
			
			while(!onTrack)
			{
				for(int xPos=-moveAmount; xPos<=moveAmount; xPos++)
				{
					resetState.x=offTrackX+xPos;
					resetState.y=offTrackY+moveAmount;
					if(isOnTrack(resetState))
					{
						onTrack=true;
						break;
					}
					resetState.y=offTrackY-moveAmount;
					if(isOnTrack(resetState))
					{
						onTrack=true;
						break;
					}
				}
				if(onTrack)
				{
					break;
				}
				
				for(int yPos=-moveAmount; yPos<=moveAmount; yPos++)
				{
					resetState.y=offTrackY+yPos;
					resetState.x=offTrackX+moveAmount;
					if(isOnTrack(resetState))
					{
						onTrack=true;
						break;
					}
					resetState.x=offTrackX-moveAmount;
					if(isOnTrack(resetState))
					{
						onTrack=true;
						break;
					}
				}
				if(onTrack)
				{
					break;
				}
				moveAmount++;
			}
			*/
			currentState=resetState;
			if(finished(currentState.previousState.previousState, currentState))
			{
				return -1;
			}
			else
			{
				return 0;
			}
		}
		else
		{
			currentState=nextState;
			if(finished(currentState.previousState, currentState))
			{
				return -1;
			}
			else
			{
				return 0;
			}
		}
	}

	@Override
	public void setStartState(State startState)
	{
		this.startState=(CarRaceState)startState;
		this.currentState=(CarRaceState)startState;
	}

	@Override
	public void reset() 
	{
		currentState=startState;
	}

	@Override
	public double transistionProbability(State oldState, State newState, Action action) 
	{
		// TODO Auto-generated method stub
		return 0;
	}
	
	public boolean isOnTrack(CarRaceState position)
	{
		try
		{
		if(position.x>=track.length || position.x<0
			|| position.y>=track[0].length || position.y<0)
		{
			return false;
		}
		else if(track[position.x][position.y]==-1)
		{
			return false;
		}
		else
		{
			return true;
		}
	}
	catch(Exception e)
	{
		e.printStackTrace();
	}
		return false;
	}
	
	public boolean finished(CarRaceState oldPosition, CarRaceState currentPosition)
	{
		if(oldPosition.x==currentPosition.x && oldPosition.y==currentPosition.y)
		{
			return (track[currentPosition.x][currentPosition.y]==1);
		}
		double xSlope=(double)(currentPosition.y-oldPosition.y)/(double)(currentPosition.x-oldPosition.x);
		if(currentPosition.x-oldPosition.x!=0 && xSlope<=1)
		{
			for(int xPos=0; Math.abs(xPos)<=Math.abs(currentPosition.x-oldPosition.x); xPos+=Math.signum(currentPosition.x-oldPosition.x))
			{
				try
				{
					if(track[oldPosition.x+xPos][(int)(oldPosition.y+Math.round(xPos*xSlope))]==1)
					{
						return true;
					}
				}
				catch(Exception e)
				{
					e.printStackTrace();
				}
			}
		}
		else
		{
			double ySlope=(double)(currentPosition.x-oldPosition.x)/(double)(currentPosition.y-oldPosition.y);
			for(int yPos=0; Math.abs(yPos)<=Math.abs(currentPosition.y-oldPosition.y); yPos+=Math.signum(currentPosition.y-oldPosition.y))
			{
				if(track[(int)(oldPosition.x+Math.round(yPos*ySlope))][oldPosition.y+yPos]==1)
				{
					return true;
				}
			}
		}
		if(track[currentPosition.x][currentPosition.y]==1)
		{
			int u=0;
		}
		return false;
	}

}
