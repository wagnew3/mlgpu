package offPolicyMonteCarlo;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.State;

public class CarRaceState extends State
{

	public static final int numberVelocities=11;
	public static final int numberXPositions=6;
	public static final int numberYPositions=1;
	
	public int x;
	public int y;
	
	public int xVel;
	public int yVel;
	
	public CarRaceState previousState;
	public CarRaceAction actionTaken;
	
	public double reward=0.0;
	
	public CarRaceState(int ID, CarRaceState previousState)
	{
		createFromID(ID);
		this.previousState=previousState;
	}
	
	public CarRaceState(int x, int y, int xVel, int yVel, CarRaceState previousState, CarRaceAction actionTaken)
	{
		this.x=x;
		this.y=y;
		this.xVel=xVel;
		this.yVel=yVel;
		this.previousState=previousState;
		this.actionTaken=actionTaken;
	}
	
	@Override
	public int getID() 
	{
		return x+y*numberXPositions+(xVel+numberVelocities/2)*numberXPositions*numberYPositions+(yVel+numberVelocities/2)*numberXPositions*numberYPositions*numberVelocities;
	}

	@Override
	public void createFromID(int ID) 
	{
		x=ID%numberXPositions;
		y=(ID/numberXPositions)%numberYPositions;
		
		xVel=((ID/(numberXPositions*numberYPositions))%numberVelocities)-numberVelocities/2;
		yVel=((ID/(numberXPositions*numberYPositions*numberVelocities))%numberVelocities)-numberVelocities/2;
	}

	@Override
	public Action getActionTaken() 
	{
		return actionTaken;
	}

	@Override
	public State getPreviousState() 
	{
		return previousState;
	}

}
