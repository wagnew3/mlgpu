package cardGame4540;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.State;

public class CardGameState extends State
{
	
	public int numberRedLeft;
	public int numberBlackLeft;
	
	private CardGameAction actionTaken;
	private CardGameState previousState;
	
	public CardGameState(int numberRedLeft, int numberBlackLeft, CardGameAction actionTaken, CardGameState previousState)
	{
		this.numberRedLeft=numberRedLeft;
		this.numberBlackLeft=numberBlackLeft;
		this.actionTaken=actionTaken;
		this.previousState=previousState;
	}

	@Override
	public int getID() 
	{
		return numberRedLeft+27*numberBlackLeft;
	}

	@Override
	public void createFromID(int ID) 
	{
		numberRedLeft=ID%27;
		numberBlackLeft=ID/27;
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
