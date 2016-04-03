package cardGame4540;

import ReinforcementMachineLearningFramework.Action;

public class CardGameAction extends Action
{
	
	public int action;
	
	public CardGameAction(int action)
	{
		this.action=action;
	}

	@Override
	public void setActionFromID(int ID) 
	{
		action=ID;
	}

	@Override
	public int getActionID() 
	{
		return action;
	}

}
