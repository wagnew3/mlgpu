package ReinforcementMachineLearningFramework;

public abstract class LimitedActionsEnvironment extends Environment
{
	
	public abstract StateAction[] getAllPossibleStateActions(State currentState);

}
