package ReinforcementMachineLearningFramework;

public abstract class RewardFunctionWithHeuristic extends RewardFunction
{
	public abstract double getHeuristicReward(State previousState, State currentState, Action action);
}
