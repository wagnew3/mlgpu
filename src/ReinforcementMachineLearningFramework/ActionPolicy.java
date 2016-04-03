package ReinforcementMachineLearningFramework;

import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

public abstract class ActionPolicy 
{
	
	public abstract Action getKthBestAction(State state, int k, Environment environment);
	
	public abstract void setStateActionValue(State state, Action action, double value);
	
	public abstract double getStateActionValue(State state, Action action);
	
	public abstract ActionListElement getActionListElement(State state, Action action);
	
	public abstract Set<Entry<State, Object[]>> getStateActionValues();

}
