package ReinforcementMachineLearningFramework;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

public class EGreedyPolicy extends ActionPolicy
{

	double e;
	Hashtable<State, Object[]> stateActions; //[0]=List<ActionListElement>, [1]=Hashtable<ActionListElement, ActionListElement>
	
	public EGreedyPolicy(double exploreChance)
	{
		e=exploreChance;
		stateActions=new Hashtable<>();
	}

	@Override
	public Action getKthBestAction(State state, int k, Environment environment) 
	{
		Object[] stateInfo=stateActions.get(state);
		if(stateInfo==null 
				|| Math.random()<e 
				|| ((List<ActionListElement>)stateInfo[0]).size()<=k
				/*|| ((List<ActionListElement>)stateInfo[0]).get(k).getValue()<=0*/)
		{
			Action action=environment.getRandomAction(state);
			ActionListElement actionListElement=new ActionListElement(action, 0.0);
			putActionListElement(state, actionListElement);
			return action;
		}
		return ((List<ActionListElement>)stateInfo[0]).get(k).getAction();
	}

	@Override
	public void setStateActionValue(State state, Action action, double value) 
	{
		ActionListElement actionListElement=new ActionListElement(action, value);
		putActionListElement(state, actionListElement);
	}

	@Override
	public double getStateActionValue(State state, Action action) 
	{
		ActionListElement actionListElement=new ActionListElement(action, 0.0);
		Object[] stateInfo=stateActions.get(state);
		ActionListElement result=null;
		if(stateInfo==null
				|| (result=((Hashtable<ActionListElement, ActionListElement>)stateInfo[1]).get(actionListElement))==null)
		{
			
			putActionListElement(state, actionListElement);
			return actionListElement.getValue();
		}
		else
		{
			return result.getValue();
		}
	}
	
	protected void putActionListElement(State state, ActionListElement actionListElement)
	{
		Object[] stateInfo=stateActions.get(state);
		if(stateInfo==null)
		{
			List<ActionListElement> actionList=new ArrayList<>();
			actionList.add(actionListElement);
			stateInfo=new Object[2];
			stateInfo[0]=actionList;
			Hashtable<ActionListElement, ActionListElement> actionHashtable=new Hashtable<>();
			actionHashtable.put(actionListElement, actionListElement);
			stateInfo[1]=actionHashtable;
			stateActions.put(state, stateInfo);
		}
		else if(((Hashtable<ActionListElement, ActionListElement>)stateInfo[1]).get(actionListElement)==null)
		{
			int ind=Collections.binarySearch(((List<ActionListElement>)stateInfo[0]), actionListElement);
			ind=-(ind+1);
			((List<ActionListElement>)stateInfo[0]).add(ind, actionListElement);
			((Hashtable<ActionListElement, ActionListElement>)stateInfo[1]).put(actionListElement, actionListElement);
		}
		else
		{
			int ind=Collections.binarySearch(((List<ActionListElement>)stateInfo[0]), ((Hashtable<ActionListElement, ActionListElement>)stateInfo[1]).get(actionListElement));
			if(ind<0) //TODO: figure out why sort the bsearch fails rarely
			{
				for(ind=0; ind<((List<ActionListElement>)stateInfo[0]).size(); ind++)
				{
					if(((List<ActionListElement>)stateInfo[0]).get(ind).equals(((Hashtable<ActionListElement, ActionListElement>)stateInfo[1]).get(actionListElement)))
					{
						break;
					}
				}
			}
			ActionListElement removed=((List<ActionListElement>)stateInfo[0]).remove(ind);
			removed.value=actionListElement.value;
			
			ind=Collections.binarySearch(((List<ActionListElement>)stateInfo[0]), removed);
			ind=-(ind+1);
			try
			{
				((List<ActionListElement>)stateInfo[0]).add(ind, removed);
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
			((Hashtable<ActionListElement, ActionListElement>)stateInfo[1]).put(removed, removed);
		}
	}

	@Override
	public Set<Entry<State, Object[]>> getStateActionValues() 
	{
		return stateActions.entrySet();
	}

	@Override
	public ActionListElement getActionListElement(State state, Action action) 
	{
		return ((Hashtable<ActionListElement, ActionListElement>)stateActions.get(state)[1]).get(new ActionListElement(action, 0.0));
	}
	
}


