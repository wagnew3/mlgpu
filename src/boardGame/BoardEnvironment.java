package boardGame;

import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;

public abstract class BoardEnvironment extends LimitedActionsEnvironment 
{

	public int turn;
	int numberPlayers;
	protected BoardState currentState;
	JFrame frame=null;
	
	public BoardEnvironment(int numberPlayers)
	{
		this.numberPlayers=numberPlayers;
		turn=0;
	}
	
	@Override
	public void setStartState(State startState) 
	{
		currentState=(BoardState)startState;
		turn=0;
	}

	@Override
	public State getCurrentState() 
	{
		return currentState;
	}

	@Override
	public double takeAction(Action action, int actor) 
	{
		turn=(actor+1)%numberPlayers;
		double reward=takePlayerAction(action);
		return reward;
	}
	
	public abstract double takePlayerAction(Action action);
	
	public void displayBoard()
	{
		if(frame==null)
		{
			frame=new JFrame();
		}
		int[][] imageData=currentState.getBoardImage();
		BufferedImage boardImage=new BufferedImage(imageData.length, imageData[0].length, BufferedImage.TYPE_INT_RGB);
		for(int rowInd=0; rowInd<imageData.length; rowInd++)
		{
			for(int colInd=0; colInd<imageData[0].length; colInd++)
			{
				boardImage.setRGB(rowInd, colInd, imageData[rowInd][colInd]);
			}
		}
		frame.getContentPane().removeAll();
		frame.getContentPane().setLayout(new FlowLayout());
		frame.getContentPane().add(new JLabel(new ImageIcon(boardImage)));
		frame.pack();
		frame.setVisible(true);
	}
	
	public abstract void humanMove(int playerNumber);

}
