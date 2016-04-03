package offPolicyMonteCarlo;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class Test 
{
	
	private static final int squareSize=5;
	
	public static void main(String[] test)
	{
		//testOPMCLine();
		testOPMCLine2();
	}
	
	public static void testOPMCLine()
	{
		int[][] track=new int[][]{new int[]{1},
								  new int[]{0},
								  new int[]{0},
								  new int[]{0},
								  new int[]{0},
								  new int[]{0}};
		CarRaceEnvironment carRaceEnvironment=new CarRaceEnvironment(track, 5, 5);
		OffPolicyMonteCarlo opmc=new OffPolicyMonteCarlo(new State[]{new CarRaceState(5, 0, 0, 0, null, null)},
				new CarRacePolicy(track.length, track[0].length, 11, 11, 0.0, 0.25),
				new CarRacePolicy(track.length, track[0].length, 11, 11, 0.0, 0.0),
				carRaceEnvironment,
				new CarRaceRewardFunction(carRaceEnvironment),
				track.length*track[0].length*11*11,
				9,
				0.9);
		opmc.learn();
	}
	
	public static void testOPMCLine2()
	{
		int[][] track=new int[][]{new int[]{1, 1, 1},
								  new int[]{0, 0, 0},
								  new int[]{0, 0, 0},
								  new int[]{0,-1, -1},
								  new int[]{0,-1, -1},
								  new int[]{0, 0, 0}};
		CarRaceEnvironment carRaceEnvironment=new CarRaceEnvironment(track, 5, 5);
		OffPolicyMonteCarlo opmc=new OffPolicyMonteCarlo(new State[]{new CarRaceState(5, 1, 0, 0, null, null)},
				new CarRacePolicy(track.length, track[0].length, 11, 11, 0.0, 0.25),
				new CarRacePolicy(track.length, track[0].length, 11, 11, 0.0, 0.0),
				carRaceEnvironment,
				new CarRaceRewardFunction(carRaceEnvironment),
				track.length*track[0].length*11*11,
				9,
				0.9);
		opmc.learn();
	}
	
	
	private void displayNextStateOnClick(CarRaceState endState, CarRaceEnvironment environment)
	{
		List<CarRaceState> states=new ArrayList<>();
		states.add(endState);
		while(states.get(0).previousState!=null)
		{
			states.add(0, states.get(0).previousState);
		}
	}
	
	private BufferedImage drawTrack(CarRaceState state, CarRaceEnvironment environment)
	{
		BufferedImage trackImage=new BufferedImage(environment.track.length*squareSize, environment.track[0].length*squareSize, BufferedImage.TYPE_INT_RGB);
		for(int trackX=0; trackX<environment.track.length; trackX++)
		{
			for(int trackY=0; trackY<environment.track[0].length; trackY++)
			{
				if(environment.track[trackX][trackY]==1) //1=finish
				{
					drawSquare(trackImage, trackX*squareSize, trackY*squareSize, squareSize, (byte)9);
				}
				else if(environment.track[trackX][trackY]==0) //0=track
				{
					drawSquare(trackImage, trackX*squareSize, trackY*squareSize, squareSize, (byte)0);
				}
				else if(environment.track[trackX][trackY]==-1)//-1=offtrack
				{
					drawSquare(trackImage, trackX*squareSize, trackY*squareSize, squareSize, (byte)12);
				}
			}
		}	
		do
		{
			drawSquare(trackImage, state.x*squareSize+1, state.y*squareSize+1, squareSize-2, (byte)10);
			state=state.previousState;
		}
		while(state!=null);
		return trackImage;
	}
	
	private void drawSquare(BufferedImage image, int x, int y, int sideLength, byte color)
	{
		for(int xPos=x; xPos<x+sideLength; xPos++)
		{
			for(int yPos=y; yPos<y+sideLength; yPos++)
			{
				image.setRGB(xPos, yPos, color);
			}
		}
	}

}
