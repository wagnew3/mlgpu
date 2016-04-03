package checkers;

import java.awt.Color;

import boardGame.BoardState;

public class CheckersState extends BoardState
{

	//true=red, false=blue
	public boolean color;
	/*
	 * 0=none
	 * -1=black
	 * -2=black king
	 * 1=red
	 * 2=red king
	 */
	public CheckersState(byte[][] boardState) 
	{
		super(boardState);
	}
	
	@Override
	public boolean isEndState() 
	{
		int numBlack=0;
		int numRed=0;
		for(int rowInd=0; rowInd<boardState.length; rowInd++)
		{
			for(int colInd=0; colInd<boardState[rowInd].length; colInd++)
			{
				if(boardState[rowInd][colInd]<0)
				{
					numBlack++;
				}
				else if(boardState[rowInd][colInd]>0)
				{
					numRed++;
				}
			}
		}
		if(numBlack==0 || numRed==0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	public int[][] getBoardImage()
	{
		int size=32;
		int[][] boardImage=new int[boardState.length*size][boardState[0].length*size];
		
		for(int rowInd=0; rowInd<boardState.length; rowInd++)
		{
			for(int colInd=0; colInd<boardState[rowInd].length; colInd++)
			{
				if((rowInd+colInd)%2==0)
				{
					setTile(boardImage, rowInd*size, colInd*size, size, 0x000000); //black
				}
				else
				{
					setTile(boardImage, rowInd*size, colInd*size, size, 0xFFFFFF); //white
				}
				
				switch(boardState[rowInd][colInd])
				{
					case -2: //black king
						setTile(boardImage, rowInd*size, colInd*size, size, 0x0000FF); //blue
						setTile(boardImage, rowInd*size+size/4, colInd*size+size/4, size/2, 0xFFFF00); //yellow
						break;
					case -1: //black
						setTile(boardImage, rowInd*size, colInd*size, size, 0x0000FF); //blue
						break;
					case 1: //red
						setTile(boardImage, rowInd*size, colInd*size, size, 0xFF0000); //red
						break;
					case 2: //red king
						setTile(boardImage, rowInd*size, colInd*size, size, 0xFF0000); //red
						setTile(boardImage, rowInd*size+size/4, colInd*size+size/4, size/2, 0xFFFF00); //yellow
						break;
				}
			}
		}
		
		return boardImage;
		
	}
	
	private void setTile(int[][] board, int x, int y, int sideLength, int value)
	{
		for(int rowInd=x; rowInd<x+sideLength; rowInd++)
		{
			for(int colInd=y; colInd<y+sideLength; colInd++)
			{
				board[rowInd][colInd]=value;
			}
		}
	}
	
	@Override
	public double[] getNNValue() 
	{
		double[] value=new double[boardState.length*boardState[0].length+4];
		for(int rowInd=0; rowInd<boardState.length; rowInd++)
		{
			for(int colInd=0; colInd<boardState[rowInd].length; colInd++)
			{
				value[(rowInd*boardState.length+colInd)/2]=boardState[rowInd][colInd];
				if(boardState[rowInd][colInd]==-2)
				{
					value[value.length-4]++;
				}
				else if(boardState[rowInd][colInd]==-1)
				{
					value[value.length-3]++;
				}
				else if(boardState[rowInd][colInd]==1)
				{
					value[value.length-2]++;
				}
				else if(boardState[rowInd][colInd]==2)
				{
					value[value.length-1]++;
				}
			}
		}
		
		for(int numberInfoInd=1; numberInfoInd<=4; numberInfoInd++)
		{
			value[value.length-numberInfoInd]/=(0.5*boardState.length*boardState[0].length);
		}
		
		return value;
	}
	
	public int numberPieces()
	{
		int numberPieces=0;
		for(int rowInd=0; rowInd<boardState.length; rowInd++)
		{
			for(int colInd=0; colInd<boardState[rowInd].length; colInd++)
			{
				if(boardState[rowInd][colInd]!=0)
				{
					numberPieces++;
				}
			}
		}
		return numberPieces;
	}
	
}
