package layer;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import activationFunctions.ActivationFunction;

public class ConvolutionLayer extends Layer
{
	
	protected int[] inputsDimensions;
	protected int stride;
	protected int boxSize;
	protected int numberRows;
	protected int numberColumns;
	protected BlockRealMatrix fullWeights;

	public ConvolutionLayer(ActivationFunction activationFunction, int[] inputsDimensions, int stride, int boxSize) 
	{
		super(activationFunction);
		this.inputsDimensions=inputsDimensions;
		this.stride=stride;
		if((inputsDimensions[0]-boxSize)%stride!=0)
		{
			System.out.println("inputsDimesions[0]%stride!=boxSize-1");
		}
		if((inputsDimensions[1]-boxSize)%stride!=0)
		{
			System.out.println("inputsDimesions[1]%stride!=boxSize-1");
		}
		this.boxSize=boxSize;
		
		weights=new BlockRealMatrix(1, boxSize*boxSize);
		NormalDistribution nInvGaussian=new NormalDistribution(0.0, 1.0/Math.sqrt(boxSize));
		for(int rowIndex=0; rowIndex<weights.getRowDimension(); rowIndex++)
		{
			for(int colIndex=0; colIndex<weights.getColumnDimension(); colIndex++)
			{
				weights.setEntry(rowIndex, colIndex, nInvGaussian.sample());
			}
		}
		
		biases=new ArrayRealVector(1);
		NormalDistribution zeroGuassian=new NormalDistribution(0.0, 1.0);
		for(int rowIndex=0; rowIndex<biases.getDimension(); rowIndex++)
		{
			biases.setEntry(rowIndex, zeroGuassian.sample());
		}
		
		numberRows=(inputsDimensions[0]-Math.max(boxSize-stride, 0))/stride;
		numberColumns=(inputsDimensions[1]-Math.max(boxSize-stride, 0))/stride;
		fullWeights=getWeights();
	}

	@Override
	public ArrayRealVector getOutput(ArrayRealVector input) 
	{
		ArrayRealVector z=getActivations(input);
		return activationFunction.applyActivationFunction(z);
	}
	
	protected ArrayRealVector matrixToVector(BlockRealMatrix matrix)
	{
		ArrayRealVector vector=new ArrayRealVector(matrix.getRowDimension()*matrix.getColumnDimension());
		for(int rowInd=0; rowInd<matrix.getRowDimension(); rowInd++)
		{
			for(int colInd=0; colInd<matrix.getColumnDimension(); colInd++)
			{
				vector.setEntry(rowInd*matrix.getColumnDimension()+colInd, matrix.getEntry(rowInd,  colInd));
			}
		}
		return vector;
	}
	
	protected double vectorMatrixDotProduct(BlockRealMatrix vectorMat, BlockRealMatrix matrix)
	{
		double dotProduct=0.0;
		RealVector vector=vectorMat.getRowVector(0);
		for(int rowInd=0; rowInd<matrix.getRowDimension(); rowInd++)
		{
			dotProduct+=vector.getSubVector(rowInd*matrix.getRowDimension(), matrix.getRowDimension()).dotProduct(matrix.getRowVector(rowInd));
		}
		return dotProduct;
	}

	@Override
	public int getOutputSize()
	{
		return numberRows*numberColumns;
	}

	@Override
	public ArrayRealVector getActivations(ArrayRealVector input)
	{
		BlockRealMatrix matrixInputs=new BlockRealMatrix(inputsDimensions[0], inputsDimensions[1]);
		ArrayRealVector z=new ArrayRealVector(getOutputSize());
		for(int rowInd=0; rowInd<matrixInputs.getRowDimension(); rowInd++)
		{
			matrixInputs.setRow(rowInd, input.getSubVector(rowInd*inputsDimensions[1], inputsDimensions[1]).toArray());
		}
		
		int neuronIndex=0;
		for(int rowInd=0; rowInd<numberRows; rowInd+=stride)
		{
			for(int colInd=0; colInd<numberColumns; colInd+=stride)
			{
				//z.setEntry(neuronIndex, weights.operate(matrixToVector(matrixInputs.getSubMatrix(rowInd, rowInd+boxSize-1, colInd, colInd+boxSize-1))).getEntry(0));
				z.setEntry(neuronIndex, vectorMatrixDotProduct(weights, matrixInputs.getSubMatrix(rowInd, rowInd+boxSize-1, colInd, colInd+boxSize-1)));
				neuronIndex++;
			}
		}
		z=(ArrayRealVector)z.mapAdd(biases.getEntry(0));
		return z;
	}

	@Override
	public ArrayRealVector getOutputDerivatives(ArrayRealVector input) 
	{
		ArrayRealVector z=getActivations(input);
		return activationFunction.getDerivatives(z);
	}

	@Override
	public ArrayRealVector getDeltas(BlockRealMatrix nextLayerWeights, ArrayRealVector nextLayerDeltas, ArrayRealVector activationDerivative) 
	{
		return (ArrayRealVector)nextLayerWeights.transpose().operate(nextLayerDeltas).ebeMultiply(activationDerivative);
	}
	
	@Override
	public RealMatrix getWeightPDs(ArrayRealVector previousLayerOutputs, ArrayRealVector deltas) 
	{
		double[][] weights=fullWeights.getData();
		double[][] weightsPD=deltas.outerProduct(previousLayerOutputs).getData();
		
		for(int rowInd=0; rowInd<weights.length; rowInd++)
		{
			for(int colInd=0; colInd<weights[rowInd].length; colInd++)
			{
				if(weights[rowInd][colInd]==0.0)
				{
					weightsPD[rowInd][colInd]=0.0;
				}
			}
		}
		
		return new BlockRealMatrix(weightsPD);
	}
	
	public BlockRealMatrix getWeights()
	{
		BlockRealMatrix fullWeights=new BlockRealMatrix(getOutputSize(), inputsDimensions[0]*inputsDimensions[1]);
		
		for(int rowInd=0; rowInd<numberRows; rowInd+=stride)
		{
			for(int colInd=0; colInd<numberColumns; colInd+=stride) //for each neuron
			{
				
				for(int boxRowInd=0; boxRowInd<boxSize; boxRowInd++) //for each input in neuron's box
				{
					for(int boxColInd=0; boxColInd<boxSize; boxColInd++)
					{
						int neuronIndex=(rowInd/stride)*numberColumns+(colInd/stride);
						try
						{
						fullWeights.setEntry(neuronIndex, (rowInd+boxRowInd)*numberColumns+(colInd+boxColInd), weights.getEntry(0, boxRowInd*boxSize+boxColInd));
						}
						catch(Exception e)
						{
							e.printStackTrace();
						}
					}
				}
				
			}
		}
		
		return fullWeights;
	}
	
	public ArrayRealVector getBiases()
	{
		ArrayRealVector fullBiases=new ArrayRealVector(getOutputSize(), biases.getEntry(0));
		return fullBiases;
	}
	
	//weightpds=deltas*activations
	// sum and average columns to get weight pds with single delta (single bias derivative)
	//result is inputs, add to appropriate weight variable for weights pd 
	public void updateWeights(BlockRealMatrix weightPDs, double learningRate)
	{
		if(Double.isNaN(weightPDs.getEntry(0, 0)))
		{
			int i=0;
		}
		ArrayRealVector weightPDsVector=new ArrayRealVector(weightPDs.getColumnDimension());
		ArrayRealVector columnAdd=new ArrayRealVector(weightPDs.getRowDimension(), 1.0);
		for(int columnInd=0; columnInd<weightPDsVector.getDimension(); columnInd++)
		{
			ArrayRealVector column=(ArrayRealVector)weightPDs.getColumnVector(columnInd);
			weightPDsVector.setEntry(columnInd, columnAdd.dotProduct(column));
		}
		
		BlockRealMatrix condensedWeightsPDVector=new BlockRealMatrix(1, weights.getColumnDimension());
		for(int columnInd=0; columnInd<weightPDsVector.getDimension(); columnInd++)
		{
			int row=columnInd/inputsDimensions[0];
			int column=columnInd%inputsDimensions[0];
			
			condensedWeightsPDVector.addToEntry(0, (row%boxSize)*boxSize+column%boxSize, weightPDsVector.getEntry(columnInd));
		}
		
		weights=weights.subtract(condensedWeightsPDVector.scalarMultiply(learningRate));
		fullWeights=getWeights();
	}
	
	public void updateBiases(ArrayRealVector biasesPDs, double learningRate)
	{
		if(Double.isNaN(biasesPDs.getEntry(0)))
		{
			int i=0;
		}
		double biasPD=0.0;
		for(int biasInd=0; biasInd<biasesPDs.getDimension(); biasInd++)
		{
			biasPD+=biasesPDs.getEntry(biasInd);
		}
		biasPD/=biasesPDs.getDimension();
		biases=(ArrayRealVector)biases.mapSubtract(biasPD*learningRate);
	}

	@Override
	public Layer clone() 
	{
		// TODO Auto-generated method stub
		return null;
	}

}
