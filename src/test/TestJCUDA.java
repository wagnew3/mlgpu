package test;

public class testJCUDA 
{
	
	public static void main(String args[])
    {
        testSgemm(1000);
    }

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n)
    {
    	/*
        float alpha = 0.3f;
        float beta = 0.7f;
        int nn = n * n;

        System.out.println("Creating input data...");
        float h_A[] = createRandomFloatData(nn);
        float h_B[] = createRandomFloatData(nn);
        float h_C[] = createRandomFloatData(nn);
        float h_C_ref[] = h_C.clone();

        System.out.println("Performing Sgemm with Java...");
        sgemmJava(n, alpha, h_A, h_B, beta, h_C_ref);

        System.out.println("Performing Sgemm with JCublas...");
        sgemmJCublas(n, alpha, h_A, h_B, beta, h_C);

        boolean passed = isCorrectResult(h_C, h_C_ref);
        System.out.println("testSgemm "+(passed?"PASSED":"FAILED"));
        */
    }

}
