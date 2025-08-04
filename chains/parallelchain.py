from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(
    model="gpt-3.5-turbo",
)

model2 = ChatOpenAI(
    model="gpt-4",
)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template= "Genrate a 3 question quiz from the following text: {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz questions into a single document: {notes} {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)
merged_chain = prompt3 | model1 | parser

chain = parallel_chain | merged_chain

text = """
### Introduction to Linear Regression
Linear regression is a type of supervised machine-learning algorithm that learns from labeled datasets and maps the data points with the most optimized linear functions for prediction on new datasets. It assumes a linear relationship between the input and output, meaning the output changes at a constant rate as the input changes. This relationship is represented by a straight line.

**Example**:  
We want to predict a student's exam score based on how many hours they studied.  
- **Independent variable (input)**: Hours studied (factor we control or observe).  
- **Dependent variable (output)**: Exam score (depends on hours studied).  

---

### Why Linear Regression is Important
1. **Simplicity and Interpretability**: Easy to understand and interpret, making it a starting point for learning machine learning.  
2. **Predictive Ability**: Helps predict future outcomes based on past data.  
3. **Basis for Other Models**: Forms the foundation for advanced algorithms like logistic regression or neural networks.  
4. **Efficiency**: Computationally efficient for problems with a linear relationship.  
5. **Widely Used**: Commonly applied in statistics and machine learning for regression tasks.  
6. **Analysis**: Provides insights into relationships between variables (e.g., how one variable influences another).  

---

### Best Fit Line in Linear Regression
The best-fit line is the straight line that minimizes the error between observed data points and predicted values.  

1. **Goal**: Minimize the error (difference) between observed data points and predicted values.  
2. **Equation**:  
   \( y = mx + b \)  
   - \( y \): Predicted value (dependent variable).  
   - \( x \): Input (independent variable).  
   - \( m \): Slope of the line (rate of change).  
   - \( b \): Intercept (value of \( y \) when \( x = 0 \)).  

---

### Least Squares Method
To find the best-fit line, the **Least Squares Method** minimizes the sum of squared residuals:  

- **Residual**: \( yᵢ - ŷᵢ \)  
- **Sum of Squared Errors (SSE)**: \( Σ(yᵢ - ŷᵢ)² \)  

This ensures the line best represents the data by minimizing the squared differences between predicted and actual values.  

---

### Interpretation of the Best-Fit Line
1. **Slope (m)**: Indicates how much the dependent variable (\( y \)) changes with each unit change in the independent variable (\( x \)).  
   - Example: If the slope is 5, \( y \) increases by 5 units for every 1-unit increase in \( x \).  
2. **Intercept (b)**: Represents the predicted value of \( y \) when \( x = 0 \).  

---

### Limitations of Linear Regression
1. **Assumes Linearity**: Assumes a linear relationship between variables.  
2. **Sensitivity to Outliers**: Outliers can significantly affect the slope and intercept, skewing the best-fit line.  

---

### Hypothesis Function in Linear Regression
The hypothesis function represents the relationship between input features and the target output.  

1. **Simple Linear Regression**:  
   \( h(x) = β₀ + β₁x \)  
   - \( h(x) \): Predicted value (\( y \)).  
   - \( x \): Independent variable.  
   - \( β₀ \): Intercept.  
   - \( β₁ \): Slope.  

2. **Multiple Linear Regression**:  
   \( h(x₁, x₂, ..., xₖ) = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ \)  
   - \( x₁, x₂, ..., xₖ \): Independent variables.  
   - \( β₀ \): Intercept.  
   - \( β₁, β₂, ..., βₖ \): Coefficients for each independent variable.  

---

### Assumptions of Linear Regression
1. **Linearity**: Relationship between inputs (\( X \)) and output (\( Y \)) is linear.  
2. **Independence of Errors**: Errors in predictions should not affect each other.  
3. **Constant Variance (Homoscedasticity)**: Errors should have equal spread across all input values.  
4. **Normality of Errors**: Errors should follow a normal distribution.  
5. **No Multicollinearity**: Input variables should not be too closely related.  
6. **No Autocorrelation**: Errors should not show repeating patterns (especially in time-based data).  
7. **Additivity**: Total effect on \( Y \) is the sum of effects from each \( X \).  

---

### Types of Linear Regression
1. **Simple Linear Regression**:  
   - One independent variable.  
   - Formula: \( ŷ = θ₀ + θ₁x \).  

2. **Multiple Linear Regression**:  
   - More than one independent variable.  
   - Formula: \( ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ \).  

---

### Use Cases of Multiple Linear Regression
1. **Real Estate Pricing**: Predict property prices based on location, size, etc.  
2. **Financial Forecasting**: Predict stock prices or economic indicators.  
3. **Agricultural Yield Prediction**: Estimate crop yields based on rainfall, temperature, etc.  
4. **E-commerce Sales Analysis**: Assess factors like product price, promotions, and trends.  

---

### Cost Function for Linear Regression
The **Mean Squared Error (MSE)** cost function calculates the average of squared errors between predicted (\( ŷᵢ \)) and actual (\( yᵢ \)) values:  

\[ J = \frac{1}{n} Σ (ŷᵢ - yᵢ)² \]  

---

### Gradient Descent Optimization
Gradient descent iteratively updates parameters (\( θ₁, θ₂ \)) to minimize the cost function. This ensures the MSE converges to the global minimum, providing the most accurate fit for the data.  
"""

result = chain.invoke({"text":text})
print(result)
