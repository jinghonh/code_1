\begin{algorithm}[h]
    \caption{Two-stage TrAdaBoost.R2 Algorithm}
    \label{alg:TwoStageTrAdaBoostR2}
    \renewcommand{\algorithmicrequire}{\textbf{Input:}}
    \renewcommand{\algorithmicensure}{\textbf{Output:}}
    \begin{algorithmic}[1]
      \Require two domain data set: \( X_{s}=(\hat{X}_{s}^{'*} , \widehat{x}_s) \) (of size $n$) and \(X_{t}= (X_t^k, y) \) (of size $l$), the number of steps: \( S \), the maximum iterations: \( I \),  the number of folds \( F \) for cross validation, and a base learning algorithm: \( L \)
    %   \renewcommand{\algorithmicrequire}{\textbf{Automatically learned parameters:}}
    %   \Require Weights \( w_s \), \( w_t \), parameter \( \beta \), normalization factor \( Z_t \)
      \Ensure Estimated values \( \dot{y}_{ap} \)
    %   \renewcommand{\algorithmicrequire}{\textbf{Initialize:}}
      \State  Initialize: source domain weight vector: \( \mathbf{w}_{s}^1 = \frac{1}{n + l} \) and target domain weight vector: \( \mathbf{w}_{t}^1 = \frac{1}{n + l} \)
      
      \For{\( step = 1, \ldots, S \)}
        % \State Use labeled data \( (x_s^k, y_s) \) and \( (x_t^k, y_t') \) with weights \( w_{s}^t \) and \( w_{t}^t \)
        
        % \State Train using the base learner \( L \) with the AdaboostR2 algorithm, keeping source domain weights \( w_{s}^t \) fixed during training, and perform \( F \)-fold cross-validation to select the optimal model \( \text{model}_t \)

        
        % \State Record the error for each sample \( e_{s,i} \), \( e_{t,j} \), and the total error \( \text{error} \)
        \State Call $L$ with $X_{s}$ and $X_{t}$ and distribution $\mathbf{w}_{s}^{step}$, $\mathbf{w}_{t}^{step}$, and get a hypothesis $h_t:X\rightarrow \mathbb{R} $.

        \State Calculate the adjusted error $e_{s,i}^{step}$ for each source instance and the adjusted error $e_{t,j}^{step}$ for each target instance in Adaboost.R2.
        
        \State Update weights:
        \begin{itemize}
            \item For source domain weights: \( w_{s,i}^{{step}+1} = w_{s,i}^{step} \cdot \beta_{step}^{e_{s,i}} \), where \( 1 \le i \le n \)
            \item For target domain weights: \( w_{t,j}^{{step}+1} = w_{t,j}^{step} / Z_{step} \), where \( 1 \le j \le l \)
            \item where \( \beta_{step} \) is chosen such that the resulting weight of the target domain weights instances is \( \frac{l}{n+l} + \frac{t}{T-1}(1 - \frac{l}{n+l}) \), and \( Z_{step} \) is a normalization constant.
        \end{itemize}
         
      \EndFor
      
      \State Select the best model \( {model}_{best} \), where \( best = \arg \min_{step}({error}_{step}) \)
      
      \State Output the estimated values \( \dot {y}_{ap} = {model}_{best}(\widehat{x}_t^k) \)
    \end{algorithmic}
  \end{algorithm}