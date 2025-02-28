# **LQR Homework**

## **1. Problem Formulation**
Consider a discrete-time linear system with **state** \( x \) and **control** \( u \), evolving according to:

\[    x[t+1] = A x[t] + B u[t]
\]

where:
- \( x[t] \in \mathbb{R}^n \) is the state vector.
- \( u[t] \in \mathbb{R}^m \) is the control input.
- \( A \in \mathbb{R}^{n \times n} \) is the state transition matrix.
- \( B \in \mathbb{R}^{n \times m} \) is the control matrix.

We seek to determine the optimal control sequence \( u[t] \) that minimizes the **quadratic cost function** over a horizon \( T \):

\[    J = \sum_{t=0}^{T} \left( x[t]^T Q x[t] + u[t]^T R u[t] \right)
\]

where:
- \( Q \in \mathbb{R}^{n \times n} \) is the **state cost matrix** (positive semi-definite).
- \( R \in \mathbb{R}^{m \times m} \) is the **control cost matrix** (positive definite).

The **cost-to-go function** at time \( t \), assuming optimal actions are taken thereafter, is a **quadratic function** of the state:

\[    V_t(x) = x[t]^T P[t] x[t].
\]

Our goal is to **derive the recursive equations** to compute \( P[t] \) and obtain the **optimal control law** \( u[t] \).

---

## **2. Bellman Equation for the Optimal Cost-to-Go Function**
The **Bellman equation** states that the optimal cost-to-go at time \( t \) satisfies:

\[    V_t(x) = \min_{u[t]} \left( x[t]^T Q x[t] + u[t]^T R u[t] + V_{t+1}(x[t+1]) 
ight).
\]

Substituting the system dynamics:

\[    x[t+1] = A x[t] + B u[t],
\]

we get:

\[    V_t(x) = \min_{u[t]} \Big( x[t]^T Q x[t] + u[t]^T R u[t] + (A x[t] + B u[t])^T P[t+1] (A x[t] + B u[t]) \Big).
\]

Expanding the quadratic terms:

\[    V_t(x) = \min_{u[t]} \Big( x[t]^T Q x[t] + u[t]^T R u[t] + x[t]^T A^T P[t+1] A x[t] 
\]
\[    + u[t]^T B^T P[t+1] B u[t] + 2 x[t]^T A^T P[t+1] B u[t] \Big).
\]

---

## **3. Computing the Optimal Control \( u^* \)**
To find the optimal \( u[t] \), differentiate \( V_t(x) \) with respect to \( u[t] \) and set it to zero:

\[    \frac{d}{du} \left( u^T R u + 2 x^T A^T P[t+1] B u + u^T B^T P[t+1] B u \right) = 0.
\]

\[    (R + B^T P[t+1] B) u + B^T P[t+1] A x = 0.
\]

Solving for \( u^* \):

\[    u^* = - (R + B^T P[t+1] B)^{-1} B^T P[t+1] A x.
\]

Thus, the **optimal control law** is:

\[    u[t] = -K[t] x[t],
\]

where the **LQR feedback gain matrix** is:

\[    K[t] = (R + B^T P[t+1] B)^{-1} B^T P[t+1] A.
\]

---

## **4. Computing \( P[t] \) Using the Riccati Recursion**
Substituting \( u^* \) into the value function:

\[    V_t(x) = x^T Q x + (-K x)^T R (-K x) + x^T A^T P[t+1] A x
\]
\[    + 2 x^T A^T P[t+1] B (-K x) + (-K x)^T B^T P[t+1] B (-K x).
\]

Expanding:

\[    V_t(x) = x^T (Q + A^T P[t+1] A - A^T P[t+1] B K - K^T B^T P[t+1] A + K^T B^T P[t+1] B K + K^T R K) x.
\]

Since we showed that \( -K^T B^T P[t+1] A + K^T B^T P[t+1] B K + K^T R K = 0 \), the equation simplifies to:

\[    P[t] = Q + A^T P[t+1] A - A^T P[t+1] B (R + B^T P[t+1] B)^{-1} B^T P[t+1] A.
\]

This is the **discrete-time Riccati equation**, which we solve **backward in time** starting from the terminal condition:

\[    P[T] = Q.
\]

---

## **5. Summary of LQR Solution**
1. **Compute \( P[t] \) recursively** using the Riccati equation.
2. **Compute the optimal control gain matrix \( K[t] \).**
3. **Apply the optimal control law \( u[t] = -K[t] x[t] \).**

This ensures the system is driven optimally toward stability while minimizing the cost function.

---

## **6. Key Takeaways**
- The **cost-to-go function** is quadratic: \( V_t(x) = x^T P[t] x \).
- The **optimal control law** is **linear**: \( u[t] = -K[t] x[t] \).
- The **Riccati equation** computes \( P[t] \) recursively.
- LQR finds the **best trade-off between control effort and state deviation**.

