# Mathematical Model & Derivations — 9-State GNSS/IMU EKF

## 1. Overview and State Vector Definition

The 9-state navigation Extended Kalman Filter (EKF) estimates the spacecraft's inertial position, inertial velocity, and accelerometer bias vector in the Earth-Centered Inertial (ECI) frame:

$$\mathbf{x}(t) = \begin{bmatrix} \mathbf{r}(t) \\ \mathbf{v}(t) \\ \mathbf{b}_a(t) \end{bmatrix} \in \mathbb{R}^9$$

where:
- $\mathbf{r} = [r_x, r_y, r_z]^T$ is the position vector in ECI frame $[m]$
- $\mathbf{v} = [v_x, v_y, v_z]^T$ is the velocity vector in ECI frame $[m/s]$
- $\mathbf{b}_a = [b_{ax}, b_{ay}, b_{az}]^T$ is the accelerometer bias vector expressed in the body/IMU frame $[m/s^2]$

---

## 2. Physics & Sensor Measurement Models

### 2.1 Gravitational Acceleration vs. Accelerometer Specific Force
In free-fall orbit, a spacecraft's accelerometer **does not measure gravitational acceleration**. It measures only non-gravitational contact and specific forces $\mathbf{f}_a$ (such as thruster firing, aerodynamic drag, or solar radiation pressure).

The continuous-time accelerometer measurement model is:

$$\tilde{\mathbf{f}}_a = \mathbf{f}_a + \mathbf{b}_a + \mathbf{n}_a$$

where:
- $\tilde{\mathbf{f}}_a$ is the raw accelerometer reading $[m/s^2]$
- $\mathbf{f}_a$ is the true non-gravitational specific force $[m/s^2]$
- $\mathbf{b}_a$ is the accelerometer bias $[m/s^2]$
- $\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_a)$ is wide-band Gaussian accelerometer measurement noise

Rearranging for the true non-gravitational acceleration:

$$\mathbf{f}_a = \tilde{\mathbf{f}}_a - \mathbf{b}_a - \mathbf{n}_a$$

---

## 3. Continuous-Time State Equations

Combining Keplerian two-body central gravity $\mathbf{g}(\mathbf{r})$ with the IMU-inferred non-gravitational acceleration, the continuous-time dynamics equation is:

$$\dot{\mathbf{x}}(t) = \begin{bmatrix} \dot{\mathbf{r}} \\ \dot{\mathbf{v}} \\ \dot{\mathbf{b}}_a \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{g}(\mathbf{r}) + \mathbf{C}_{IB} (\tilde{\mathbf{f}}_a - \mathbf{b}_a - \mathbf{n}_a) \\ \mathbf{n}_{ba} \end{bmatrix}$$

where:
- $\mathbf{g}(\mathbf{r}) = -\frac{\mu}{\|\mathbf{r}\|^3} \mathbf{r}$ is two-body central Earth gravity $[m/s^2]$
- $\mathbf{C}_{IB}$ is the rotation matrix from spacecraft Body frame $B$ to ECI frame $I$ (assumed known/identity for orbit-only estimation)
- $\mathbf{n}_{ba} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_b)$ is the accelerometer bias random-walk process noise

---

## 4. Linearization and State Transition Matrix Jacobian $\mathbf{F}_{9\times9}$

The continuous-time system Jacobian matrix $\mathbf{F}(t) = \left. \frac{\partial \mathbf{f}(\mathbf{x}, \mathbf{u})}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}}$ is defined as:

$$\mathbf{F}_{9\times9} = \begin{bmatrix} 
\frac{\partial \dot{\mathbf{r}}}{\partial \mathbf{r}} & \frac{\partial \dot{\mathbf{r}}}{\partial \mathbf{v}} & \frac{\partial \dot{\mathbf{r}}}{\partial \mathbf{b}_a} \\[6pt]
\frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{r}} & \frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{v}} & \frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{b}_a} \\[6pt]
\frac{\partial \dot{\mathbf{b}}_a}{\partial \mathbf{r}} & \frac{\partial \dot{\mathbf{b}}_a}{\partial \mathbf{v}} & \frac{\partial \dot{\mathbf{b}}_a}{\partial \mathbf{b}_a}
\end{bmatrix} = \begin{bmatrix}
\mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} \\[6pt]
\mathbf{G}(\mathbf{r}) & \mathbf{0}_{3\times3} & -\mathbf{C}_{IB} \\[6pt]
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3}
\end{bmatrix}$$

### 4.1 Proof of Sign Convention for $\frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{b}_a}$
Since $\dot{\mathbf{v}} = \mathbf{g}(\mathbf{r}) + \mathbf{C}_{IB}(\tilde{\mathbf{f}}_a - \mathbf{b}_a)$, differentiating directly with respect to $\mathbf{b}_a$ yields:

$$\frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{b}_a} = -\mathbf{C}_{IB}$$

For identity orientation ($\mathbf{C}_{IB} = \mathbf{I}_{3\times3}$), this block is strictly $-\mathbf{I}_{3\times3}$.

### 4.2 Gravity Gradient Matrix $\mathbf{G}(\mathbf{r})$
The position derivative of central gravity $\mathbf{g}(\mathbf{r}) = -\mu r^{-3} \mathbf{r}$ is:

$$\mathbf{G}(\mathbf{r}) = \frac{\partial \mathbf{g}}{\partial \mathbf{r}} = \frac{\mu}{\|\mathbf{r}\|^3} \left( 3 \hat{\mathbf{r}} \hat{\mathbf{r}}^T - \mathbf{I}_{3\times3} \right)$$

---

## 5. Discrete-Time State Transition Matrix $\mathbf{\Phi}$ and Process Noise $\mathbf{Q}_d$

For propagation interval $\Delta t$, the discrete state transition matrix $\mathbf{\Phi}_k$ is computed via matrix exponential or truncated Taylor series:

$$\mathbf{\Phi}_k = \exp(\mathbf{F} \Delta t) \approx \mathbf{I}_{9\times9} + \mathbf{F} \Delta t + \frac{1}{2} \mathbf{F}^2 \Delta t^2$$

Explicit block evaluation:

$$\mathbf{\Phi}_k \approx \begin{bmatrix}
\mathbf{I} + \frac{1}{2} \mathbf{G} \Delta t^2 & \mathbf{I} \Delta t & -\frac{1}{2} \mathbf{C}_{IB} \Delta t^2 \\[6pt]
\mathbf{G} \Delta t & \mathbf{I} + \frac{1}{2} \mathbf{G} \Delta t^2 & -\mathbf{C}_{IB} \Delta t \\[6pt]
\mathbf{0} & \mathbf{0} & \mathbf{I}
\end{bmatrix}$$

### 5.1 Continuous-to-Discrete Process Noise Mapping
The process noise matrix $\mathbf{Q}_c = \operatorname{diag}(\mathbf{Q}_a, \mathbf{Q}_b)$ is integrated over interval $\Delta t$:

$$\mathbf{Q}_d = \int_0^{\Delta t} \mathbf{\Phi}(\tau) \mathbf{G}_{proc} \mathbf{Q}_c \mathbf{G}_{proc}^T \mathbf{\Phi}^T(\tau) d\tau$$

Using 2nd-order Van Loan approximation:

$$\mathbf{Q}_d \approx \mathbf{W} \Delta t + \frac{1}{2} (\mathbf{F} \mathbf{W} + \mathbf{W} \mathbf{F}^T) \Delta t^2$$

where $\mathbf{W} = \mathbf{G}_{proc} \mathbf{Q}_c \mathbf{G}_{proc}^T = \operatorname{diag}(\mathbf{0}_{3\times3}, \, \mathbf{Q}_a, \, \mathbf{Q}_b)$.

---

## 6. GNSS Measurement Model

The GNSS receiver provides solution-level position and velocity measurements:

$$\mathbf{z}_k = \begin{bmatrix} \mathbf{r}_{gnss} \\ \mathbf{v}_{gnss} \end{bmatrix} = \mathbf{H}_{6\times9} \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

The measurement matrix is:

$$\mathbf{H}_{6\times9} = \begin{bmatrix} \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\ \mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} \end{bmatrix} = \begin{bmatrix} \mathbf{I}_{6\times6} & \mathbf{0}_{6\times3} \end{bmatrix}$$

---

## 7. Initial Covariance $\mathbf{P}_0$

$$\mathbf{P}_0 = \begin{bmatrix}
\sigma_{r0}^2 \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \sigma_{v0}^2 \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \sigma_{ba0}^2 \mathbf{I}_{3\times3}
\end{bmatrix}$$
