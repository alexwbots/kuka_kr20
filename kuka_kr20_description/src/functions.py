import numpy as np
from copy import copy
from pyquaternion import Quaternion

cos=np.cos; sin=np.sin; pi=np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
    T = np.array([[cos(theta), -cos(alpha)*sin(theta), sin(alpha)*sin(theta) , a*cos(theta)],
                  [sin(theta), cos(alpha)*cos(theta) , -sin(alpha)*cos(theta), a*sin(theta)],
                  [0.0       , sin(alpha)            , cos(alpha)            , d           ],
                  [0.0       , 0.0                   , 0.0                   , 1.0         ]])
    return T
    
    
def fkine_kr20(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """
    # Longitudes (en metros)

    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = dh(0.52, q[0], 0.16, pi/2)
    T2 = dh(0, -q[1]+pi/2, 0.78, 0)
    T3 = dh(0, q[2], 0.15, pi/2)
    T4 = dh(0.86, q[3], 0, pi/2)
    T5 = dh(0, -q[4], 0, -pi/2)
    T6 = dh(0.153, q[5], 0, 0)
    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
    return T


def jacobian_kr20(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
    """
    # Crear una matriz 3x6
    J = np.zeros((3,6))
    # Utilizar la funcion que calcula la cinematica directa, para encontrar x,y,z usando q
    T = fkine_kr20(q)
    # Iteracion para la derivada de cada columna
    for i in range(6):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i]+delta
	# Transformacion homogenea luego del incremento (q+delta)
        T_inc = fkine_kr20(dq)
	# Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0:3,i] = (T_inc[0:3,3]-T[0:3,3])/delta
    return J


def ikine_kr20(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo de newton
    """
    epsilon  = 0.005
    max_iter = 10000
    delta    = 0.000001

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_kr20(q,delta)
        T = fkine_kr20(q)
        f = T[0:3,3]
        e = xdes-f
        q = q + np.dot(np.linalg.pinv(J),e)
        # Condicion de termino
        if(np.linalg.norm(e) < epsilon):
            break
    
    return q


def jacobian_pose(q, delta=0.000001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    J = np.zeros((7,6))
    # Transformacion homogenea inicial (usando q)
    T0 = fkine_kr20(q)
    pose_inicial=TF2xyzquat(T0)
    
    for i in range(6):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i]= dq[i]+delta
        # Transformacion homogenea (TH) luego del incremento (q+delta)
        T_inc= fkine_kr20(dq)
        # Pose of the robot from TH
        pose=TF2xyzquat(T_inc)
        # Aproximacion del Jacobiano de la tarea usando diferencias finitas
        J[0:7,i]=(pose-pose_inicial)/delta
    
    return J


def rot2quat(RM):
  
    quat = Quaternion(matrix=RM)  
    return quat


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = Quaternion(matrix=T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat.w, quat.x, quat.y, quat.z]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R
