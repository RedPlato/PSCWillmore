import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from scipy import stats
from scipy.interpolate import interp1d

#Afficher sphere, méridien et parallèle sur matplotlib 
def draw_normal_sphere():

    #nombre de méridien
    nb_long = 16
    #nombre de parallèle
    nb_lat = 8
    #paramètre du maillage
    N = 128

    #maillage
    theta = np.linspace(0, 2 * np.pi, N)
    phi = np.linspace(0, np.pi, N)
    x_sphere = np.outer(np.cos(theta), np.sin(phi))
    y_sphere = np.outer(np.sin(theta), np.sin(phi))
    z_sphere = np.outer(np.ones(np.size(theta)), np.cos(phi))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.5)

    #méridiens
    for i in range(nb_long + 1):
        phi = np.linspace(0, np.pi, N)
        theta = i * 2 * np.pi / nb_long
        x_circle = np.sin(phi)*np.sin(theta)
        y_circle = np.sin(phi)* np.cos(theta)
        z_circle = np.cos(phi)
        ax.plot(x_circle, y_circle, z_circle, color='black', alpha=1)

    #parallèles
    for i in range(1,nb_lat + 1):
        phi = i * np.pi / nb_lat
        theta = np.linspace(0, 2*np.pi, N)
        x_circle = np.sin(phi)*np.sin(theta)
        y_circle = np.sin(phi)* np.cos(theta)
        z_circle = np.cos(phi)
        ax.plot(x_circle, y_circle, z_circle, color='black', alpha=1)

    plt.show()

#Fonction F_(u,v,w)
#X,Y,Z : liste de coordonnées de S^2
#u,v,w : point de B^3
def f(X,Y,Z,u,v,w):
    a = 1-u**2 - v**2 - w**2
    B = (X-u)*(X-u) + (Y-v)*(Y-v) + (Z-w)*(Z-w)
    R = a * (X-u) / B - u
    S = a * (Y-v) / B - v
    T = a * (Z-w) / B - w
    return (R,S,T)

#Distance géodésique entre deux liste de points (X,Y,Z) et (R,S,T)
def d_geo(X,Y,Z,R,S,T):
    return np.arccos(X*R+Y*S+Z*T)

#Afficher image des méridiens et parallèles par f(.,.,.,u,v,w) sur mlab
def draw_sphere(u,v,w):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    nb_long = 32
    nb_lat = 16
    N = 256
    M = N*4

    dphi, dtheta = pi/N, pi/N
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    x = sin(phi)*cos(theta)
    y = cos(phi)
    z = sin(phi)*sin(theta)
    mlab.mesh(x, y, z, color=(1,0.55,0))

    for i in range(nb_long + 1):
        phi = np.linspace(0, np.pi, M)
        theta = i * 2 * np.pi / nb_long
        X = np.sin(phi)*np.sin(theta)
        Y = np.sin(phi)* np.cos(theta)
        Z = np.cos(phi)
        x,y,z = f(X,Y,Z,u,v,w)

        mlab.plot3d(x, y, z, tube_radius=0.003, color=(1,1,1))

    for i in range(1,nb_lat):
        phi = i * np.pi / nb_lat
        theta = np.linspace(0, 2*np.pi, M)
        X = np.sin(phi)*np.sin(theta)
        Y = np.sin(phi)* np.cos(theta)
        Z = np.cos(phi)
        x,y,z = f(X,Y,Z,u,v,w)

        mlab.plot3d(x, y, z, tube_radius=0.003, color=(1,1,1))

    mlab.show()

def curve(t):
    theta = 2*np.pi*t
    phi = np.pi/2 + np.sin(4*theta)/4

    X = np.sin(phi)*np.sin(theta)
    Y = np.sin(phi)*np.cos(theta)
    Z = np.cos(phi)
    return X,Y,Z

def tangent(X,Y,Z):

    N = len(X)
    T_x, T_y, T_z = N*(X-np.append(X[1:],X[0])),N*(Y-np.append(Y[1:],Y[0])),N*(Z-np.append(Z[1:],Z[0]))
    return T_x, T_y, T_z

def normal(X,Y,Z):
    T_x, T_y, T_z = tangent(X,Y,Z)
    V = np.stack((X,Y,Z), axis=-1)
    T = np.stack((T_x,T_y,T_z), axis=-1)
    N = np.cross(V,T)
    A,B,C = np.array([N[k][0] for k in range(len(N))]),np.array([N[k][1] for k in range(len(N))]),np.array([N[k][2] for k in range(len(N))])

    A,B,C= (A-(A*X +B*Y + C*Z)*A),(B-(A*X +B*Y + C*Z)*B),(C-(A*X +B*Y + C*Z)*C)
    norm= np.sqrt(A*A+B*B+C*C)
    A,B,C = A/norm,B/norm,C/norm
    return A,B,C
    
#Afficher image d'une courbe par f(.,.,.,u,v,w) sur mlab
def draw_curve(u,v,w):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    #Paramètre d'échantillonage
    N_sphere = 100
    N_gradient = 1000
    N_courbe = 2000

    #Calcul des points de la courbe avec échantillonage uniforme
    s = np.linspace(0, 1, N_gradient)
    X,Y,Z = curve(s)
    mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(1,0,0), line_width = 4.0)
    x,y,z = f(X,Y,Z,u,v,w)

    #Estimation des distance entre les images
    d = np.sqrt((x-u)*(x-u) + (y-v)*(y-v) + (z-w)*(z-w))
    d /= np.sum(d) 
    d = np.cumsum(d)

    #Echantillonage non uniforme afin que les images soient régulièrement espacées
    interp = interp1d(s,d,kind='linear')
    new_t = interp(np.linspace(0, 1, N_courbe))
    X,Y,Z = curve(new_t)
    x,y,z = f(X,Y,Z,u,v,w)
    mlab.plot3d(x, y, z, tube_radius=0.003, color=(0,1,0))
    
    #Affiichage de la normale
    a,b,c = normal(x,y,z)
    mlab.quiver3d(x,y,z,a,b,c,color = (0,0,0), line_width = 6.0)


    #Afficher les points v/|v| et -v/|v|
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    #Afficher la sphère
    dphi, dtheta = pi/N_sphere, pi/N_sphere
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    x = sin(phi)*cos(theta)
    y = cos(phi)
    z = sin(phi)*sin(theta)
    mlab.mesh(x, y, z, color=(1,0.55,0))

    mlab.show()

#Afficher la famille canonique de paramètre ((u,v,w), t) d'une courbe sur mlab
def draw_canonical(u,v,w,t):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    #Paramètre d'échantillonage
    N_sphere = 1000
    N_gradient = 100
    N_courbe = 100

    #Calcul des points de la courbe avec échantillonage uniforme
    s = np.linspace(0, 1, N_gradient)
    X,Y,Z = curve(s)
    #mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(1,0,0))
    x,y,z = f(X,Y,Z,u,v,w)

    #Estimation des distance entre les images
    d = np.sqrt((x-u)*(x-u) + (y-v)*(y-v) + (z-w)*(z-w))
    d /= np.sum(d) 
    d = np.cumsum(d) 

    #Echantillonage non uniforme afin que les images soient régulièrement espacées
    interp = interp1d(s,d,kind='linear')
    new_s = interp(np.linspace(0, 1, N_courbe))
    X,Y,Z = curve(new_s)
    x,y,z = f(X,Y,Z,u,v,w)
    mlab.plot3d(x, y, z, tube_radius=0.003, color=(0,1,0))

    #Afficher les points v/|v| et -v/|v|
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    #Maillage de la sphere
    dphi, dtheta = pi/N_sphere, 2*pi/N_sphere
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    X_sphere = sin(phi)*cos(theta)
    Y_sphere = cos(phi)
    Z_sphere = sin(phi)*sin(theta)

    n,m = np.shape(X_sphere)
    d = np.zeros(np.shape(X_sphere))

    #Calcul de la distance d_v d'un point du maillage à la courbe \Sigma_v
    for i in range(n):
        print("Progression : " + str(i+1) + "/" + str(n))
        for j in range(m):
            k0 = 0
            d[i,j] = np.arccos(X_sphere[i,j]*x[k0]+Y_sphere[i,j]*y[k0] + Z_sphere[i,j]*z[k0])
            for k in range(1,N_courbe):
                 c = np.arccos(X_sphere[i,j]*x[k]+Y_sphere[i,j]*y[k] + Z_sphere[i,j]*z[k])
                 if c < d[i,j]:
                     k0 = k
                     d[i,j] = c
            k1 = (k0 + 1) % N_courbe
            u = np.array([X_sphere[i,j],Y_sphere[i,j], Z_sphere[i,j]])
            v0 = np.array([x[k0],y[k0],z[k0]])
            v1 = np.array([x[k1],y[k1],z[k1]])
            w = np.cross(u,v1-v0)
            r = np.linalg.norm(w + v0)
            if r < 1:
                d[i,j] = - d[i,j]
    
    e = (d < t).astype(int)

    x = X_sphere*e
    y = Y_sphere*e
    z = Z_sphere*e

    mlab.mesh(x, y, z, color=(1,0.55,0))

    e = 1-e
    x = X_sphere*e
    y = Y_sphere*e
    z = Z_sphere*e

    mlab.mesh(x, y, z, color=(1,0.7,0))

    mlab.show()

def newdraw_canonical(u,v,w,t,eps):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    #Paramètre d'échantillonage
    N_sphere = 1000
    N_gradient = 100
    N_courbe = 100

    #Calcul des points de la courbe avec échantillonage uniforme
    s = np.linspace(0, 1, N_gradient)
    X,Y,Z = curve(s)
    #mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(1,0,0))
    x,y,z = f(X,Y,Z,u,v,w)

    #Estimation des distance entre les images
    d = np.sqrt((x-u)*(x-u) + (y-v)*(y-v) + (z-w)*(z-w))
    d /= np.sum(d) 
    d = np.cumsum(d) 

    #Echantillonage non uniforme afin que les images soient régulièrement espacées
    interp = interp1d(s,d,kind='linear')
    new_s = interp(np.linspace(0, 1, N_courbe))
    X,Y,Z = curve(new_s)
    x,y,z = f(X,Y,Z,u,v,w)
    mlab.plot3d(x, y, z, tube_radius=0.003, color=(0,1,0))
    n_x, n_y, n_z = normal(x,y,z)

    #Calcule de \Sigma_(v,t) par la géodésique
    c_x, c_y, c_z = cos(t)*x +sin(t)*n_x, cos(t)*y +sin(t)*n_y, cos(t)*z +sin(t)*n_z
    C_0 = np.stack((c_x, c_y, c_z), axis = -1)
    C = C_0
    for i in range(len(C)):
        for j in range(len(x)):
            d = d_geo(C_0[i][0],C_0[i][1],C_0[i][2],x[j],y[j],z[j])
            if t>0:
                if d< t-eps:
                    C[i] = [0,0,0]
           
            else:
                if -d> t+eps:
                    C[i] = [0,0,0]
            


    c_x,c_y,c_z = np.array([C[k][0] for k in range(len(C))]),np.array([C[k][1] for k in range(len(C))]),np.array([C[k][2] for k in range(len(C))])
    mlab.points3d(c_x, c_y, c_z, scale_factor = 0.01, color=(0,0,1))

    #Afficher les points v/|v| et -v/|v|
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))


    #Afficher la sphère
    dphi, dtheta = pi/N_sphere, pi/N_sphere
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    x = sin(phi)*cos(theta)
    y = cos(phi)
    z = sin(phi)*sin(theta)
    mlab.mesh(x, y, z, color=(1,0.55,0))





    mlab.show()
def draw_tubular(r):

    N_courbe = 1000
    N_sphere = 1000

    kappa = 7

    s = np.linspace(0, 1, N_courbe)
    t = np.linspace(-0.65, 3.8, N_courbe)

    X,Y,Z = curve(s)

    mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(0,1,0))

    X1 = np.roll(X,1)
    Y1 = np.roll(Y,1)
    Z1 = np.roll(Z,1)
    x0 = np.column_stack((X,Y,Z))
    x1 = np.column_stack((X1,Y1,Z1))
    N = np.cross(x1-x0,x0)
    N = N / np.linalg.norm(N, axis=0)

    cost = (1-r/kappa*np.sin(t))*np.cos(r*np.cos(t))
    sint = (1-r/kappa*np.sin(t))*np.sin(r*np.cos(t))
    
    x = np.outer(cost,X) + np.outer(sint,N[:,0])
    y = np.outer(cost,Y) + np.outer(sint,N[:,1])
    z = np.outer(cost,Z) + np.outer(sint,N[:,2])



    mlab.mesh(x, y, z, color=(1,1,0))
    #mlab.quiver3d(X,Y,Z,N[:,0], N[:,1], N[:,2])
    mlab.show()

def plot_T():

    e = 0.15
    theta_lim = 0.5
    N = 30
    fig, ax = plt.subplots()

    def function_phi(t):
        return 0 if t <= e else 1 if t >= 2*e else t/e - 1

    theta = np.linspace(-theta_lim, theta_lim, N)
    X = np.sin(theta)
    Y = np.cos(theta)

    ax.plot(X,Y)

    dr, dphi = e/5, pi/30
    [r,phi] = mgrid[0:3*e:dr,0:pi+dphi/2:dphi]

    s1 = r*np.sin(phi)
    s2 = r*np.cos(phi)

    X = (1-s1)*np.sin(s2)
    Y = (1-s1)*np.cos(s2)

    T_X = (1-s1*np.array(list(map(function_phi, r))))*np.sin(s2*np.array(list(map(function_phi, r))))
    T_Y = (1-s1*np.array(list(map(function_phi, r))))*np.cos(s2*np.array(list(map(function_phi, r))))

    color = np.sqrt(T_X**2 + T_Y**2) 

    ax.quiver(X,Y,T_X,T_Y, color, scale=30)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0.2, 1.3])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    fig.set_size_inches(10, 10, forward=True)

    plt.show()

def plot_Q():

    e = 0.1
    theta_lim = 0.3
    N = 30
    N_radial = 20
    N_angle = 15
    scale = 70
    width = 0.001

    fig, ax = plt.subplots()

    def function_phi(t):
        return 0 if t <= e else 1 if t >= 2*e else t/e - 1

    theta = np.linspace(-theta_lim, theta_lim, N)
    X = np.sin(theta)
    Y = np.cos(theta)

    ax.plot(X,Y, color="slateblue",zorder=0)

    theta = np.linspace(0,pi,N)
    X = (1-e*np.sin(theta))*np.sin(e*np.cos(theta))
    Y = (1-e*np.sin(theta))*np.cos(e*np.cos(theta))

    ax.plot(X,Y, color="slateblue",zorder=0)

    dr, dphi = e/N_radial, pi/N_angle
    [r,phi] = mgrid[0:e:dr,0:pi+dphi/2:dphi]

    s1 = r*np.sin(phi)
    s2 = r*np.cos(phi)

    X = (1-s1)*np.sin(s2)
    Y = (1-s1)*np.cos(s2)

    Q_X = -np.sqrt(1-s2*s2/(e*e))
    Q_Y = -s2/e

    color = np.arctan2(Q_X,Q_Y) 

    ax.quiver(X,Y,Q_X,Q_Y, color, scale=scale, width = width, zorder=1)

    s_ext = np.linspace(e, 3*e, 2*N_radial)

    X = np.sin(s_ext)
    Y = np.cos(s_ext)

    T_X = np.sin(s_ext*np.array(list(map(function_phi, s_ext))))
    T_Y = np.cos(s_ext*np.array(list(map(function_phi, s_ext))))

    color = np.arctan2(-T_X,-T_Y)
    color2 = np.arctan2(-T_X,T_Y)

    ax.quiver(X,Y,-T_X,-T_Y, color, scale=scale, width = width, zorder=1)
    ax.quiver(-X,Y,-T_X,T_Y, color2, scale=scale, width = width, zorder=1) 

    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([0.5, 1.3])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])


    fig.set_size_inches(100, 100, forward=True)

    plt.savefig('D:/antom/Desktop/PSC/Q.png')
    plt.show()

plot_Q()

