# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import NuSVR
from sklearn.metrics import f1_score
from scipy import misc



def create_cv_set(X,y,k):
	n=X.shape[0]
	R=np.random.permutation(n) # Vecteur de permutation aléatoire
	step=int(np.floor(1.0*n/k)) # Nombre d'éléments par sous-enssemble

	Xcv=[]
	Ycv=[]

	for i in range(k):
		if(i<k-1):
			Xcv.append(X[R[i*step:(i+1)*step],:])
			Ycv.append(y[R[i*step:(i+1)*step]])
		else: #On remplit le dernier sous-enssemble avec tout les éléments restants
			Xcv.append(X[R[i*step:n],:])
			Ycv.append(y[R[i*step:n]])
	
	return Xcv, Ycv

def create_train_set(Xcv,Ycv,n):
	k=len(Xcv)
	
	#l'enssemble de test correspont au n-ième élément
	X_test=Xcv[n]
	y_test=Ycv[n]
	
	#l'enssemble d'entrainement correspond à la concaténation de tous les autres
	X_train=np.zeros((0,0))
	y_train=np.zeros(0)

	for i in range(k):
		if(i!=n):
			if(X_train.shape[0]==0 and X_train.shape[1]==0 ): #initialisation de l'enssemble d'entrainement
				X_train=Xcv[i]
				y_train=Ycv[i]
			else: #concaténation
				X_train=np.concatenate((X_train,Xcv[i]),axis=0)
				y_train=np.concatenate((y_train,Ycv[i]),axis=0)
	
	return X_train, y_train, X_test, y_test

def cv_SVM_linear(X,y,K,C_test):
	#Vecteur contenant les précision moyennes pour chaque valeur de c dans C_test
	Accuracy=np.zeros((len(C_test),1))

	Xcv, Ycv=create_cv_set(X,y,K)

	k1=0;
	for c in C_test:
		#Précision moyenne pour la valeur de c courante
		current_acc=0.0;
		#on définit le svm et son paramètre
		svc = SVC(kernel="linear", C=c)

		for n in range(K):
			
			X_train, y_train, X_test, y_test=create_train_set(Xcv,Ycv,n)
			
			#On entraine le SVM
			svc.fit(X_train, y_train)
			#On recupère le score de bonne classification
			y_pred = svc.predict(X_test)
			res_tmp= f1_score(y_test, y_pred)
			#On met à jour la précision moyenne pour la valeur de c courante
			current_acc=current_acc+res_tmp/(1.0*K)
		
		#On affecte la précision moyenne pour la valeur de c courante au vecteur contenant les précision moyennes pour chaque valeur de c dans C_test
		Accuracy[k1,0]=current_acc
				
		k1=k1+1

	# On récupère la meilleure valeur
	acc_test=0;
	C_opt=0;
	for k1 in range(Accuracy.shape[0]):
		for k2 in range(Accuracy.shape[1]):
			if(Accuracy[k1,k2]>acc_test):
				acc_test=Accuracy[k1,k2]
				C_opt=C_test[k1]
	
	
#	print("SVM lineaire, Parametre optimal: C=",C_opt)
	
	return C_opt

def cv_SVM_rbf(X,y,K,C_test,gamma_test):
	Accuracy=np.zeros((len(C_test),len(gamma_test)))

	Xcv, Ycv=create_cv_set(X,y,K)

	k1=0;
	for c in C_test:
		k2=0
		for g in gamma_test:			

			current_acc=0.0;

			for n in range(K):
				
				svc = SVC(kernel="rbf", C=c,gamma=g)
				
				X_train, y_train, X_test, y_test=create_train_set(Xcv,Ycv,n)
				
				#On entraine le SVM
				svc.fit(X_train, y_train)

				res_tmp= svc.score(X_test, y_test)
				
				current_acc=current_acc+res_tmp/(1.0*K)
			
			Accuracy[k1,k2]=current_acc
			
			k2=k2+1		
		k1=k1+1
	
		
	acc_test=0;
	C_opt=0;
	gamma_opt=0;
	for k1 in range(Accuracy.shape[0]):
		for k2 in range(Accuracy.shape[1]):
			if(Accuracy[k1,k2]>acc_test):
				acc_test=Accuracy[k1,k2]
				C_opt=C_test[k1]
				gamma_opt=gamma_test[k2]

	print("SVM rbf, Parametres optimaux: C=",C_opt," gamma=",gamma_opt)
		
	return C_opt, gamma_opt

def cv_nu_SVR(X,y,K,C_test,nu_test):
	Accuracy=np.zeros((len(C_test),len(nu_test)))

	Xcv, Ycv=create_cv_set(X,y,K)

	k1=0;
	for c in C_test:
		k2=0
		for nu in nu_test:			

			current_acc=0.0;

			for n in range(K):
				
				svc = NuSVR(C=c, nu=nu)
				
				X_train, y_train, X_test, y_test=create_train_set(Xcv,Ycv,n)
				
				#On entraine le SVM
				svc.fit(X_train, y_train)

				res_tmp= svc.score(X_test, y_test)
				
				current_acc=current_acc+res_tmp/(1.0*K)
			
			Accuracy[k1,k2]=current_acc
			
			k2=k2+1		
		k1=k1+1
	
		
	acc_test=0;
	C_opt=0;
	nu_opt=0;
	for k1 in range(Accuracy.shape[0]):
		for k2 in range(Accuracy.shape[1]):
			if(Accuracy[k1,k2]>acc_test):
				acc_test=Accuracy[k1,k2]
				C_opt=C_test[k1]
				nu_opt=nu_test[k2]

	print("NuSVR, Parametres optimaux: C=",C_opt," nu=",nu_opt)
		
	return C_opt, nu_opt




def RFE(X,y,K,C_test):
	# On calcul le paramètre optimal
	C_opt=cv_SVM_linear(X,y,K,C_test)
	
	# On crée le SVM linéaire
	svc = SVC(kernel="linear",C=C_opt)
	
	# On coupe les échantillons en 2
	Xcv, Ycv=create_cv_set(X,y,2)	
	X_train, y_train, X_test, y_test=create_train_set(Xcv,Ycv,1)
	
	# Nombre d'attributs
	nb_att_tmp=X.shape[1]
	# liste contenant les précisions
	res=[]
	while nb_att_tmp !=0:
		#On entraine le SVM
		svc.fit(X_train, y_train)
            
		#On recupère le score de bonne classification
		y_pred = svc.predict(X_test)
		res_tmp= f1_score(y_test, y_pred)
		
		res = np.append(res_tmp,res)
		
		#On cherche la variable ayant l importance la plus faible
		w=svc.coef_
		if(nb_att_tmp==X.shape[1]):
			w_init=w
		# A noter : dans le cas ou l'on a que 2 classes, w est un vecteur,
		# autrement, w est une matrice
		tt=np.sum(np.abs(w),axis=0)
		idMin=tt.argmin() # variable ayant la plus faible importance
		
		#On la retire
		X_train=np.delete(X_train,idMin,1)
		X_test=np.delete(X_test,idMin,1)
		nb_att_tmp -=1

	acc=0
	n_feat_opt=0;
	for i in range(len(res)):
		if(res[i]>acc):
			n_feat_opt=i+1
			acc=res[i]

	print("Dimension du W initial",w_init.shape)

	print("Nombre d attributs optimal :",n_feat_opt)
	print("Meilleur f1 score :",acc)
	
	plt.figure()
	plt.xlabel("Nombre d attributs selectionnes")
	plt.ylabel("Taux de bonne classification")
	plt.plot(np.linspace(1,len(res),len(res)), res)
	plt.show()
	
	return n_feat_opt

def prune_dataset(Xo,yo,npc):
	# Réduit la taille des données d'aprentissage
	print("Pruning dataset")
	nc=2
	N=Xo.shape[0]
	
	X=[]
	y=[]
	
	for c in range(nc):
		Xtemp=[]
		ytemp=[]
		for n in range(N):
			if(yo[n]==c):
				x=Xo[n,:]
				Xtemp.append(x)
				ytemp.append(c)
		r=np.random.choice(range(len(Xtemp)), npc, replace = False)
		
		
		for k in range(npc):
			X.append(Xtemp[r[k]])
			y.append(ytemp[r[k]])
		
	X=np.array(X)
	y=np.array(y)
	
	return X,y

def prune_dataset2(Xo,yo,npc):
	# Réduit la taille des données d'aprentissage (2 classes)
	print("Pruning dataset")
	nc=2
	N=Xo.shape[0]
	
	X=[]
	y=[]
	
	for c in range(nc):
		Xtemp=[]
		ytemp=[]
		for n in range(N):
			if(yo[n]==c):
				x=Xo[n,:]
				Xtemp.append(x)
				ytemp.append(c)
		r=np.random.choice(range(len(Xtemp)), npc[c], replace = False)
		
		
		for k in range(npc[c]):
			X.append(Xtemp[r[k]])
			y.append(ytemp[r[k]])
		
	X=np.array(X)
	y=np.array(y)
	
	return X,y