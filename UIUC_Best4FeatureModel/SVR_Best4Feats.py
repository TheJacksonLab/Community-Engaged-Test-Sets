import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.inspection import permutation_importance

df = pd.read_csv("train_OnlyBest4Feature.csv") #molecules with data must be at the top
molecs = 42 #40 T80 measurements
data_labels = np.array(df.columns.values[3:250])
all_features = np.array(df.iloc[:molecs,3:250])
T80 = np.array(df.iloc[:molecs,1])

#Scale X to unit variance and zero mean
st = StandardScaler()
X = st.fit_transform(all_features)
feats = X.shape[1]
y = T80
"""
C_reg = np.array([10,30,100,1000]) #reg. strength = 1/C_reg
C_regs = C_reg.shape[0]
MSEs = np.zeros(C_regs)
R2s = np.zeros(C_regs)
top_y_pred = np.zeros((feats**4,molecs))
top_C_reg = np.zeros(feats**4)
top_R2s = np.zeros(feats**4)
out = "FeatID,FeatA,FeatB,FeatC,FeatD,LOOV_R2,Creg\n"

#Select all 4-feature combinations
for k in range(feats):
    for l in range(k+1,feats):
        for m in range(l+1,feats):
            for n in range(m+1,feats):
                y_pred = np.zeros((C_regs,molecs))
                del_feats = np.arange(0,feats,dtype=int)
                del_feats = np.delete(del_feats,k,0)
                del_feats = np.delete(del_feats,l-1,0)
                del_feats = np.delete(del_feats,m-2,0)
                del_feats = np.delete(del_feats,n-3,0)
                X_temp = np.delete(X,del_feats,1) #4 unique features
                
                #LOOV over selected regularization strengths
                for j in range(C_regs):
                    for i in range(molecs):
                        X_train = np.delete(X_temp,i,0)
                        X_test = [X_temp[i]]
                        y_train = np.delete(y,i,0)
                        y_test = y[i]
                        
                        svr = SVR(kernel="rbf", C=C_reg[j])
                        svr.fit(X_train,y_train)
                        y_pred[j,i] = svr.predict(X_test)
                    MSEs[j] = mean_squared_error(y,y_pred[j])
                    R2s[j]  = r2_score(y,y_pred[j])
        
                #Select and store best regularization strength results
                bestC = 0
                for i in range(len(MSEs)):
                    if MSEs[i] < MSEs[bestC]:
                        bestC = i
                featID = k*feats**3+l*feats**2+m*feats**1+n
                top_y_pred[featID] = y_pred[bestc]
                top_c_reg[featid] = c_reg[bestc]
                top_r2s[featid] = r2s[bestc]
                print("{} r2={:.2f}, creg={:.0f}".format(featid,r2s[bestc],c_reg[bestc]))
                out += "{},{},{},{},{},{:.2f},{:.0f}\n".format(featid,data_labels[k],data_labels[l],data_labels[m],data_labels[n],r2s[bestc],c_reg[bestc])

#find the best features
bestf = np.argmax(top_r2s)
bestfeata = int(np.floor(bestf/feats**3))
bestf -= bestfeata*feats**3
bestfeatb = int(np.floor(bestf/feats**2))
bestf -= bestfeatb*feats**2
bestfeatc = int(np.floor(bestf/feats**1))
bestf -= bestfeatc*feats**1
bestfeatd = bestf
bestf = np.argmax(top_r2s)

print("best model is {}, r2 = {:.2f}".format(bestf,np.amax(top_r2s)))

fileo = open("fourfeaturemodels.csv",'w')
fileo.write(out)
fileo.close()

## plot best model performance ##
plt.rcparams['font.family'] = "sans-serif"
plt.rcparams['font.sans-serif'] = "arial"
fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(y, top_y_pred[bestf], color='mediumvioletred')
ax.plot([0,100],[0,100],color='gray')
ax.annotate("$r^2$ = {:.2f}".format(top_r2s[bestf]), xy=(10,80),size=15)
ax.annotate("$c_r$ = {:.0f}".format(top_c_reg[bestf]), xy=(10,60),size=15)
ax.set_ylabel("predicted t80",fontsize='x-large');
ax.set_xlabel("actual t80",fontsize='x-large');
fig.tight_layout()
fig.savefig("top4featuremodel_{}.png".format(bestf))
plt.close()
"""

df = pd.read_csv("test_onlybest4feature.csv") #molecules with data must be at the top
data_labels = np.array(df.columns.values[3:250])
all_features = np.array(df.iloc[:,3:250])
t80 = np.array(df.iloc[:,1])

#scale x to unit variance and zero mean
X_val = st.transform(all_features)
feats = X_val.shape[1]
y_val = t80

svr = SVR(kernel="rbf", C=100)
svr.fit(X,y)
y_val_pred = svr.predict(X_val)
print(y_val_pred)
print(mean_squared_log_error(y_val, y_val_pred))

