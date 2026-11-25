from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(OUT/'appointments_clean.csv')
appointment_ids=df['appointment_id'].copy()
X=df[['age','distance_km','fee_inr','lead_days','prior_no_shows','reminder_sent','booking_channel','visit_type']].copy()
X['reminder_sent']=X['reminder_sent'].map({'Yes':1,'No':0})
X=pd.get_dummies(X,columns=['booking_channel','visit_type'],dtype=int)
y=df['no_show'].astype(int)
X_train,X_test,y_train,y_test,ids_train,ids_test=train_test_split(X,y,appointment_ids,test_size=0.2,random_state=42)
print('FEATURES',X.columns.tolist())
print('X shape',X.shape,'train',X_train.shape,'test',X_test.shape,'test no-shows',int(y_test.sum()),'baseline',accuracy_score(y_test,np.zeros(len(y_test),dtype=int)))

# 3.2 Logistic Regression with train-only scaling
scaler=StandardScaler(); X_train_s=scaler.fit_transform(X_train); X_test_s=scaler.transform(X_test)
lr=LogisticRegression(max_iter=1000).fit(X_train_s,y_train)
lr_probs=lr.predict_proba(X_test_s)[:,1]; lr_pred=(lr_probs>=0.5).astype(int)
lr_metrics={'accuracy':accuracy_score(y_test,lr_pred),'confusion_matrix':confusion_matrix(y_test,lr_pred).tolist(),'precision':precision_score(y_test,lr_pred,zero_division=0),'recall':recall_score(y_test,lr_pred,zero_division=0),'f1':f1_score(y_test,lr_pred,zero_division=0),'roc_auc':roc_auc_score(y_test,lr_probs)}
coefs=pd.DataFrame({'feature':X.columns,'coefficient_scaled':lr.coef_[0]})
rem_idx=list(X.columns).index('reminder_sent')
coef=lr.coef_[0][rem_idx]; scale=scaler.scale_[rem_idx]
lr_reminder_odds_sd=float(np.exp(coef)); lr_reminder_odds_yes_vs_no=float(np.exp(coef/scale))
coefs.to_csv(OUT/'logistic_coefficients.csv',index=False)
print('LOGISTIC',json.dumps(lr_metrics,indent=2)); print('COEFFICIENTS\n',coefs.to_string(index=False)); print('Reminder OR per SD',lr_reminder_odds_sd,'No->Yes',lr_reminder_odds_yes_vs_no,'scale',scale)

# 3.3 Trees
full_tree=DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
depth_tree=DecisionTreeClassifier(random_state=42,max_depth=3).fit(X_train,y_train)
tree_metrics={'full_depth_train_accuracy':accuracy_score(y_train,full_tree.predict(X_train)),'full_depth_test_accuracy':accuracy_score(y_test,full_tree.predict(X_test)),'full_depth_actual_depth':full_tree.get_depth(),'depth3_train_accuracy':accuracy_score(y_train,depth_tree.predict(X_train)),'depth3_test_accuracy':accuracy_score(y_test,depth_tree.predict(X_test))}
print('TREE',json.dumps(tree_metrics,indent=2))

# 3.4 Random forest
rf=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train)
rf_probs=rf.predict_proba(X_test)[:,1]; rf_pred=rf.predict(X_test)
rf_metrics={'accuracy':accuracy_score(y_test,rf_pred),'recall':recall_score(y_test,rf_pred,zero_division=0),'roc_auc':roc_auc_score(y_test,rf_probs)}
importances=pd.DataFrame({'feature':X.columns,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
print('RF',json.dumps(rf_metrics,indent=2)); print('TOP IMPORTANCES\n',importances.head(3).to_string(index=False))

# 3.5 cross-validation and threshold selection, training set only
param_grid={'n_estimators':[100,200],'max_depth':[3,5,8]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5,scoring='roc_auc',n_jobs=-1).fit(X_train,y_train)
tuned=grid.best_estimator_
lr_pipe=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000))])
models={'logistic_regression':lr_pipe,'decision_tree_depth3':depth_tree,'random_forest':rf,'tuned_random_forest':tuned}
cv_scores={name:float(cross_val_score(model,X_train,y_train,cv=5,scoring='roc_auc',n_jobs=-1).mean()) for name,model in models.items()}
# choose highest training-only mean CV ROC AUC
winner_name=max(cv_scores,key=cv_scores.get); winner=models[winner_name]
oof=cross_val_predict(winner,X_train,y_train,cv=5,method='predict_proba',n_jobs=-1)[:,1]
threshold_results=[]
for threshold in [0.5,0.35]:
    pred=(oof>=threshold).astype(int)
    threshold_results.append({'threshold':threshold,'precision':precision_score(y_train,pred,zero_division=0),'recall':recall_score(y_train,pred,zero_division=0),'f1':f1_score(y_train,pred,zero_division=0),'flagged':int(pred.sum())})
chosen=max(threshold_results,key=lambda x:x['f1'])['threshold']
# final fit on all training rows, single test evaluation
winner.fit(X_train,y_train)
test_probs=winner.predict_proba(X_test)[:,1]
test_flags=(test_probs>=chosen).astype(int)
final_metrics={'selected_model':winner_name,'cv_auc_scores':cv_scores,'grid_best_params':grid.best_params_,'grid_best_cv_auc':float(grid.best_score_),'threshold_comparison':threshold_results,'selected_threshold':chosen,'test_precision':precision_score(y_test,test_flags,zero_division=0),'test_recall':recall_score(y_test,test_flags,zero_division=0),'test_f1':f1_score(y_test,test_flags,zero_division=0),'test_roc_auc':roc_auc_score(y_test,test_probs),'test_flagged':int(test_flags.sum()),'test_rows':len(y_test)}
print('GRID BEST',grid.best_params_,grid.best_score_)
print('CV SCORES',json.dumps(cv_scores,indent=2))
print('THRESHOLD RESULTS',json.dumps(threshold_results,indent=2))
print('FINAL',json.dumps(final_metrics,indent=2))

risk=pd.DataFrame({'appointment_id':ids_test.values,'no_show':y_test.values,'risk_prob':test_probs,'flag':test_flags})
risk.to_csv(OUT/'test_risk_scores.csv',index=False)
summary={'feature_columns':X.columns.tolist(),'train_rows':len(X_train),'test_rows':len(X_test),'test_no_shows':int(y_test.sum()),'baseline_accuracy':float(accuracy_score(y_test,np.zeros(len(y_test),dtype=int))),'logistic_metrics':lr_metrics,'reminder_odds_ratio_per_sd':lr_reminder_odds_sd,'reminder_odds_ratio_no_to_yes':lr_reminder_odds_yes_vs_no,'tree_metrics':tree_metrics,'random_forest_metrics':rf_metrics,'rf_top_features':importances.head(3).to_dict('records'),'model_selection':final_metrics}
(OUT/'part3_summary.json').write_text(json.dumps(summary,indent=2))
