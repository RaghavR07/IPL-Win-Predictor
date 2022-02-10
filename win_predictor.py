import pandas as pd
see = pd.read_csv(r"C:\Users\ASUS\Desktop\IPL Project\Training\deliveries.csv")
see.to_csv("deliveries.csv")
delivery=pd.read_csv("deliveries.csv")

see = pd.read_csv(r"C:\Users\ASUS\Desktop\IPL Project\Training\matches.csv")
see.to_csv("matches.csv")
match=pd.read_csv("matches.csv")

total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df=total_score_df[total_score_df['inning']==1]
# print(total_score_df.shape)

match_df=match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')

# print(match_df['team1'].unique())

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]

# print(match_df.shape)
# print(match_df['team1'].unique())

# print(match_df['dl_applied'].value_counts())

match_df=match_df[match_df['dl_applied']==0]
# print(match_df.shape)

match_df=match_df[['match_id','city','winner','total_runs']]
delivery_df=match_df.merge(delivery,on='match_id')

# print(delivery_df)

delivery_df=delivery_df[delivery_df['inning']==2]
# print(delivery_df.shape)

delivery_df['current_score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']
# print(delivery_df['current_score'])

delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']

delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])

delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x=="0" else "1")
delivery_df['player_dismissed']=delivery_df['player_dismissed'].astype('int')
wickets=delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets']=10-wickets

# print(delivery_df.head())

delivery_df['crr']=delivery_df['current_score']/((120-delivery_df['balls_left'])/6)

delivery_df['rrr']=delivery_df['runs_left']/((delivery_df['balls_left'])/6)

def result(row):
    return 1 if row['batting_team']==row['winner'] else 0

delivery_df['result']=delivery_df.apply(result,axis=1)

final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]

# print(final_df.shape)

final_df=final_df.sample(final_df.shape[0])
# print(final_df.sample())

final_df.dropna(inplace=True)

final_df=final_df[final_df['balls_left']!=0]

X=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]

# print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

# print(X_train.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,y_pred))

# print(pipe.predict_proba(X_test)[10])


# CREATING WEB APP USING STREAMLIT


import streamlit as st

st.title('IPL Win Predictor')

col1, col2 = st.beta_columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.beta_columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")