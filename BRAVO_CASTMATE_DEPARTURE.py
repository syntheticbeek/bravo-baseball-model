import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Build your training dataset
data = {
    'cast_member': [
        'Denise Richards', 'Raquel Leviss', 'Olivia Flowers',
        'Taylor Ann Green', 'Kyle Richards',
        'Tom Sandoval', 'Austen Kroll', 'Whitney Sudler-Smith'
    ],
    'show': [
        'RHOBH', 'VPR', 'Southern Charm',
        'Southern Charm', 'RHOBH',
        'VPR', 'Southern Charm', 'Southern Charm'
    ],
    'trigger_timing': [
        'mid', 'late', 'early',
        'early', 'early',
        'late', 'early', 'mid'
    ],
    'confirmed': [0, 0, 1, 0, 0, 1, 0, 0],
    'cast_reaction': [2, 2, 1, 2, 1, 2, 1, 1],
    'betrayal_index': [8, 10, 6.5, 8, 5, 10, 7, 6],
    'narrative_equity': [7, 7, 7, 8, 10, 10, 8, 5],
    'victim_narrative_index': [8, 8, 7, 9, 8, 1, 1, 3],
    'prior_incident': [0, 0, 1, 0, 0, 1, 1, 0],
    'gender': [0, 0, 0, 0, 0, 1, 1, 1],
    'outcome': [1, 1, 1, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

timing_map = {'early': 0, 'mid': 1, 'late': 2, 'after_filming': 3}
df['trigger_timing_encoded'] = df['trigger_timing'].map(timing_map)

print(df[['cast_member', 'gender', 'betrayal_index', 'victim_narrative_index', 'outcome']])

features = ['trigger_timing_encoded', 'confirmed', 'cast_reaction',
            'betrayal_index', 'narrative_equity', 'victim_narrative_index',
            'prior_incident', 'gender']

X_train = df[features]
y_train = df['outcome']

model = LogisticRegression()
model.fit(X_train, y_train)

print("\n--- Variable Weights ---")
weights = pd.DataFrame({
    'Variable': features,
    'Weight': model.coef_[0]
}).sort_values('Weight', ascending=False)
print(weights.to_string(index=False))

# Amanda and West — West prior_incident corrected to 1 (Ciara breakup)
test_data = {
    'cast_member': ['Amanda', 'West'],
    'trigger_timing_encoded': [3, 3],
    'confirmed': [0, 0],
    'cast_reaction': [2, 2],
    'betrayal_index': [10, 6],
    'narrative_equity': [10, 8],
    'victim_narrative_index': [10, 3],
    'prior_incident': [0, 1],              # West = 1
    'gender': [0, 1]
}

test_df = pd.DataFrame(test_data)
predictions = model.predict_proba(test_df[features])

print("\n--- Predictions ---")
for i, name in enumerate(['Amanda', 'West']):
    print(f"{name} — Exit probability: {predictions[i][1]:.1%}")

# Without victim narrative
features_no_victim = ['trigger_timing_encoded', 'confirmed', 'cast_reaction',
                      'betrayal_index', 'narrative_equity',
                      'prior_incident', 'gender']

model_no_victim = LogisticRegression()
model_no_victim.fit(df[features_no_victim], df['outcome'])

test_no_victim = pd.DataFrame([{
    'trigger_timing_encoded': 3,
    'confirmed': 0,
    'cast_reaction': 2,
    'betrayal_index': 10,
    'narrative_equity': 10,
    'prior_incident': 0,
    'gender': 0
},
{
    'trigger_timing_encoded': 3,
    'confirmed': 0,
    'cast_reaction': 2,
    'betrayal_index': 6,
    'narrative_equity': 8,
    'prior_incident': 1,                   # West = 1
    'gender': 1
}])

preds_no_victim = model_no_victim.predict_proba(test_no_victim[features_no_victim])

print("\n--- Without Victim Narrative Variable ---")
for i, name in enumerate(['Amanda', 'West']):
    print(f"{name} — Exit probability: {preds_no_victim[i][1]:.1%}")
print(f"Gap without victim narrative: {abs(preds_no_victim[0][1] - preds_no_victim[1][1]):.1%}")

# Without betrayal index
features_no_betrayal = ['trigger_timing_encoded', 'confirmed', 'cast_reaction',
                        'narrative_equity', 'victim_narrative_index',
                        'prior_incident', 'gender']

model_no_betrayal = LogisticRegression()
model_no_betrayal.fit(df[features_no_betrayal], df['outcome'])

test_no_betrayal = pd.DataFrame([{
    'trigger_timing_encoded': 3,
    'confirmed': 0,
    'cast_reaction': 2,
    'narrative_equity': 10,
    'victim_narrative_index': 10,
    'prior_incident': 0,
    'gender': 0
},
{
    'trigger_timing_encoded': 3,
    'confirmed': 0,
    'cast_reaction': 2,
    'narrative_equity': 8,
    'victim_narrative_index': 3,
    'prior_incident': 1,                   # West = 1
    'gender': 1
}])

preds_no_betrayal = model_no_betrayal.predict_proba(test_no_betrayal[features_no_betrayal])

print("\n--- Without Betrayal Index Variable ---")
for i, name in enumerate(['Amanda', 'West']):
    print(f"{name} — Exit probability: {preds_no_betrayal[i][1]:.1%}")
print(f"Gap without betrayal index: {abs(preds_no_betrayal[0][1] - preds_no_betrayal[1][1]):.1%}")

# Without gender
features_no_gender = ['trigger_timing_encoded', 'confirmed', 'cast_reaction',
                      'betrayal_index', 'narrative_equity', 'victim_narrative_index',
                      'prior_incident']

model_no_gender = LogisticRegression()
model_no_gender.fit(df[features_no_gender], df['outcome'])

test_no_gender = pd.DataFrame([{
    'trigger_timing_encoded': 3,
    'confirmed': 0,
    'cast_reaction': 2,
    'betrayal_index': 10,
    'narrative_equity': 10,
    'victim_narrative_index': 10,
    'prior_incident': 0
},
{
    'trigger_timing_encoded': 3,
    'confirmed': 0,
    'cast_reaction': 2,
    'betrayal_index': 6,
    'narrative_equity': 8,
    'victim_narrative_index': 3,
    'prior_incident': 1                    # West = 1
}])

preds_no_gender = model_no_gender.predict_proba(test_no_gender[features_no_gender])

print("\n--- Without Gender Variable ---")
for i, name in enumerate(['Amanda', 'West']):
    print(f"{name} — Exit probability: {preds_no_gender[i][1]:.1%}")
print(f"Gap without gender: {abs(preds_no_gender[0][1] - preds_no_gender[1][1]):.1%}")

# Summary — all calculated dynamically
full_gap = abs(predictions[0][1] - predictions[1][1])
gender_gap = abs(preds_no_gender[0][1] - preds_no_gender[1][1])
betrayal_gap = abs(preds_no_betrayal[0][1] - preds_no_betrayal[1][1])
victim_gap = abs(preds_no_victim[0][1] - preds_no_victim[1][1])

print(f"\n--- Summary: Gap Under Each Condition ---")
print(f"Full model (all variables):     {full_gap:.1%}")
print(f"Without gender:                 {gender_gap:.1%}")
print(f"Without betrayal index:         {betrayal_gap:.1%}")
print(f"Without victim narrative:       {victim_gap:.1%}")
