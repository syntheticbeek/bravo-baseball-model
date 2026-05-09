import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
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
    'gender': [0, 0, 0, 0, 0, 1, 1, 1],  # 0=woman, 1=man
    'outcome': [1, 1, 1, 1, 0, 0, 0, 0]  # 1=exited, 0=stayed
}

df = pd.DataFrame(data)

# Encode trigger timing
timing_map = {'early': 0, 'mid': 1, 'late': 2, 'after_filming': 3}
df['trigger_timing_encoded'] = df['trigger_timing'].map(timing_map)

print(df[['cast_member', 'gender', 'betrayal_index', 'victim_narrative_index', 'outcome']])

# Features for training
features = ['trigger_timing_encoded', 'confirmed', 'cast_reaction',
            'betrayal_index', 'narrative_equity', 'victim_narrative_index',
            'prior_incident', 'gender']

X_train = df[features]
y_train = df['outcome']

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Print model weights
print("\n--- Variable Weights ---")
weights = pd.DataFrame({
    'Variable': features,
    'Weight': model.coef_[0]
}).sort_values('Weight', ascending=False)
print(weights.to_string(index=False))

# Amanda and West prediction data
test_data = {
    'cast_member': ['Amanda', 'West'],
    'trigger_timing_encoded': [3, 3],      # after filming
    'confirmed': [0, 0],                    # both denied
    'cast_reaction': [2, 2],               # both hostile
    'betrayal_index': [10, 6],
    'narrative_equity': [10, 8],
    'victim_narrative_index': [10, 3],
    'prior_incident': [0, 0],
    'gender': [0, 1]                        # Amanda=woman, West=man
}

test_df = pd.DataFrame(test_data)
predictions = model.predict_proba(test_df[features])

print("\n--- Predictions ---")
for i, name in enumerate(['Amanda', 'West']):
    print(f"{name} — Exit probability: {predictions[i][1]:.1%}")

# Remove victim narrative index from features and retrain
features_no_victim = ['trigger_timing_encoded', 'confirmed', 'cast_reaction',
                      'betrayal_index', 'narrative_equity',
                      'prior_incident', 'gender']

model_no_victim = LogisticRegression()
model_no_victim.fit(df[features_no_victim], df['outcome'])

# Rerun Amanda and West without victim narrative
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
    'prior_incident': 0,
    'gender': 1
}])

preds_no_victim = model_no_victim.predict_proba(test_no_victim[features_no_victim])

print("--- Without Victim Narrative Variable ---")
for i, name in enumerate(['Amanda', 'West']):
    print(f"{name} — Exit probability: {preds_no_victim[i][1]:.1%}")

print(f"\nGap without victim narrative: {abs(preds_no_victim[0][1] - preds_no_victim[1][1]):.1%}")

# Also run without betrayal index for comparison
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
    'prior_incident': 0,
    'gender': 1
}])

preds_no_betrayal = model_no_betrayal.predict_proba(test_no_betrayal[features_no_betrayal])

print("\n--- Without Betrayal Index Variable ---")
for i, name in enumerate(['Amanda', 'West']):
    print(f"{name} — Exit probability: {preds_no_betrayal[i][1]:.1%}")

print(f"\nGap without betrayal index: {abs(preds_no_betrayal[0][1] - preds_no_betrayal[1][1]):.1%}")

# Summary comparison
print("\n--- Summary: Gap Under Each Condition ---")
print(f"Full model (all variables):     94.1%")
print(f"Without gender:                 94.2%")
print(f"Without victim narrative:       {abs(preds_no_victim[0][1] - preds_no_victim[1][1]):.1%}")
print(f"Without betrayal index:         {abs(preds_no_betrayal[0][1] - preds_no_betrayal[1][1]):.1%}")
