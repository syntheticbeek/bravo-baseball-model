**Bravo Departure Predictor**

A logistic regression model trained on 8 Bravo cast members predicts Amanda Batula's departure probability at 96.9% and West Wilson's at 2.8% despite identical timing, allegation status, and castmate reaction scores.

**The finding**

Cast departure on Bravo is not arbitrary. It follows a measurable pattern driven by a small number of high-signal variables. The most powerful of these is not incident severity, not gender, and not cast value — it is the victim narrative the audience held about the cast member before the incident occurred.
When the victim narrative variable is removed from the model, the gap between Amanda and West collapses from 94.1 points to 35.2. Leave it in and the model separates them with near-total certainty.
It is not what you did. It is who the audience thought you were.

**The Bravo Departure Index**

Eight variables scored for each cast member:

<img width="635" height="490" alt="Screenshot 2026-05-09 at 7 47 46 PM" src="https://github.com/user-attachments/assets/20ce2b0c-3e24-46b4-88a4-07a8b99dd882" />


**Training data**
<img width="641" height="509" alt="Screenshot 2026-05-09 at 7 49 00 PM" src="https://github.com/user-attachments/assets/16ab57dd-520a-43f8-85ca-92bfa2380d04" />

**Predictions: Summer House Season 8**

Both subjects were involved in the same post-filming infidelity scandal. Same timing. Same allegation status. Same castmate reaction.
<img width="639" height="142" alt="Screenshot 2026-05-09 at 7 49 53 PM" src="https://github.com/user-attachments/assets/05230186-d37e-4129-9ac9-a3be47e99bcd" />

**Run it yourself**

git clone https://github.com/syntheticbeek/bravo-departure-model.git
cd bravo-departure-model
pip install -r requirements.txt
python bravo_departure_model.py

To score a new cast member, edit the input variables at the bottom of bravo_departure_model.py.

**Files**

bravo_departure_model.py — clean Python script

bravo_departure_model.ipynb — original Colab notebook

training_cases.csv — full training dataset with all variable scores
