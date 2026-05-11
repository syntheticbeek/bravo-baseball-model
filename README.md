**Bravo Departure Predictor**

A logistic regression model trained on 8 Bravo cast members predicts Amanda Batula's departure probability at 96.9% and West Wilson's at 2.8% despite identical timing, allegation status, and castmate reaction scores.

**The finding**

Cast departure on Bravo is not arbitrary. It follows a measurable pattern driven by a small number of high-signal variables. The most powerful of these is not incident severity, not gender, and not cast value, but the victim narrative the audience held about the cast member before the incident occurred.
When the victim narrative variable is removed from the model, the gap between Amanda and West collapses from 94.1 points to 35.2. Leave it in and the model separates them with near-total certainty.
***It is not what you did. It is who the audience thought you were.***

**The Bravo Departure Index**

Eight variables scored for each cast member:

<img width="630" height="521" alt="Screenshot 2026-05-10 at 7 56 44 PM" src="https://github.com/user-attachments/assets/39e29ccf-6a6e-4bb8-a712-a78239f84ab8" />

**Training data**

<img width="623" height="382" alt="Screenshot 2026-05-10 at 8 04 57 PM" src="https://github.com/user-attachments/assets/6fb90ea0-7673-45ad-9359-e1012906d834" />


**Predictions: Summer House Season 11**

Both subjects were involved in the same post-filming infidelity scandal. Same timing. Same allegation status. Same castmate reaction.
<img width="632" height="111" alt="Screenshot 2026-05-10 at 7 57 46 PM" src="https://github.com/user-attachments/assets/18e8d1a6-33db-4b10-8893-9b705c89e352" />


**Ablation Analysis**


To isolate what drives the 94.1-point gap, I retrained the model three times, and removed one variable each time.

<img width="340" height="165" alt="Screenshot 2026-05-10 at 8 06 04 PM" src="https://github.com/user-attachments/assets/74ba2be0-cd48-4a5a-b32e-ba7a8f52471e" />


Removing gender changes almost nothing (−0.1 pts). Removing the betrayal index reduces the gap by 11.4 points. Removing the victim narrative variable collapses the gap by 58.9 points — nearly two thirds of the model's total predictive power.

***It is not what you did. It is who the audience thought you were.***

The Innocent Edit Score: how thoroughly producers constructed a cast member as sympathetic before the incident, is the primary structural determinant of departure outcomes on Bravo programming.

**Score a New Cast Member**


Want to apply the model to a different franchise or cast member? Clone the repo and edit the scoring inputs in bravo_departure_model.ipynb. Each variable is documented inline.


**Methodology**


Logistic regression trained on 8 verified cases. Franchises selected for documented scandal involvement and confirmed departure outcomes (RHOBH, VPR, Southern Charm, Summer House). Model validated by classification of all training cases before being applied to Summer House Season 11 predictions.


**Files**

bravo_departure_model.py — clean Python script

bravo_departure_model.ipynb — original Colab notebook

training_cases.csv — full training dataset with all variable scores
