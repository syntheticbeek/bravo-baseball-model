**Bravo Departure Index**
A logistic regression model for predicting cast departures across Bravo reality franchises.

**Background**

Departure outcomes on Bravo are not random. They follow a pattern and patterns can be modeled.

I defined eight variables, scored them across eight documented scandal cases from four franchises, trained a logistic regression model, 
and applied it to predict outcomes for Amanda Batula and West Wilson following the Summer House Post-Season 10 scandal.

**The Model Results**

96.9% cast departure probability for Amanda Batula


3.4% cast departure probability for West Wilson


Same incident. Same show. Same season. 93.6 pt difference.

**8 Variables Defined:**

Betrayal Index: (1–10) measures how severely the incident violated audience trust specifically, the gap between who the cast member was understood to be and what they actually did.


Innocent Edit Score: (1–10) captures how thoroughly producers had constructed the cast member as sympathetic, victimized, or morally trustworthy before the incident occurred. This is the variable that changes everything.


Cast Currency: (1–10) reflects narrative value to the show (tenure, fan investment, centrality to the storyline).


Castmate Reaction: (Ordinal) encodes whether fellow cast members responded with support, neutrality, or hostility. Producer decisions on Bravo have historically tracked cast consensus.


Incident Timing: (Ordinal) captures when in the production cycle the incident occurred, which determines how much rehabilitation window exists before renewal decisions are made.


Allegations: (Binary) confirmed or denied


Gender: (Binary) male or female


Prior Incident: (Binary) first offense or repeat

**Training Data**

<img width="481" height="413" alt="Screenshot 2026-05-12 at 4 44 06 PM" src="https://github.com/user-attachments/assets/d974c15f-fe28-4bfc-a036-a5a84b5251d2" />



**Predictions: Summer House Season 11**

<img width="315" height="82" alt="Screenshot 2026-05-10 at 9 30 29 PM" src="https://github.com/user-attachments/assets/7f68f0bf-0e79-405a-9313-1d0a7e2455ce" />


**Ablation Analysis**

To isolate what drives the 93.6 point gap (calculated from raw model output), I retrained the model three times 
each time removing one variable.

<img width="390" height="97" alt="Screenshot 2026-05-10 at 9 31 06 PM" src="https://github.com/user-attachments/assets/b2f6d817-9d63-4c94-9ab4-49a9d8ddaa20" />



**Permutation Test**

The outcome labels were randomly shuffled 10,000 times and the model retrained on each permutation. Only 256 of 10,000 random configurations produced a gap as large as the observed 93.6%. The mean gap under random conditions was 40.4%.

P-value: 0.0256

The observed gap is statistically significant at the 0.05 threshold. The finding is unlikely to be a product of random chance despite the small sample size.

**THE UPSHOT:**

***It is not what you did. It is who the audience thought you were.***

*The Innocent Edit Score* is determined based on how thoroughly producers constructed a cast member as a 'victim' to gain audiences' sympathy 
before the cast members negative reputation incident occurs- this is the primary structural determinant of departure outcomes on Bravo programming.


**Methodology**

Logistic regression trained on eight verified cases across four Bravo franchises (Real Housewives of Beverly Hills, Vanderpump Rules, Southern Charm, and Summer House). Cases were selected based on documented scandal involvement and confirmed departure outcomes. All variable scores were assigned by the researcher based on direct observation of available footage and established Bravo narrative conventions.


Want more info? Check out my substack: (https://syntheticbeek.substack.com/p/i-built-a-model-to-predict-bravo)

