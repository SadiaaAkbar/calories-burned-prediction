# Case Study: Calories Burned Predictor

**The problem.** Most consumer fitness apps and wearables estimate calories
burned using a crude formula based on heart rate and duration alone (e.g. a
flat "calories/minute" multiplier). This ignores factors that meaningfully
change energy expenditure — body composition, workout type, and how hard the
heart is actually working relative to the person's own resting/max range.
The result is estimates that are often noticeably off for anyone outside the
"average" body the formula was tuned for, which erodes trust in the app and
can mislead users tracking calorie balance for weight goals.

**The approach.** I framed this as a supervised regression problem: given a
person's body stats, heart-rate profile, and workout details, predict total
session calorie burn. Beyond the raw features, I engineered heart-rate-reserve
and intensity ratios (borrowed from exercise physiology, where "% of heart
rate reserve used" is a standard measure of workout intensity) and a
weight × duration interaction term to capture total mechanical work. I
trained and cross-validated six models spanning linear, regularized-linear,
and tree/kernel-based approaches, so the final choice was evidence-based
rather than assumed.

**The result.** A Gradient Boosting model outperformed the alternatives,
explaining ~95% of the variance in calories burned with an average error of
about 60 kcal per session — roughly the calorie difference of a 5-minute
walk. Session duration and average heart rate emerged as the dominant
drivers, confirming the model learned physiologically sensible relationships
rather than spurious patterns.

**The real-world value.** A model like this is directly usable by fitness
apps, gyms, and personal trainers as a lightweight, personalized calorie
estimator that doesn't require a dedicated wearable — just a few inputs a
user already knows or can measure with a basic heart-rate strap. For a gym
chain or fitness-app product team, this kind of feature increases the
perceived accuracy and value of their calorie tracking (a top driver of user
retention in fitness apps), and the same pipeline generalizes easily: swap in
a company's own session-log data and retrain to get a model tuned to their
actual user base. Because it's deployed as a simple, fast Streamlit app, it
also serves as a working prototype a product or data science team could point
to when scoping a production feature — de-risking the "will this actually
work" question before committing engineering resources to a full
wearable-integrated system.
