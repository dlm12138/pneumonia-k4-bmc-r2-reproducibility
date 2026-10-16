# Locked K4 model and access status

K=4 was fitted in the historical MIMIC-IV derivation population, not re-fitted in the revised strict pneumonia application cohorts. Features comprise SBP, DBP, HR and RR measured in each of twelve hourly bins: four vital signs × twelve hours = 48 dimensions. Feature order is vital-block order SBP, DBP, HR, RR, with h0 through h11 within each block.

Observation is [0,12) hours after ICU admission; the landmark is 12 hours. Primary assignability requires at least 32/48 observed dimensions and at least one observation of each of the four vital signs. Coverage sensitivity thresholds are 24, 28, 32, 36 and 40.

External assignment uses the original derivation standardisation and frozen mixture, not external re-standardisation or refitting. Observed-dimension marginal Gaussian likelihood integrates out unavailable dimensions. The highest posterior component is mapped to the original P1–P4 labels. P2 is the unchanged reference group.

`code/core/locked_k4.py` contains the actual assignment-only implementation. It contains no fitted parameters and no training operation. The early-assignment and fixed-mask functions likewise calculate with supplied assets rather than fit a mixture.

Excluded assets include the original standardisation means/SDs, component weights, component means, covariance matrices, original numerical component-to-label mapping files and model binaries. No operational approved distribution channel exists for these assets. No PhysioNet companion project application or approval is claimed.

Appropriately credentialed researchers may contact the corresponding author using the contact information in the manuscript to ask about a compliant reproduction arrangement, stating their proposed use and access qualifications. This is a consultation route, not a guarantee of transfer. The authors must first determine applicable permissions and an approved distribution mechanism. No parameter-download URL or automatic entitlement is available.

Consequently, the present package provides inspectable model mathematics and downstream code, **not a fully publicly available locked numerical model**. Exact new-patient assignment and parameter-dependent diagnostics remain unavailable from public files alone. The documented model-access request remains only partially addressed.
