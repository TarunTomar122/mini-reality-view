# Long bet swap (Fable)
1. Pool: all 48 drives, four s values; training draws s ~ U[0, 0.7], and 12 drives are at s = 0.3.
2. Refresh: 8k / 12k / 16k; training never waits, a late cache means keep the old one.
3. Stop: apply both. Astra's F1 rule covers mine; keep my late-drift rule and the 15k smoothness check.
