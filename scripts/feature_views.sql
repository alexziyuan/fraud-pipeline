-- Aggregated features per originating account
CREATE OR REPLACE VIEW v_account_features AS
SELECT
    name_orig,
    COUNT(*)                                         AS tx_count,
    AVG(amount)                                      AS avg_amount,
    MAX(amount)                                      AS max_amount,
    SUM(CASE WHEN type = 'TRANSFER' THEN 1 ELSE 0 END) AS transfer_count,
    SUM(CASE WHEN type = 'CASH_OUT' THEN 1 ELSE 0 END) AS cashout_count,
    SUM(CASE WHEN old_balance_orig = 0 THEN 1 ELSE 0 END) AS zero_balance_starts
FROM transactions_raw
GROUP BY name_orig;

-- Transaction-level feature table (what the model will actually train on)
CREATE OR REPLACE VIEW v_features AS
SELECT
    t.step,
    t.type,
    t.amount,
    t.old_balance_orig,
    t.new_balance_orig,
    t.old_balance_dest,
    t.new_balance_dest,
    -- Engineered features
    t.old_balance_orig - t.new_balance_orig          AS balance_diff_orig,
    t.new_balance_dest - t.old_balance_dest          AS balance_diff_dest,
    CASE WHEN t.old_balance_orig = 0 THEN 1 ELSE 0 END AS orig_zero_start,
    CASE WHEN t.new_balance_orig = 0 THEN 1 ELSE 0 END AS orig_zero_end,
    CASE WHEN t.amount > a.avg_amount * 3 THEN 1 ELSE 0 END AS is_high_amount,
    a.tx_count                                        AS account_tx_count,
    a.cashout_count                                   AS account_cashout_count,
    t.is_fraud
FROM transactions_raw t
LEFT JOIN v_account_features a ON t.name_orig = a.name_orig
WHERE t.type IN ('TRANSFER', 'CASH_OUT');  -- fraud only occurs in these types