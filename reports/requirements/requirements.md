# Case Anti-Fraud 3 — Requirements

## Target
- target_column: `is_fraud` (present only in `train_users.csv`)
- fraud_definition: користувач, позначений як fraud у `is_fraud = 1` (визначення в кейсі не деталізоване, використовуємо мітку з датасету)
- aggregation_level: user

## Time Period
- date_range: не вказано у PDF (timestamps у UTC)

## Tables
- train_transactions.csv: історія транзакцій користувачів (transaction-level)
- test_transactions.csv: історія транзакцій користувачів (transaction-level)
- train_users.csv: параметри користувачів + мітка `is_fraud`
- test_users.csv: параметри користувачів без `is_fraud`

## Join Keys
- user_id: `id_user` (ключ для join users ↔ transactions)
- transaction_id: окремого id транзакції в описі немає

## Validation / Filters
- valid_transaction_rules: не задано чітких правил фільтрації у PDF; є поля `status` (success/fail) та `error_group` для неуспішних транзакцій
- exclusions: не задано
- leakage_fields: мітка `is_fraud` не має використовуватись як фіча

## Metrics
- evaluation_metrics:
  - F1-score на тестовій частині (60% ваги оцінки)
  - топ-5 ключових фіч/правил з поясненням (20%)
  - обґрунтування інтеграції рішення в бізнес-процес (20%)

## Output
- result_file: один `.csv` файл з двома полями:
  - `id_user` — всі id з `test_users.csv`
  - `is_fraud` — прогноз (0/1)
- додатково: Notion-док з описом рішення та відповідями на критерії

## Notes
- статуси транзакцій: `success` / `fail`
- типи транзакцій: `card_init`, `card_recurring`, `google-pay`, `apple-pay`, `resign`
- error_group для неуспішних: `fraud`, `antifraud`, `3ds error`, `insufficient funds error`, `do not honor`, `card problem`, `cvv error`, `issuer decline`, `invalid data`, `expired error`, інші
- валюта: `USD`, `EUR`, ...
- бренди карт: `VISA`, `MASTERCARD`, `AMEX`, ...
- типи карт: `DEBIT`, `CREDIT`, `PREPAID`
- регіони: `card_country`, `payment_country`, `reg_country`
