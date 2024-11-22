### Create db and import CSV
```bash
sqlite3 monarch.db <<EOS
> .mode csv
> .import /path/to/transactions.csv transactions
> EOS
```

### Connect to DB
```bash
sqlite monarch.db
```
