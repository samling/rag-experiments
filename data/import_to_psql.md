### Connect to db

```bash
# from somewhere with local access to csv

psql -U postgres -h postgresql.postgres -d monarch`
```

### Create table
```bash
CREATE TABLE transactions (
    date DATE NOT NULL,
    merchant VARCHAR(255),
    category VARCHAR(255),
    account VARCHAR(255),
    original_statement TEXT,
    notes TEXT,
    amount NUMERIC(10, 2),
    tags TEXT
);
```

### Modify CSV
Make sure headers match case of table columns.

### Import CSV

```bash
\copy transactions FROM '/path/to/transactions.csv' DELIMITER ',' CSV HEADER;
```
