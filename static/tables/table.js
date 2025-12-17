const path = require('path');
const sqlite3 = require('sqlite3').verbose();

// Allow overriding via env; default to repo-root armor.db relative to this script.
const dbPath =
  process.env.ARMOR_DB_PATH ||
  path.resolve(__dirname, '..', '..', 'armor.db');
const db = new sqlite3.Database(dbPath);

db.all("SELECT * FROM armor_records", [], (err, rows) => {
    if (err) {
        throw err;
    }
    rows.forEach((row) => {
        console.log(row);
    });
});

db.close();
