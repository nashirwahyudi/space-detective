import { Pool } from 'pg';

export const pool = new Pool({
  user: process.env.NEXT_PUBLIC_DB_USERNAME,
  host: process.env.NEXT_PUBLIC_DB_SERVER,
  database: process.env.NEXT_PUBLIC_DB_DATABASE,
  password: process.env.NEXT_PUBLIC_DB_PASSWORD,
  port: parseInt(process.env.NEXT_PUBLIC_DB_PORT || '5432', 10),
  ssl: {
    rejectUnauthorized: false, // Required for RDS with SSL
  },
});
